//! Config-in-a-hot-loop demo: `tick` reads a `let`-bound dict every call —
//! the generic JIT pays a hash lookup per `CFG["k"]`, while a specialized
//! compile bakes the frozen dict's entries into constant loads behind a
//! payload-pointer guard.
//!
//! Lanes: interpreter, `jit::compile`, and the full `JitSession` cycle
//! (observe ~500 iterations → freeze the observed dict payload →
//! `compile_with` → reinstall).
//!
//!     cargo run --release -p mimas-jit-test --example specbench

use jit::{FrozenKind, JitSession, ObsTag};
use std::time::Instant;

const SRC: &str = r#"
let CFG = ~{ rate = 2, scale = 3, bias = 1 };
fn tick(i: int) -> int {
    CFG["rate"]! * i + CFG["scale"]! * i + CFG["bias"]!
}
fn run(n: int) -> int {
    let t = 0;
    let i = 0;
    while i < n {
        t += tick(i);
        i += 1;
    }
    t
}
let TEST_VALUE = run(1);
"#;

fn fresh_vm() -> (vm::Vm, compile::Program) {
    let (program, sources) =
        vm::Vm::compile_parts(&[("main", SRC)], jit_test::natives::install).expect("compiles");
    let jitprog = program.clone();
    let mut vm = vm::Vm::new();
    vm.load_prebuilt(program, sources, jit_test::natives::install);
    vm.run().unwrap();
    (vm, jitprog)
}

fn time_call(vm: &mut vm::Vm, n: i64) -> (i64, std::time::Duration) {
    let t0 = Instant::now();
    let v = vm.call::<i64>("run", (n,)).unwrap();
    (v, t0.elapsed())
}

fn main() {
    let n: i64 = 500_000;
    let warm: i64 = 500;

    // -- interpreter ------------------------------------------------------
    let (mut vm, _p) = fresh_vm();
    let (v0, d_interp) = time_call(&mut vm, n);

    // -- generic jit -------------------------------------------------------
    let (mut vm, program) = fresh_vm();
    let j = jit::compile(&program).unwrap();
    vm.install_bc(j.bodies());
    let (v1, d_jit) = time_call(&mut vm, n);
    drop(j);

    // -- observe -> freeze -> specialize -----------------------------------
    let (mut vm, program) = fresh_vm();
    let mut sess = JitSession::observe(&mut vm, &program).unwrap();
    vm.call::<i64>("run", (warm,)).unwrap();
    let mut facts = sess.facts();
    let ptrs: Vec<usize> = facts
        .bodies
        .iter()
        .flat_map(|bf| bf.sites.values())
        .filter(|o| o.tag == ObsTag::Dict && o.ptr != 0)
        .map(|o| o.ptr)
        .collect();
    for &p in &ptrs {
        sess.freeze(&mut facts, p, FrozenKind::Dict);
    }
    sess.specialize(&mut vm, &program, &facts).unwrap();
    let (v2, d_spec) = time_call(&mut vm, n);

    assert_eq!(v0, v1);
    assert_eq!(v1, v2);
    println!("run({n}) = {v0}   (froze {} dict site(s))", ptrs.len());
    println!("interp:      {:>9.1?}", d_interp);
    println!("jit generic: {:>9.1?}   ({:.2}x vs interp)", d_jit, d_interp.as_secs_f64() / d_jit.as_secs_f64());
    println!("jit + facts: {:>9.1?}   ({:.2}x vs generic, {:.2}x vs interp)",
        d_spec,
        d_jit.as_secs_f64() / d_spec.as_secs_f64(),
        d_interp.as_secs_f64() / d_spec.as_secs_f64());
}

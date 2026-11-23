//! Not a gate — a `#[ignore]`d timing probe for "is the JIT actually
//! faster on this machine". Run with:
//!
//!     cargo test -p mimas-jit-test --test it -- speed:: --ignored --nocapture
//!
//! Each case runs the same program through the interpreter and through
//! `install_bc` JIT bodies and prints both wall times. Correctness of
//! the lanes is checked by `parity.rs`; this only reports speed, so it
//! asserts nothing about the ratio — hardware and workload decide that.

use jit_test::natives;
use std::time::Instant;
use vm::Vm;

/// Times `run()` only — each lane gets a fresh `Vm` built from a
/// precompiled `Program`, so compilation is outside the measurement.
fn time_run(
    program: &compile::Program,
    sources: &vm::Sources,
    bodies: Option<Vec<Option<vm::bc::BodyFn>>>,
) -> std::time::Duration {
    let mut vm = Vm::new();
    vm.load_prebuilt(program.clone(), sources.clone(), natives::install);
    if let Some(bodies) = bodies {
        vm.install_bc(bodies);
    }
    let t = Instant::now();
    vm.run().unwrap();
    t.elapsed()
}

fn compare(name: &str, source: &str) {
    let (program, sources) =
        Vm::compile_parts(&[("main", source)], natives::install).expect("program compiles");
    let j = jit::compile(&program).expect("jit compile");
    let plain = time_run(&program, &sources, None);
    let jitted = time_run(&program, &sources, Some(j.bodies()));
    eprintln!(
        "{name}: interp {plain:?}  jit {jitted:?}  ({:.2}x)",
        plain.as_secs_f64() / jitted.as_secs_f64()
    );
}

#[test]
#[ignore = "timing probe, not a gate"]
fn speed_smoke() {
    compare(
        "calls",
        "fn fib(n: int) -> int { if n < 2 { n } else { fib(n - 1) + fib(n - 2) } }
         let r = 0;
         for i in 6 { r = fib(22); }
         let TEST_VALUE = f\"{r}\";",
    );
    compare(
        "floats",
        "let z = 0.0; let s = 0;
         for i in 400000 {
             z = z * 0.9999 + 0.25;
             if z * z > 3.0 { s += 1; z = z * 0.5; }
         }
         let TEST_VALUE = f\"{s}\";",
    );
    compare(
        "format",
        "let s = 0;
         for i in 20000 {
             let t = f\"{i}:{i * 2}\";
             if t == \"x\" { s += 1; }
         }
         let TEST_VALUE = f\"{s}\";",
    );
    compare(
        "short_strs",
        "let s = 0;
         for i in 60000 {
             let t = f\"{i mod 10}!\";
             if t == \"3!\" { s += 1; }
         }
         let TEST_VALUE = f\"{s}\";",
    );
    compare(
        "distinct_strs",
        "let s = 0;
         for i in 200000 {
             let t = f\"v{i}\";
             if t == \"nope\" { s += 1; }
         }
         let TEST_VALUE = f\"{s}\";",
    );
    compare(
        "subscripts",
        "struct C { data: [int] }
         impl C {
             subscript(i: int) -> int {
                 get { self.data[i] }
                 set { self.data[i] = new_value; }
                 inout { yield self.data[i]; }
             }
         }
         let c = C { data = [0, 0, 0, 0, 0, 0, 0, 0] };
         let xs = [0, 0, 0, 0, 0, 0, 0, 0];
         for i in 4000 {
             c[i mod 8] += 1;
             xs[i mod 8] += 1;
             { let p = &xs[i mod 8]; p += 1; };
         }
         let TEST_VALUE = f\"{c.data}|{xs}\";",
    );
}

//! `JitSession` runtime-specialization tests: observe → `facts` →
//! `compile_with` → reinstall, plus hand-built `Facts` for the deopt and
//! frozen-bake paths.
//!
//! Every speculation is *guarded* — wrong facts must cost speed, never
//! correctness — so each test diffs the specialized lane's answers against
//! the plain interpreter's.

use jit::{BodyFacts, Facts, FrozenKind, JitSession, Obs, ObsTag};
use jit_test::natives;
use vm::{Captured, Vm};

/// A `Vm` for `source` plus a second copy of its `Program` — the session
/// borrows the program for facts lookups while the vm owns its own
/// (`compile_parts` output is deterministic, so body indices line up).
fn session_vm(source: &str) -> (Vm, compile::Program) {
    let (program, sources) =
        Vm::compile_parts(&[("main", source)], natives::install).expect("fixture compiles");
    let jitprog = program.clone();
    let mut vm = Vm::new();
    vm.load_prebuilt(program, sources, natives::install);
    (vm, jitprog)
}

/// Dynamic-call monomorphism: `tick` calls `f` through a mutable global, so
/// the callee is an `Op::Call` site — observing `fadd` installs a
/// one-target IC; flipping `f` to `fsub` afterwards must still answer
/// correctly through the guarded miss path.
#[test]
fn call_ic_hit_and_miss() {
    // `pad` takes entry reg 0 — `Vm::call` uses that slot as its return
    // scratch and restores it, so a global written by an injected call must
    // not be the first binding or the write is silently restored away.
    let source = r#"
let pad = 0;
fn fadd(a: int, b: int) -> int { a + b }
fn fsub(a: int, b: int) -> int { a - b }
let f = fadd;
fn flip() { f = fsub; }
fn tick(x: int) -> int { f(x, 2) * 3 }
let TEST_VALUE = tick(5);
"#;
    let (mut vm, program) = session_vm(source);
    vm.run().unwrap();
    let tick_b = program.root.function("tick").unwrap().body.index();

    let mut sess = JitSession::observe(&mut vm, &program).unwrap();
    for _ in 0..4 {
        assert_eq!(vm.call::<i64>("tick", (5i64,)).unwrap(), 21);
    }
    let facts = sess.facts();
    // the profiler saw `tick`'s `Op::Call` site, monomorphic on fadd
    assert_eq!(facts.bodies[tick_b].calls.len(), 1, "one call site observed");
    sess.specialize(&mut vm, &program, &facts).unwrap();

    // IC hit: same answer through the inlined call
    assert_eq!(vm.call::<i64>("tick", (5i64,)).unwrap(), 21);
    // reseat the callee — the IC's body-id guard misses, `mj_call_dyn`
    // runs the real dynamic call
    vm.call::<()>("flip", ()).unwrap();
    assert_eq!(vm.call::<i64>("tick", (5i64,)).unwrap(), 9);
}

/// A polymorphic call site (both targets observed) installs no IC — the
/// body still answers through the generic dynamic-call path.
#[test]
fn polymorphic_call_stays_generic() {
    let source = r#"
fn fadd(a: int, b: int) -> int { a + b }
fn fsub(a: int, b: int) -> int { a - b }
fn apply(sub: bool, x: int) -> int {
    let f = if sub { fsub } else { fadd };
    f(x, 2) * 3
}
let TEST_VALUE = apply(false, 5);
"#;
    let (mut vm, program) = session_vm(source);
    vm.run().unwrap();
    let apply_b = program.root.function("apply").unwrap().body.index();

    let mut sess = JitSession::observe(&mut vm, &program).unwrap();
    for _ in 0..4 {
        assert_eq!(vm.call::<i64>("apply", (false, 5i64)).unwrap(), 21);
        assert_eq!(vm.call::<i64>("apply", (true, 5i64)).unwrap(), 9);
    }
    let facts = sess.facts();
    assert!(facts.bodies[apply_b].calls.is_empty(), "two callees → no IC");
    sess.specialize(&mut vm, &program, &facts).unwrap();

    assert_eq!(vm.call::<i64>("apply", (false, 5i64)).unwrap(), 21);
    assert_eq!(vm.call::<i64>("apply", (true, 5i64)).unwrap(), 9);
}

/// The config-in-hot-loop case: `tick` reads a `let`-bound dict every call
/// via `LoadEntry` + `GetIndex`. Entry facts can't see the dict (the reg's
/// entry value isn't it) — `BodyFacts::sites` records the receiver at the
/// op, so a `freeze`d CFG bakes its entries into the body.
#[test]
fn frozen_config_dict_bakes() {
    let source = r#"
let CFG = ~{ g = 10, k = 4 };
fn tick() -> int { CFG["g"]! * 100 + CFG["k"]! }
let TEST_VALUE = tick();
"#;
    let (mut vm, program) = session_vm(source);
    vm.run().unwrap();
    let tick_b = program.root.function("tick").unwrap().body.index();

    let mut sess = JitSession::observe(&mut vm, &program).unwrap();
    for _ in 0..4 {
        assert_eq!(vm.call::<i64>("tick", ()).unwrap(), 1004);
    }
    let mut facts = sess.facts();
    // both GetIndex sites observed the same Dict payload
    let ptrs: Vec<usize> = facts.bodies[tick_b]
        .sites
        .values()
        .filter(|o| o.tag == ObsTag::Dict && o.ptr != 0)
        .map(|o| o.ptr)
        .collect();
    assert_eq!(ptrs.len(), 2, "two dict-index sites, one stable payload");
    assert!(ptrs.iter().all(|&p| p == ptrs[0]));
    sess.freeze(&mut facts, ptrs[0], FrozenKind::Dict);
    sess.specialize(&mut vm, &program, &facts).unwrap();

    assert_eq!(vm.call::<i64>("tick", ()).unwrap(), 1004);
}

/// Observed entry tags force scalar shadows — and a *wrong* guess must
/// still answer correctly: claim `pick`'s int param was seen `Float` and
/// the emitted body has to deopt through `step` on the tag probe.
#[test]
fn wrong_entry_facts_deopt_not_break() {
    let source = r#"
fn pick(x: int) -> int { x + 1 }
let TEST_VALUE = pick(41);
"#;
    let (program, sources) =
        Vm::compile_parts(&[("main", source)], natives::install).expect("fixture compiles");
    let pick = program.root.function("pick").unwrap().body;
    let param0 = program.chunks[pick].params[0].index();

    let mut facts = Facts::default();
    facts.bodies = vec![BodyFacts::default(); program.chunks.len()];
    let mut entry = vec![Obs::default(); program.chunks[pick].regs as usize];
    entry[param0] = Obs {
        tag: ObsTag::Float, // a lie — `pick` is only ever called with Int
        ptr: 0,
    };
    facts.bodies[pick.index()].entry = entry;

    let j = jit::compile_with(&program, &facts).expect("jit compile_with");
    let mut vm = Vm::new();
    vm.load_prebuilt(program, sources, natives::install);
    vm.install_bc(j.bodies());
    vm.run().unwrap();
    assert_eq!(vm.resolve_name("TEST_VALUE"), Some(Captured::Int(42)));
}

/// End-to-end observed-entry specialization: `step`-shaped bodies get
/// forced `Int` shadows from the profiler's facts — same answers as the
/// interpreter, exercising `JitSession::specialize_now` (the
/// `facts()` + `specialize` convenience path).
#[test]
fn observed_shadows_parity() {
    let source = r#"
fn area(w: int, h: int) -> int { w * h }
fn tick(i: int) -> int { area(i, i + 1) }
let TEST_VALUE = tick(6);
"#;
    let (mut vm, program) = session_vm(source);
    vm.run().unwrap();
    let mut sess = JitSession::observe(&mut vm, &program).unwrap();
    for i in 0..6 {
        assert_eq!(
            vm.call::<i64>("tick", (i as i64,)).unwrap(),
            i as i64 * (i as i64 + 1)
        );
    }
    sess.specialize_now(&mut vm, &program).unwrap();
    for i in 0..6 {
        assert_eq!(
            vm.call::<i64>("tick", (i as i64,)).unwrap(),
            i as i64 * (i as i64 + 1)
        );
    }
}

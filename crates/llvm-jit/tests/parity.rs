//! Mirrors `jit-test`'s parity harness: each fixture runs once on the plain
//! interpreter and once with the LLVM-JIT body table installed — `TEST_VALUE`,
//! the `keep` log, and the error outcome must diff clean. Fixtures are shared
//! with `crates/jit-test/fixtures/`.

use jit_test::natives;
use vm::{Captured, Vm};

fn run(
    source: &str,
    bodies: Option<Vec<Option<vm::bc::BodyFn>>>,
) -> (Option<Captured>, Vec<Captured>, Option<String>) {
    let (program, sources) =
        Vm::compile_parts(&[("main", source)], natives::install).expect("fixture compiles");
    let mut vm = Vm::new();
    vm.load_prebuilt(program, sources, natives::install);
    if let Some(bodies) = bodies {
        vm.install_bc(bodies);
    }
    let err = vm.run().err().map(|e| e.to_string());
    let test_value = vm.resolve_name("TEST_VALUE");
    let kept = vm.fixture::<natives::Kept>().0.borrow().clone();
    (test_value, kept, err)
}

/// Compile the fixture and LLVM-JIT all its bodies. The returned
/// [`mimas_llvm_jit::Jit`] owns the execution engine (and code memory) —
/// callers must keep it alive for the whole `install_bc` run.
fn jit_bodies(source: &str) -> mimas_llvm_jit::Jit {
    let (program, _sources) =
        Vm::compile_parts(&[("main", source)], natives::install).expect("fixture compiles");
    mimas_llvm_jit::compile(&program).expect("llvm-jit compile")
}

#[test]
fn basic_parity() {
    let source = include_str!("../../jit-test/fixtures/basic.mimas");
    let j = jit_bodies(source);
    let bodies = j.bodies();
    assert!(!bodies.is_empty());
    assert!(bodies.iter().all(|b| b.is_some()));
    assert_eq!(run(source, None), run(source, Some(bodies)));
}

#[test]
fn cold_parity() {
    let source = include_str!("../../jit-test/fixtures/cold.mimas");
    let j = jit_bodies(source);
    assert_eq!(run(source, None), run(source, Some(j.bodies())));
}

/// Recursion past the inline-call cap — `mj_call_body`/`mj_call_dyn` hand
/// calls off to `Flow::Call` when the frame stack needs it; both lanes must
/// land the same answer.
#[test]
fn deep_parity() {
    let source = include_str!("../../jit-test/fixtures/deep.mimas");
    let j = jit_bodies(source);
    let (v, k, e) = run(source, Some(j.bodies()));
    assert!(e.is_none(), "llvm-jit deep run errored: {e:?}");
    assert!(matches!(v, Some(Captured::Int(400))));
    assert_eq!(run(source, None), (v, k, e));
}

#[test]
fn mixed_parity() {
    let source = include_str!("../../jit-test/fixtures/mixed.mimas");
    let j = jit_bodies(source);
    assert_eq!(run(source, None), run(source, Some(j.bodies())));
}

/// Sanity that the JIT path is actually being exercised, not silently skipped
/// — a jit-run fixture must produce the right answer, not just *an* answer
/// matching the interpreter's (which would also pass if `install_bc` were a
/// no-op and both runs ran plain).
#[test]
fn jit_run_produces_values() {
    let source = include_str!("../../jit-test/fixtures/basic.mimas");
    let j = jit_bodies(source);
    let (test_value, kept, err) = run(source, Some(j.bodies()));
    assert!(err.is_none(), "llvm-jit run errored: {err:?}");
    assert!(matches!(test_value, Some(Captured::Int(208))));
    assert_eq!(kept.len(), 2);
}

/// Build a Vm for `source`, optionally with JIT bodies installed.
fn vm_for(source: &str, bodies: Option<Vec<Option<vm::bc::BodyFn>>>) -> Vm {
    let (program, sources) =
        Vm::compile_parts(&[("main", source)], natives::install).expect("fixture compiles");
    let mut vm = Vm::new();
    vm.load_prebuilt(program, sources, natives::install);
    if let Some(bodies) = bodies {
        vm.install_bc(bodies);
    }
    vm
}

/// The host's per-entry op budget: an unbounded loop must fault with the same
/// "ran too long" error in both lanes — the JIT body decrements `ops_left`
/// itself, so the *count* it consumes must match the interpreter's.
#[test]
fn op_budget_parity() {
    let source = "let i = 0;\nwhile true { i += 1; }";
    let j = jit_bodies(source);

    let mut plain = vm_for(source, None);
    plain.set_op_budget(50_000);
    let err_plain = plain.run().unwrap_err().to_string();

    let mut jitted = vm_for(source, Some(j.bodies()));
    jitted.set_op_budget(50_000);
    let err_jit = jitted.run().unwrap_err().to_string();

    assert!(err_plain.contains("ran too long"), "{err_plain}");
    assert_eq!(err_plain, err_jit);
    assert_eq!(plain.ops_left(), jitted.ops_left());
}

/// A finite loop consumes the same budget under both lanes — this catches
/// bookkeeping drift (an op double-counted or skipped) that same-result tests
/// would hide.
#[test]
fn op_budget_consumption_parity() {
    let source = "let i = 0;\nwhile i < 100 { i += 1; }\nlet TEST_VALUE = i;";
    let j = jit_bodies(source);

    let mut plain = vm_for(source, None);
    plain.set_op_budget(1_000_000);
    plain.run().unwrap();

    let mut jitted = vm_for(source, Some(j.bodies()));
    jitted.set_op_budget(1_000_000);
    jitted.run().unwrap();

    assert_eq!(plain.ops_left(), jitted.ops_left());
}

/// The dispatch `fuel` window (FUEL ops per `run_dispatch` entry) — a loop far
/// past it forces the body to return `Flow::Next` mid-body and the driver to
/// re-enter at the exact op boundary.
#[test]
fn fuel_window_parity() {
    let source = "let i = 0;\nlet t = 0;\nwhile i < 10000 { i += 1; t += i mod 3; }\nkeep(i);\nlet TEST_VALUE = t;";
    let j = jit_bodies(source);
    assert_eq!(run(source, None), run(source, Some(j.bodies())));
}

/// Cooperative pause: `pause()` latches `state.paused` inside a native; the
/// next op boundary must observe it and hand control back (`run_frame` →
/// `false`), with the body's register shadows fully flushed and the top
/// frame's ip saved — resuming must continue at the right op.
#[test]
fn pause_parity() {
    let source = r#"
let hits = 0;
let i = 0;
while i < 5 {
    i += 1;
    pause();
    keep(i);
    hits += i;
}
let TEST_VALUE = hits;
"#;
    let j = jit_bodies(source);

    let run_frames = |bodies: Option<Vec<Option<vm::bc::BodyFn>>>| {
        let mut vm = vm_for(source, bodies);
        let mut frames = 0;
        loop {
            frames += 1;
            if vm.run_frame().expect("frame runs") {
                break;
            }
            assert!(frames < 100, "program never finished");
        }
        (
            frames,
            vm.resolve_name("TEST_VALUE"),
            vm.fixture::<natives::Kept>().0.borrow().clone(),
        )
    };

    let plain = run_frames(None);
    let jitted = run_frames(Some(j.bodies()));
    assert_eq!(plain, jitted);
    // five pauses → five paused frames plus the finishing one
    assert_eq!(plain.0, 6);
}

/// A `Snapshot` taken mid-run (at a pause boundary) restores verbatim in the
/// other lane's world-view: restore the jit-run snapshot into a fresh jit VM,
/// run to completion, and the tail of the run must match the interpreter's.
#[test]
fn snapshot_restore_parity() {
    let source = r#"
let arr = [1, 2, 3];
let t = 0;
pause();
t = arr[0] + arr[1] + arr[2];
keep(t);
let TEST_VALUE = t;
"#;
    let j = jit_bodies(source);

    // jit lane: run to the pause, snapshot, then finish.
    let mut jit_vm = vm_for(source, Some(j.bodies()));
    assert_eq!(jit_vm.run_frame().unwrap(), false, "first frame pauses");
    let snap = jit_vm.snapshot().expect("snapshot at pause boundary");
    assert!(jit_vm.run_frame().unwrap(), "second frame finishes");
    let jit_tail = (
        jit_vm.resolve_name("TEST_VALUE"),
        jit_vm.fixture::<natives::Kept>().0.borrow().clone(),
    );

    // interpreter lane: same script, snapshot, restore the jit snapshot mid-run,
    // finish — must land the identical result.
    let mut plain_vm = vm_for(source, None);
    assert_eq!(plain_vm.run_frame().unwrap(), false);
    plain_vm.restore(&snap).expect("restore jit snapshot");
    assert!(plain_vm.run_frame().unwrap());
    let plain_tail = (
        plain_vm.resolve_name("TEST_VALUE"),
        plain_vm.fixture::<natives::Kept>().0.borrow().clone(),
    );

    assert_eq!(jit_tail, plain_tail);
    assert!(matches!(plain_tail.0, Some(Captured::Int(6))));
}

/// GC pressure: a loop that allocates fresh arrays/dicts every iteration while
/// scalar shadows stay live in registers — the body's flush-on-alloc keeps the
/// GC window consistent, and `collect_debt` between fuel batches must not
/// collect values the body still holds only in SSA vars... or in the window.
#[test]
fn gc_alloc_parity() {
    let source = r#"
let total = 0;
let i = 0;
while i < 500 {
    let a = [i, i * 2];
    let d = ~{ k = a[1] };
    total += a[0] + d["k"]!;
    i += 1;
}
keep(total);
let TEST_VALUE = total;
"#;
    let j = jit_bodies(source);
    assert_eq!(run(source, None), run(source, Some(j.bodies())));
}

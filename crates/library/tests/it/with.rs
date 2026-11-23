//! `with rng = <spec> { … }` — a scoped install on the Vm's `Rng` fixture.
//! The region draws on the spec's state (an `int` seed, a `[float]` script
//! of unit draws, or `()` to checkpoint the live stream) and the enclosing
//! state+script come back when it ends — at the region's tail normally, or
//! through `Rng::reconcile` when a hard fault unwinds the frame instead.
//! Nested scopes shadow LIFO. What `with` isn't: authority laundering — the
//! desugared `__fx_rng_*` natives are `#[effects(rng)]`, so a region using
//! randomness still carries `rng` in its inferred set.

use vm::{Captured, Vm, fixtures::Rng};

fn run(src: &str) -> Vm {
    let mut vm = Vm::compile(src, library::std).expect("with test compiled");
    vm.run().expect("with test ran");
    vm
}

fn get(vm: &mut Vm, name: &str) -> Captured {
    vm.resolve_name(name).unwrap()
}

// r[verify mimas.fx.with]
#[test]
fn with_seed_replays() {
    let mut vm = run(
        r#"
let a = with rng = 7 { int::random(1000) };
let b = with rng = 7 { int::random(1000) };
let c = with rng = 8 { int::random(1000) };
"#,
    );
    assert_eq!(get(&mut vm, "a"), get(&mut vm, "b"));
    assert_ne!(get(&mut vm, "a"), get(&mut vm, "c"));
}

// r[verify mimas.fx.with]
#[test]
fn with_draws_match_the_same_native_seed() {
    // `with rng = 7`'s stream is `random::seed(7)`'s stream — same
    // xorshift64*, so one seed replays identically either way.
    let mut vm = run("let inside = with rng = 7 { int::random(1000) };");
    let mut native = run("random::seed(7);\nlet outside = int::random(1000);");
    assert_eq!(get(&mut vm, "inside"), get(&mut native, "outside"));
}

// r[verify mimas.fx.with.restore]
#[test]
fn with_restores_the_outer_stream() {
    // the scope draws off its own stream; the enclosing stream resumes at
    // the index it left, untouched by anything the scope drew
    let mut vm = run(
        r#"
random::seed(1);
let before = int::random(1000);
let inside = with rng = 7 { int::random(1000) };
let after = int::random(1000);
"#,
    );
    let mut baseline = run(
        "random::seed(1);\nlet first = int::random(1000);\nlet second = int::random(1000);",
    );
    assert_eq!(get(&mut vm, "before"), get(&mut baseline, "first"));
    assert_eq!(get(&mut vm, "after"), get(&mut baseline, "second"));
    let mut seeded = run("random::seed(7);\nlet only = int::random(1000);");
    assert_eq!(get(&mut vm, "inside"), get(&mut seeded, "only"));
    assert_eq!(
        vm.fixture::<Rng>().with_scopes(),
        0,
        "a with scope outlived its region"
    );
}

// r[verify mimas.fx.with]
#[test]
fn nested_with_shadows_lifo() {
    let mut vm = run(
        r#"
let draws = with rng = 7 {
    let first = int::random(1000);
    let inner = with rng = 2 { int::random(1000) };
    let resumed = int::random(1000);
    [first, inner, resumed]
};
"#,
    );
    let mut s7 = run("random::seed(7);\nlet a = int::random(1000);\nlet b = int::random(1000);");
    let mut s2 = run("random::seed(2);\nlet a = int::random(1000);");
    assert_eq!(
        get(&mut vm, "draws"),
        Captured::Array(vec![
            get(&mut s7, "a"),
            get(&mut s2, "a"),
            get(&mut s7, "b"),
        ]),
    );
}

// r[verify mimas.fx.with]
#[test]
fn with_script_specs_exact_draws() {
    // a `[float]` spec scripts the region's draws; the script runs out and
    // the live stream continues from where it was
    let mut vm = run(
        r#"
random::seed(1);
let first = int::random(1000);
let xs = with rng = [0.5, 0.25] {
    [float::random(1.0), float::random(1.0), float::random(1.0)]
};
let second = int::random(1000);
"#,
    );
    let Captured::Array(xs) = get(&mut vm, "xs") else {
        panic!("xs wasn't an array")
    };
    assert_eq!(xs[0], Captured::Float(0.5));
    assert_eq!(xs[1], Captured::Float(0.25));
    // scripted draws *replace* the value at their position but the stream
    // still advances past them — the third draw is the stream's fourth
    let mut baseline = run(
        "random::seed(1);\nlet a = int::random(1000);\nlet b = int::random(1000);\nlet c = float::random(1.0);\nlet d = float::random(1.0);",
    );
    assert_eq!(xs[2], get(&mut baseline, "d"));
    assert_eq!(get(&mut vm, "second"), get(&mut baseline, "b"));
}

// r[verify mimas.fx.with]
#[test]
fn with_unit_checkpoints_the_live_state() {
    // `()` keeps the live stream but snapshots it — draws inside don't
    // move the enclosing stream, and a `random::seed` inside unwinds too
    let mut vm = run(
        r#"
random::seed(1);
let a = int::random(1000);
let peek = with rng = () {
    random::seed(99);
    int::random(1000)
};
let b = int::random(1000);
"#,
    );
    let mut baseline = run(
        "random::seed(1);\nlet x = int::random(1000);\nlet y = int::random(1000);",
    );
    let mut seeded = run("random::seed(99);\nlet only = int::random(1000);");
    assert_eq!(get(&mut vm, "a"), get(&mut baseline, "x"));
    assert_eq!(get(&mut vm, "peek"), get(&mut seeded, "only"));
    assert_eq!(get(&mut vm, "b"), get(&mut baseline, "y"));
}

// r[verify mimas.fx.with.restore]
#[test]
fn fault_inside_with_restores_the_scope() {
    // `panic` unwinds the frame past the desugared pop — the fixture's
    // depth-keyed reconcile restores what the scope was keeping
    let mut vm = run(
        r#"
fn boom() {
    int::random(1000);
    with rng = 7 {
        int::random(1000);
        panic("inside");
    };
}
fn draw() -> int { int::random(1000) }
"#,
    );
    vm.fixture::<Rng>().seed(1);
    let err = vm
        .call_fn_result("boom")
        .expect_err("boom should fault");
    assert!(err.to_string().contains("inside"), "{err}");
    assert_eq!(
        vm.fixture::<Rng>().with_scopes(),
        0,
        "the faulted scope was never reconciled"
    );
    // the outer stream kept its place — `boom` drew once, so `draw` reads
    // the seed-1 stream's second position
    let mut baseline = run(
        "random::seed(1);\nlet a = int::random(1000);\nlet b = int::random(1000);",
    );
    assert_eq!(
        vm.call_fn_result("draw").unwrap(),
        get(&mut baseline, "b"),
    );
    assert_eq!(vm.fixture::<Rng>().with_scopes(), 0);
}

// r[verify mimas.fx.with.restore]
#[test]
fn top_level_fault_restores_the_scope() {
    let mut vm = Vm::compile(
        r#"
random::seed(1);
let before = int::random(1000);
with rng = 7 {
    int::random(1000);
    panic("inside");
};
"#,
        library::std,
    )
    .expect("with test compiled");
    vm.run().expect_err("the with body should fault");
    assert_eq!(
        vm.fixture::<Rng>().with_scopes(),
        0,
        "a top-level fault left the scope installed"
    );
    // the enclosing stream drew `before` once and kept its place — the
    // restored state is exactly seed-1-after-one-draw
    let probe = Rng::default();
    probe.seed(1);
    probe.next_unit();
    assert_eq!(vm.fixture::<Rng>().state(), probe.state());
}

// r[verify mimas.fx.with]
#[test]
fn with_in_a_check_replays() {
    // the motivating spelling: `check with rng = …` — the region is the
    // check's own expression, and the seed makes its draws deterministic
    let mut vm = Vm::compile(
        r#"
fn shuffled() -> [int] {
    let xs = [1, 2, 3, 4, 5];
    xs.shuffle();
    xs
}
let want = with rng = 7 { shuffled() };
check with rng = 7 { shuffled() == want }
"#,
        library::std,
    )
    .expect("with test compiled");
    let results = vm.run_tests();
    assert_eq!(results.len(), 1, "{results:?}");
    assert!(results[0].passed(), "{results:?}");
}

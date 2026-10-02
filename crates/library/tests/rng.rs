//! The `Rng` fixture — every random native (`float::random`, `int::random`,
//! `bool::random`, `arr.choose`, `arr.shuffle`) draws through the Vm's one
//! stream, so a host that seeds it gets a replayable program, and a scripted
//! draw overrides a single position without disturbing the rest.

use vm::{Vm, fixtures::Rng};

fn run_seeded(src: &str, names: &[&str], seed: u64) -> Vec<String> {
    let mut vm = Vm::compile(src, library::std).expect("rng test compiled");
    vm.fixture::<Rng>().seed(seed);
    vm.run().expect("rng test ran");
    names
        .iter()
        .map(|n| vm.resolve_name_to_string(n).unwrap().unwrap())
        .collect()
}

const DRAWS: &str = r#"
let a = float::random(1.0);
let b = int::random(1000);
let c = bool::random();
let d = [1, 2, 3, 4, 5, 6, 7, 8];
d.shuffle();
let e = [10, 20, 30].choose();
"#;
const DRAW_NAMES: &[&str] = &["a", "b", "c", "d", "e"];

#[test]
fn seeded_runs_replay() {
    let first = run_seeded(DRAWS, DRAW_NAMES, 0xdead_beef);
    let second = run_seeded(DRAWS, DRAW_NAMES, 0xdead_beef);
    assert_eq!(first, second, "same seed, different stream");
}

#[test]
fn different_seeds_diverge() {
    let a = run_seeded(DRAWS, DRAW_NAMES, 1);
    let b = run_seeded(DRAWS, DRAW_NAMES, 2);
    assert_ne!(a, b, "different seeds drew identical streams");
}

#[test]
fn program_seed_pins_the_stream() {
    // `random::seed` inside the source starts the same stream the host
    // would with `Rng::seed` — same position, same draws.
    let via_native = r#"
random::seed(42);
let a = float::random(1.0);
let b = int::random(1000);
"#;
    let mut vm = Vm::compile(via_native, library::std).expect("rng test compiled");
    vm.run().expect("rng test ran");
    let got: Vec<String> = ["a", "b"]
        .iter()
        .map(|n| vm.resolve_name_to_string(n).unwrap().unwrap())
        .collect();
    let via_host = run_seeded(
        "let a = float::random(1.0);\nlet b = int::random(1000);",
        &["a", "b"],
        42,
    );
    assert_eq!(got, via_host, "random::seed diverged from a host-seeded stream");
}

#[test]
fn program_reseed_repeats_the_sequence() {
    // Seeding mid-run rewinds the stream — a fuzz harness gets the same
    // playout out of one seed twice.
    let src = r#"
random::seed(7);
let a = int::random(1000);
random::seed(7);
let b = int::random(1000);
"#;
    let mut vm = Vm::compile(src, library::std).expect("rng test compiled");
    vm.run().expect("rng test ran");
    assert_eq!(
        vm.resolve_name_to_string("a").unwrap().unwrap(),
        vm.resolve_name_to_string("b").unwrap().unwrap(),
    );
}

#[test]
fn unseeded_still_works() {
    let mut vm =
        Vm::compile("let out = int::random(10);", library::std).expect("rng test compiled");
    vm.run().expect("unseeded draw ran");
    let n: i64 = vm
        .resolve_name_to_string("out")
        .unwrap()
        .unwrap()
        .parse()
        .unwrap();
    assert!((0..10).contains(&n));
}

#[test]
fn scripted_draws_override_one_position() {
    let src = r#"
let a = float::random(10.0);
let b = float::random(10.0);
"#;
    // scripted run: first draw forced to 0.5
    let mut vm = Vm::compile(src, library::std).expect("rng test compiled");
    let rng = vm.fixture::<Rng>();
    rng.seed(7);
    rng.script([0.5]);
    vm.run().expect("rng test ran");
    assert_eq!(rng.scripted(), 0, "scripted draw was never consumed");
    assert_eq!(vm.resolve_name_to_string("a").unwrap().unwrap(), "5");
    let b = vm.resolve_name_to_string("b").unwrap().unwrap();

    // unscripted baseline: the scripted draw still advanced the stream,
    // so `b` must equal the baseline's *second* draw
    let baseline = run_seeded(src, &["a", "b"], 7);
    assert_eq!(b, baseline[1]);
}

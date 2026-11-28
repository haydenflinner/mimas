//! `check` inside a block: a `fn` body, a loop, an `if`, a closure, a
//! `#[test]` fn. Each runs where it stands with the block's locals in
//! scope, does not stop the run when it fails, and has its own entry in
//! the check list under its source text.

fn run(src: &str) -> Vec<vm::TestResult> {
    let mut vm = vm::Vm::compile(src, library::std).unwrap();
    vm.run_tests()
}

fn find<'a>(results: &'a [vm::TestResult], name: &str) -> &'a vm::TestResult {
    results
        .iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("no check `{name}` in {results:?}"))
}

// r[verify check.block]
#[test]
fn a_check_in_a_test_fn_passes() {
    let results = run("#[test]\nfn t() { let v = 2; check v == 2 }\n");
    assert_eq!(results.len(), 2, "{results:?}");
    assert!(results.iter().all(|r| r.passed()), "{results:?}");
}

// r[verify check.block]
#[test]
fn a_failing_check_in_a_test_fn_fails_the_test_and_names_it() {
    let results = run("#[test]\nfn t() { let v = 2; check v == 3 }\n");
    let t = find(&results, "t");
    let err = t.error.as_deref().expect("t fails");
    assert!(err.contains("in t:") && err.contains("2 == 3"), "{err}");
    let own = find(&results, "v == 3");
    assert!(!own.passed(), "{results:?}");
}

// r[verify check.block]
#[test]
fn a_check_in_a_for_body_sees_the_loop_local() {
    let results = run("for i in 0..2 { check i < 5 }\n");
    assert_eq!(results.len(), 1, "{results:?}");
    assert!(results[0].passed(), "{results:?}");
    // a failure keeps the first one and lets the loop finish
    let src = "let seen = [0]\nfor i in 0..9 { check i < 5\n seen[0] += 1 }\ncheck seen[0] == 9\n";
    let results = run(src);
    let own = find(&results, "i < 5");
    let err = own.error.as_deref().expect("i < 5 fails");
    assert!(err.contains("5 < 5"), "{err}");
    assert!(find(&results, "seen[0] == 9").passed(), "{results:?}");
}

// r[verify check.block]
#[test]
fn a_check_in_a_fn_body_or_if_does_not_crash() {
    let results = run("fn f() -> int { check 1 == 1\n 3 }\nlet x = f()\nif true { check x == 3 }\n");
    assert!(results.iter().all(|r| r.passed()), "{results:?}");
}

// A top-level check that calls a fn whose own check fails fails with
// that message, and the fn's name is in it.
// r[verify check.block]
#[test]
fn a_check_that_calls_a_failing_block_check_fails() {
    let src = "fn f(x: int) -> int {\n  check x > 0, x < 10\n  x * 2\n}\ncheck f(3) == 6\ncheck f(-1) == -2\n";
    let results = run(src);
    assert_eq!(results.len(), 4, "{results:?}");
    assert!(find(&results, "f(3) == 6").passed(), "{results:?}");
    let err = find(&results, "f(-1) == -2").error.clone().expect("fails");
    assert!(err.contains("in f:") && err.contains("-1 > 0"), "{err}");
}

// In a block `;` ends the check, as it ends any statement there; `self`
// may open one in a method body.
// r[verify check.block]
#[test]
fn a_semicolon_ends_a_block_check() {
    let src = "struct P { x: int }\nimpl P {\n  fn dbl(self) -> int { check self.x >= 0; self.x * 2 }\n}\nlet d = P { x: -4 }.dbl()\ncheck d == -8\n";
    let results = run(src);
    assert_eq!(results.len(), 2, "{results:?}");
    let err = find(&results, "self.x >= 0").error.clone().expect("fails");
    assert!(err.contains("in dbl:") && err.contains("-4 >= 0"), "{err}");
    assert!(find(&results, "d == -8").passed(), "{results:?}");
}

// A closure's check runs on each call; a notation body is a fn body.
// r[verify check.block]
#[test]
fn checks_in_closures_and_notations() {
    let src = "notation word(s: str) -> int { s.len() }\ncheck word\"abc\" == 3\n#[test]\nfn t() { check word\"ab\" == 2 }\nlet xs = [1, 2, 3].map(|v| { check v < 3; v * 10 })\n";
    let results = run(src);
    assert!(find(&results, "word\"abc\" == 3").passed(), "{results:?}");
    assert!(find(&results, "t").passed(), "{results:?}");
    assert!(find(&results, "word\"ab\" == 2").passed(), "{results:?}");
    assert!(!find(&results, "v < 3").passed(), "{results:?}");
}

// A second run of the checks starts from a clean record.
// r[verify check.block]
#[test]
fn a_rerun_starts_clean() {
    let mut vm = vm::Vm::compile("let n = [0]\nfor i in 0..3 { n[0] += 1 }\nfn g() { check n[0] == 3 }\ng()\n", library::std).unwrap();
    assert!(vm.run_tests().iter().all(|r| r.passed()));
    assert!(vm.run_tests().iter().all(|r| r.passed()));
}

// A block check no run reached is neither passed nor failed: its entry
// reports "not reached". One that ran reports pass or fail as before.
// r[verify check.block.unreached]
#[test]
fn a_block_check_no_run_reached_is_not_reached() {
    let results = run("fn f(x: int) -> int { check x > 0; x }\nfn g(x: int) -> int { check x < 100; x }\nlet a = f(1)\ncheck a == 1\n");
    let reached = find(&results, "x > 0");
    assert!(reached.passed() && !reached.not_reached, "{results:?}");
    let missed = find(&results, "x < 100");
    assert!(missed.not_reached && !missed.passed() && !missed.failed(), "{results:?}");
    // a later run reaching it settles it
    let results = run("fn g(x: int) -> int { check x < 100; x }\nlet b = g(200)\n");
    assert!(find(&results, "x < 100").failed(), "{results:?}");
}

// A block check whose entry runs before the check that reaches it is
// asked again once every test ran: reached by a later top-level check, it
// passes or fails, not "not reached".
// r[verify check.block.unreached]
#[test]
fn a_later_check_that_reaches_a_block_check_settles_it() {
    let src = "fn half(n: int) -> int {\n    check n mod 2 == 0\n    n ~/ 2\n}\ncheck half(4) == 2\n";
    let results = run(src);
    let inner = find(&results, "n mod 2 == 0");
    assert!(inner.passed() && !inner.not_reached, "{results:?}");
    let results = run("fn half(n: int) -> int {\n    check n mod 2 == 0\n    n ~/ 2\n}\ncheck half(3) == 1\ncheck half(4) == 2\n");
    assert!(find(&results, "n mod 2 == 0").failed(), "{results:?}");
    assert!(find(&results, "half(4) == 2").passed(), "{results:?}");
}

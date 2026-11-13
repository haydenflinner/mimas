#[macro_use]
mod test_runner;

test_run!(
    assert_eq_passes,
    r#"{ assert!(1 == 1); 0 }"# => "0",
    r#"{ assert!(1 != 2); 0 }"# => "0",
    r#"{ assert!(2 > 1); 0 }"# => "0",
    r#"{ assert!(1 < 2); 0 }"# => "0",
    r#"{ assert!(2 >= 2); 0 }"# => "0",
    r#"{ assert!(1 <= 1); 0 }"# => "0",
    r#"{ assert!((1 + 1) == 2); 0 }"# => "0",
);

test_run!(
    assert_in_passes,
    r#"{ assert!(1 in [1, 2, 3]); 0 }"# => "0",
    r#"{ assert!(4 !in [1, 2, 3]); 0 }"# => "0",
);

test_run!(
    assert_bool_passes,
    r#"{ assert!(true); 0 }"# => "0",
    r#"{ assert!(1 == 1 && 2 == 2); 0 }"# => "0",
);

test_run!(
    assert_add_works,
    "fn add(a: int, b: int) -> int { a + b }",
    r#"{ assert!(add(1, 2) == 3); 0 }"# => "0",
);

test_fail!(
    assert_eq_fails,
    "assert!(1 == 2);",
    "assert!(1 != 1);",
    "assert!(1 > 2);",
    "assert!(true == false);",
);

test_fail!(
    assert_in_fails,
    "assert!(4 in [1, 2, 3]);",
    "assert!(1 !in [1, 2, 3]);",
);

test_fail!(
    assert_bool_fails,
    "assert!(false);",
    "assert!(1 == 2 && true);",
);

#[test]
fn assert_eq_failure_shows_values() {
    let err = test_runner::try_execute("assert!(1 == 2);")
        .unwrap_err()
        .to_string();
    assert!(err.contains("assertion failed:"), "{err}");
    assert!(err.contains("1"), "{err}");
    assert!(err.contains("2"), "{err}");
    assert!(err.contains("=="), "{err}");
}

#[test]
fn assert_add_failure_shows_computed_values() {
    let err = test_runner::try_execute(
        "fn add(a: int, b: int) -> int { a + b }
         assert!(add(1, 2) == 4);",
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("3"), "{err}");
    assert!(err.contains("4"), "{err}");
    assert!(err.contains("=="), "{err}");
}

#[test]
fn assert_in_test_fn() {
    const SOURCE: &str = r#"
        fn add(a: int, b: int) -> int { a + b }

        #[test]
        fn add_works() {
            assert!(add(1, 2) == 3)
        }

        #[test]
        fn add_fails() {
            assert!(add(1, 2) == 4)
        }
    "#;
    let mut vm = vm::Vm::compile(SOURCE, library::std).unwrap();
    let results = vm.run_tests();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].name, "add_works");
    assert!(results[0].passed(), "{:?}", results[0]);
    assert_eq!(results[1].name, "add_fails");
    assert!(!results[1].passed(), "{:?}", results[1]);
    assert!(
        results[1]
            .error
            .as_ref()
            .is_some_and(|e| e.contains("3") && e.contains("4")),
        "{:?}",
        results[1]
    );
}

#[test]
fn tests_list_runs() {
    const SOURCE: &str = r#"
        fn add(a: int, b: int) -> int { a + b }
        #[tests] [
            add(1, 2) == 3,
            add(1, 2) == 4,
        ]
    "#;
    let mut vm = vm::Vm::compile(SOURCE, library::std).unwrap();
    let results = vm.run_tests();
    assert_eq!(results.len(), 2, "{results:?}");
    assert_eq!(results[0].name, "add(1, 2) == 3");
    assert!(results[0].passed(), "{:?}", results[0]);
    assert_eq!(results[1].name, "add(1, 2) == 4");
    assert!(!results[1].passed(), "{:?}", results[1]);
}

// r[verify check.report.call]
#[test]
fn predicate_failure_shows_its_arguments() {
    let pre = "fn near(a: float, b: float) -> bool { (a - b).abs() < 0.01 }\n\
               fn half(x: float) -> float { x / 2.0 }\n";
    let err = test_runner::try_execute(&format!("{pre}assert!(near(half(3.0), 1.0));"))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("assertion failed: near(1.5, 1.0) is false"),
        "{err}"
    );
    let err = test_runner::try_execute(&format!("{pre}assert!(!(near(half(2.0), 1.0)));"))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("assertion failed: near(1.0, 1.0) is true"),
        "{err}"
    );
}

#[test]
fn predicate_arguments_evaluate_once() {
    // each argument runs once whether the check passes or fails
    test_runner::try_execute(
        "fn tick(c: [int]) -> int { c.push(1); c.len() }\n\
         fn is_one(n: int) -> bool { n == 1 }\n\
         let c = [];\n\
         assert!(is_one(tick(c)));\n\
         assert!(c.len() == 1);",
    )
    .unwrap();
}

test_run!(
    predicate_arguments_keep_their_typing,
    "fn any_of(xs: [int], f: (int) -> bool) -> bool { for x in xs { if f(x) { return true; } } false }\n\
     fn count(xs: [str]) -> int { xs.len() }\n\
     fn pos(n: int) -> bool { n > 0 }",
    r#"{ assert!(any_of([1, -2], |x| x > 0)); 0 }"# => "0",
    r#"{ assert!(any_of([1, -2], pos)); 0 }"# => "0",
    r#"{ assert!(count([]) == 0); 0 }"# => "0",
    r#"{ assert!(!(any_of([], pos))); 0 }"# => "0",
    r#"{ assert!([1, 2].contains(1 + 1)); 0 }"# => "0",
);

// Checks read the top level's `let`s, every one of them and not only the
// first: `run_tests` runs the top level once, then the checks in order on
// that one world, with no reset between them that would blank the globals.
// r[verify check.world]
#[test]
fn every_check_reads_the_top_level() {
    const SOURCE: &str = "let w = 20\nlet xs = [1, 2, 3]\ncheck w == 20\ncheck xs.len() == 3\ncheck xs[0] == 1\ncheck w + xs[2] == 23\n";
    let mut vm = vm::Vm::compile(SOURCE, library::std).unwrap();
    let results = vm.run_tests();
    assert_eq!(results.len(), 4, "{results:?}");
    assert!(results.iter().all(|r| r.passed()), "{results:?}");
    // a second run sees the same world
    assert!(vm.run_tests().iter().all(|r| r.passed()));
}

// A check that faults doesn't blank the globals for the checks after it.
#[test]
fn a_faulting_check_leaves_the_world_for_the_next() {
    const SOURCE: &str = "let xs = [1, 2, 3]\ncheck xs[9] == 1\ncheck xs.len() == 3\n";
    let mut vm = vm::Vm::compile(SOURCE, library::std).unwrap();
    let results = vm.run_tests();
    assert!(!results[0].passed(), "{results:?}");
    assert!(results[1].passed(), "{results:?}");
}

// The failing check's operands come back on the `TestResult` as
// `Inspect` trees — the example a debugger opens, left then right,
// not just the text inside the message. r[verify check.line]
#[test]
fn a_failed_check_carries_its_operands() {
    const SOURCE: &str = "check 39 == 40\n";
    let mut vm = vm::Vm::compile(SOURCE, library::std).unwrap();
    let results = vm.run_tests();
    assert_eq!(results.len(), 1, "{results:?}");
    assert!(!results[0].passed(), "{results:?}");
    assert_eq!(
        results[0].operands,
        [vm::Inspect::Int(39), vm::Inspect::Int(40)],
    );
}

// Each side is evaluated once into the desugar's temp, so a computed
// operand lands as its whole `Inspect` shape — the dealt hand itself,
// not its source text. r[verify check.line]
#[test]
fn a_computed_operand_keeps_its_shape() {
    const SOURCE: &str = "check [1, 2, 3].len() == 4\ncheck [1, 2] == [9]\n";
    let mut vm = vm::Vm::compile(SOURCE, library::std).unwrap();
    let results = vm.run_tests();
    assert_eq!(results.len(), 2, "{results:?}");
    assert_eq!(
        results[0].operands,
        [vm::Inspect::Int(3), vm::Inspect::Int(4)],
        "{results:?}"
    );
    assert_eq!(
        results[1].operands,
        [
            vm::Inspect::Array(vec![vm::Inspect::Int(1), vm::Inspect::Int(2)]),
            vm::Inspect::Array(vec![vm::Inspect::Int(9)]),
        ],
        "{results:?}"
    );
}

// A bare `check x` records the checked value itself; a predicate check
// records its arguments in order — the same values the message names.
// r[verify check.report.call]
#[test]
fn unary_and_call_checks_carry_their_values() {
    const SOURCE: &str = "fn is_one(n: int) -> bool { n == 1 }\n\
                          let flag = false\n\
                          check flag\n\
                          check is_one(2)\n";
    let mut vm = vm::Vm::compile(SOURCE, library::std).unwrap();
    let results = vm.run_tests();
    assert_eq!(results.len(), 2, "{results:?}");
    assert_eq!(results[0].operands, [vm::Inspect::Bool(false)], "{results:?}");
    assert_eq!(results[1].operands, [vm::Inspect::Int(2)], "{results:?}");
}

// Each operand is its own `__check_operand` call, so a predicate's
// mixed-type arguments each keep their type — a `[…]` argument list
// would force them to share one and not compile.
// r[verify check.report.call]
#[test]
fn mixed_type_arguments_each_record() {
    const SOURCE: &str = "fn fits(n: int, s: str) -> bool { n == s.len() }\n\
                          check fits(2, \"abc\")\n";
    let mut vm = vm::Vm::compile(SOURCE, library::std).unwrap();
    let results = vm.run_tests();
    assert_eq!(results.len(), 1, "{results:?}");
    assert!(!results[0].passed(), "{results:?}");
    assert_eq!(
        results[0].operands,
        [
            vm::Inspect::Int(2),
            vm::Inspect::Str("abc".into()),
        ],
    );
}

// A pass records nothing, and a check that faults *before* its operands
// are compared records nothing — the record is drained after each case
// so a stale one can't attribute to the next check.
// r[verify check.world]
#[test]
fn operands_stay_with_the_check_that_failed() {
    const SOURCE: &str = "let xs = [1, 2, 3]\n\
                          check 1 == 1\n\
                          check 39 == 40\n\
                          check xs[9] == 1\n";
    let mut vm = vm::Vm::compile(SOURCE, library::std).unwrap();
    let results = vm.run_tests();
    assert_eq!(results.len(), 3, "{results:?}");
    assert!(results[0].passed(), "{results:?}");
    assert!(results[0].operands.is_empty(), "{results:?}");
    assert!(!results[1].passed(), "{results:?}");
    assert_eq!(
        results[1].operands,
        [vm::Inspect::Int(39), vm::Inspect::Int(40)],
    );
    // `xs[9]` faults while its operand let evaluates — before the
    // witness call runs — so it owns no operands, and the previous
    // check's didn't leak onto it
    assert!(!results[2].passed(), "{results:?}");
    assert!(results[2].operands.is_empty(), "{results:?}");
}


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

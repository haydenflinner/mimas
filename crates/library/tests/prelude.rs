#[macro_use]
mod test_runner;

test_run!(
    print_does_not_error,
    r#"{ print(1); print("hi"); 0 }"# => "0",
    r#"{ print([1, 2, 3]); 0 }"# => "0",
);

// a quantity prints its unit with the base-unit value (`25kW` -> `25000 W`);
// a bare number stays bare
#[test]
fn print_names_the_unit() {
    let vm = vm::Vm::execute(
        "print(25kW); print(9ft / 24px); print(42); dbg(3s); warn(5kg);",
        library::std,
    )
    .expect("runs");
    let lines = vm
        .fixture::<vm::fixtures::Prints>()
        .take()
        .into_iter()
        .map(|(_, _, text)| text)
        .collect::<Vec<_>>();
    assert_eq!(
        lines,
        [
            "25000 W",
            "0.11430000000000001 m/px",
            "42",
            "dbg value: 3 s",
            "5 kg",
        ]
    );
}

// panic has type `!`, so a panicking match/if arm coerces to the other arm's type
test_run!(
    panic_arm_coerces_via_never,
    r#"let x: int = if true { 5 } else { panic("nope") };"#,
    "x" => "5",
);

test_fail!(
    library_names_are_not_values,
    "std;",
    "array;",
    "dict;",
    "let a = std::fs;",
    r#"f"{std}";"#,
);

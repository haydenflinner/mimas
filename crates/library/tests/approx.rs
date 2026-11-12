#[macro_use]
mod test_runner;

// `a ≈ b` (typed `~=`): approximate equality through the std `Approx`
// pact. `==` stays exact. r[verify mimas.op.approx]

test_run!(
    approx_allows_for_rounding_and_eq_stays_exact,
    "0.1 + 0.2 ≈ 0.3" => "true",
    "0.1 + 0.2 ~= 0.3" => "true",
    "0.1 + 0.2 == 0.3" => "false",
    "12lb + 8lb + 3lb + 5lb + 20lb ≈ 48lb" => "true",
    "1.0 ≈ 1.001" => "false",
    "0.1 + 0.2 - 0.3 ≈ 0.0" => "true",
    "3 ≈ 3" => "true",
    // binds like `==`: the arithmetic on each side happens first
    "2.0 * 0.5 ≈ 1.0 && 1 + 1 == 2" => "true",
);

#[test]
fn a_failing_approx_assert_names_both_values() {
    assert!(test_runner::try_execute("assert!(0.1 + 0.2 ≈ 0.3);\nassert!(12lb + 36lb ≈ 48lb);").is_ok());
    let err = test_runner::try_execute("assert!(1.0 + 0.5 ≈ 2.0);").unwrap_err().to_string();
    assert!(err.contains("assertion failed: 1.5 ≈ 2.0"), "{err}");
}

// `sum` (free and method) and `avg` compensate for rounding, as Python's
// `sum` does. r[verify mimas.agg.compensated]
test_run!(
    float_sums_are_compensated,
    "sum([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) == 1.0" => "true",
    "[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1].sum() == 1.0" => "true",
    "sum([1e16, 1.0, -1e16])" => "1.0",
    "avg([0.1, 0.2, 0.3])! ≈ 0.2" => "true",
);

// A failing check names quantities in the page's units, and a written
// number as written. r[verify check.report.units]
#[test]
fn a_failing_check_says_its_units() {
    let err = test_runner::try_execute("fn wide(w: px) -> px { w * 2 }\nassert!(wide(10px) == 30px);").unwrap_err().to_string();
    assert!(err.contains("assertion failed: 20 px == 30px"), "{err}");
    let err = test_runner::try_execute("assert!(12lb + 36lb == 47lb);").unwrap_err().to_string();
    assert!(err.contains("48 lb == 47lb"), "{err}");
    let err = test_runner::try_execute("assert!(2 + 2 == 5);").unwrap_err().to_string();
    assert!(err.contains("assertion failed: 4 == 5"), "{err}");
}

// A `dec` carries a unit, held in the page's unit so it stays exact:
// a price per pound times pounds, a wage per hour times hours.
// r[verify mimas.dec.units]
test_run!(
    dec_quantities_stay_exact,
    "let price: dec<usd/lb> = 1.10usd/lb\nlet qty: dec<lb> = 12lb\nlet wage: dec<usd/h> = 25usd/h\nlet per_kg: dec<usd/kg> = 2.00usd/kg",
    "price * qty == 13.20usd" => "true",
    "qty + 36lb == 48lb" => "true",
    "wage * 40h == 1000usd" => "true",
    "per_kg * 10lb == 9.0718474usd" => "true",
);

#[test]
fn a_dec_quantity_and_a_float_quantity_dont_mix() {
    let err = test_runner::try_execute("let a: dec<usd> = 2usd;\nlet f = 1.5usd;\nlet bad = a + f;").unwrap_err().to_string();
    assert!(err.contains("a `dec` quantity meets a `float` one"), "{err}");
    let err = test_runner::try_execute("let q: dec<lb> = 12lb;\nassert!(q == 13lb);").unwrap_err().to_string();
    assert!(err.contains("assertion failed: 12 lb == 13lb"), "{err}");
}

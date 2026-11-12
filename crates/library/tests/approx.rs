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
    assert!(err.contains("assertion failed: 20.0 px == 30px"), "{err}");
    let err = test_runner::try_execute("assert!(12lb + 36lb == 47lb);").unwrap_err().to_string();
    assert!(err.contains("48.0 lb == 47lb"), "{err}");
    let err = test_runner::try_execute("assert!(2 + 2 == 5);").unwrap_err().to_string();
    assert!(err.contains("assertion failed: 4 == 5"), "{err}");
}

use crate::test_runner;

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
    assert!(err.contains("assertion failed: 20px == 30px"), "{err}");
    let err = test_runner::try_execute("assert!(12lb + 36lb == 47lb);").unwrap_err().to_string();
    assert!(err.contains("48lb == 47lb"), "{err}");
    let err = test_runner::try_execute("assert!(2 + 2 == 5);").unwrap_err().to_string();
    assert!(err.contains("assertion failed: 4 == 5"), "{err}");
}

// A `check` line, not only `assert!`, names its values in units: the
// page's checks are `check` lines. r[verify check.report.units]
#[test]
fn a_failing_check_line_says_its_units() {
    let src = "let w = 20px\nlet qty = [12lb, 8lb, 2lb]\ncheck w == 30px\ncheck sum(qty) == 48lb\ncheck 2 + 2 == 5\n";
    let mut vm = vm::Vm::compile(src, library::std).unwrap();
    let errs: Vec<String> = vm.run_tests().into_iter().map(|r| r.error.unwrap_or_default()).collect();
    assert!(errs[0].contains("assertion failed: 20px == 30px"), "{errs:?}");
    assert!(errs[1].contains("assertion failed: 22lb == 48lb"), "{errs:?}");
    assert!(errs[2].contains("assertion failed: 4 == 5"), "{errs:?}");
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
    assert!(err.contains("assertion failed: 12lb == 13lb"), "{err}");
}

// A `dec` literal too wide for 16 digits is a compile error, not a silent
// 0 (2^55) or a number rounded through a float. r[verify mimas.dec.literal]
#[test]
fn a_dec_literal_that_does_not_fit_is_an_error() {
    for src in ["let b: dec = 36028797018963968\nb", "let c: dec = 99999999999999999.5\nc"] {
        let err = test_runner::try_execute(src).unwrap_err().to_string();
        assert!(err.contains("doesn't fit in a `dec` exactly"), "{src}: {err}");
    }
    assert!(test_runner::try_execute("let d: dec = 9999999999999999\nd").is_ok());
}

// `#1.10` is an exact decimal wherever it stands, so a float line and its
// exact twin differ by one character. r[verify mimas.dec.exact-literal]
test_run!(
    a_hash_literal_is_exact,
    "0.1 + 0.2 == 0.3" => "false",
    "#0.1 + 0.2 == 0.3" => "true",
    "#262144.09 + 0.01 == 262144.10" => "true",
    "0.3 < #0.1 + 0.2" => "false",
    "(#0.7 + 0.1) * 10" => "8.0",
    "#1.10usd/lb * 12lb == 13.20usd" => "true",
);


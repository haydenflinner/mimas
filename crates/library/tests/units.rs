#[macro_use]
mod test_runner;

// a quantity is a float in base units; `to` reads it back in any unit of the same kind
test_run!(
    unit_literals_are_base_unit_floats,
    "",
    "25kW.to(\"W\")" => "25000",
    "(25kW * 4h).to(\"kWh\")" => "100",
    "(25kW * 4h).to(\"J\")" => "360000000",
    "(0.14usd/kWh * 100kWh).to(\"usd\")" => "14",
    "90deg.to(\"rad\") > 1.57" => "true",
    "((6pct * 200.0) * 100.0).round()" => "1200",
    "(3km / 2h).to(\"m/s\") < 0.42" => "true",
    "5kW.to(\"W\") == 5000.0" => "true",
);

test_run!(
    quantities_flow_through_functions,
    "fn energy(p: kW, t: h) -> kWh { p * t }
     fn cost(e: kWh, price: usd/kWh) -> usd { e * price }
     struct Plan { load: kW, tariff: usd/kWh }
     const LOAD = 25kW
     let plan = Plan { load = LOAD, tariff = 0.14usd/kWh };",
    "cost(energy(plan.load, 8h), plan.tariff).to(\"usd\") < 28.01" => "true",
    "cost(energy(plan.load, 8h), plan.tariff).to(\"usd\") > 27.99" => "true",
    "(plan.load * 2.0).to(\"kW\")" => "50",
);

fn rejected(src: &str, needle: &str) {
    let err = test_runner::try_execute(src).expect_err(&format!("`{src}` should be rejected"));
    let text = format!("{err:?}");
    assert!(text.contains(needle), "`{src}`: expected `{needle}` in:\n{text}");
}

#[test]
fn nonsense_arithmetic_is_rejected() {
    rejected("let a = 5kW + 3s;", "cannot add");
    rejected("let a = 5kW - 3s;", "cannot subtract");
    rejected("let a = 5kW > 3s;", "cannot order");
    rejected("let a = 5kW == 3m;", "cannot compare");
    rejected("let a = [5kW, 3s];", "a list mixes");
    rejected("let a = if true { 5kW } else { 3s };", "branches disagree");
    // inside an f-string interpolation too
    rejected("let a = f\"{5kW + 3s}\";", "cannot add");
}

#[test]
fn declared_dimensions_are_enforced() {
    // a plain float is dimensionless: a power isn't one
    rejected("let a: float = 5kW;", "declared type doesn't match");
    rejected("fn f(x: float) -> float { x }\nlet a = f(5kW);", "argument `x` of `f`");
    rejected("fn f(p: kW) -> kW { p }\nlet a = f(5s);", "argument `p` of `f`");
    rejected("fn f(p: kW) -> s { p }", "returns the wrong dimension");
    rejected("struct S { p: kW }\nlet a = S { p = 5s };", "field `p`");
    rejected("let a: kW = 5.0;", "declared type doesn't match");
    rejected("let a = 5kW;\nlet b = a.to(\"s\");", "cannot express");
    rejected("let a = 5kW.sin();", "needs a plain number");
    rejected("let a = 5m.sqrt();", "square root");
}

#[test]
fn unknown_units_are_named() {
    // a lone unknown name is just an unknown type, as ever
    rejected("fn f(x: bogus) {}", "bogus");
    rejected("fn f(x: m/bogus) {}", "isn't a unit");
}

#[test]
fn plain_numbers_scale_and_units_combine() {
    let fine = [
        "let a = 2.0 * 5kW;",
        "let a = 5kW / 2;",
        "let a: kWh = 5kW * 2h;",
        "let a: usd = 2kWh * 0.14usd/kWh;",
        "let a: m = (3m * 3m / 1m) + 2m;",
        "let a: m = (4m * 4m).sqrt();",
        "let a = 5kW > 0;",
        "let a = 5kW * 0 == 0;",
        "let a: float = 5kW / 2kW;",
        "let a: kW = 5kW.abs().max(2kW).min(9kW);",
        "let a: usd/kWh = 1usd / 1kWh;",
        "let a: m/s^2 = 9.81m / 1s / 1s;",
        // untyped code is unaffected
        "fn f(x: float) -> float { x * 2.0 } let a = f(3.0);",
        "let a = [1.0, 2.0].sum();",
    ];
    for src in fine {
        test_runner::try_execute(src).unwrap_or_else(|e| panic!("`{src}` should pass: {e:?}"));
    }
}

#[test]
fn a_declared_type_named_like_a_unit_wins() {
    // programs that already have a `struct W` keep it
    test_runner::try_execute("struct W { x: int }\nfn f(w: W) -> int { w.x }\nlet a = f(W { x = 1 });")
        .unwrap();
}

// ---- intervals keep their units

test_run!(
    interval_arithmetic_carries_units,
    "fn saved() -> Interval<usd> { Interval::pm(30000.0usd, 0.1) * 1.0 }
     let g = Interval::pm(2kW, 0.5) * Interval::pm(4h, 0.25);",
    "saved().lo.to(\"usd\")" => "27000",
    "g.mid.to(\"kWh\")" => "8",
    "g.hi.to(\"kWh\") > 11.2" => "true",
    "(g * 2.0).width().to(\"kWh\") > 0.0" => "true",
);

#[test]
fn intervals_are_dimension_checked() {
    let fine = [
        "let a: Interval<kW> = Interval::pm(5kW, 0.1);",
        "let a: Interval<kWh> = Interval::pm(5kW, 0.1) * Interval::pm(2h, 0.1);",
        "let a: Interval<usd> = Interval::span(1usd, 3usd) + 2usd;",
        "let a: kW = Interval::pm(5kW, 0.1).hi;",
        "let a: kW = Interval::pm(5kW, 0.1).sample(0.5);",
        "let a = Interval::pm(5kW, 0.1) < 9kW;",
        "let a: Interval<m> = Interval::of(1m, 2m, 4m);",
        // an unannotated Interval is unchecked, as ever
        "let a = Interval::pm(10.0, 0.1) + Interval::pm(3.0, 0.1);",
    ];
    for src in fine {
        test_runner::try_execute(src).unwrap_or_else(|e| panic!("`{src}` should pass: {e:?}"));
    }
    rejected("let a = Interval::pm(5kW, 0.1) + Interval::pm(3s, 0.1);", "cannot add");
    rejected("let a = Interval::pm(5kW, 0.1) + 3s;", "cannot add");
    rejected("let a: Interval<s> = Interval::pm(5kW, 0.1);", "declared type doesn't match");
    rejected("let a: s = Interval::pm(5kW, 0.1).hi;", "declared type doesn't match");
    rejected("let a = Interval::pm(5kW, 0.1) < 3s;", "cannot order");
    rejected("let a = Interval::span(1m, 2s);", "need the same dimension");
    rejected("let a = Interval::pm(5kW, 3s);", "plain fraction");
    rejected("fn f(x: Interval<kW>) {}\nlet a = f(Interval::pm(5s, 0.1));", "argument `x` of `f`");
}

// ---- data columns with units
#[cfg(feature = "dataframe")]
test_run!(
    pulled_columns_are_quantities,
    "use std::polars::*;
     let df = from_csv(\"month,kwh\\njan,7400\\nfeb,6900\\n\")!;
     let xs = df.pull_as(\"kwh\", \"kWh\")!;
     let total: kWh = xs.sum();",
    "total.to(\"kWh\")" => "14300",
    "xs[0].to(\"MJ\")" => "26640",
);

#[cfg(feature = "dataframe")]
#[test]
fn pulled_columns_are_dimension_checked() {
    let head = "use std::polars::*;\nlet df = from_csv(\"a,b\\n1,2\\n\")!;\n";
    let ok = format!("{head}let e: kWh = df.pull_as(\"a\", \"kWh\")![0]; let p: [kW] = df.pull_as(\"b\", \"kW\")!;");
    test_runner::try_execute(&ok).unwrap_or_else(|e| panic!("should pass: {e:?}"));
    rejected(&format!("{head}let t = df.pull_as(\"a\", \"kW\")![0] + 3s;"), "cannot add");
    rejected(&format!("{head}let t = df.pull_as(\"a\", \"bogus\")!;"), "isn't a unit");
    rejected(&format!("{head}let t: [s] = df.pull_as(\"a\", \"kW\")!;"), "declared type doesn't match");
}

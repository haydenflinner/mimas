#[macro_use]
mod test_runner;

// `Interval` — a number with its uncertainty: lo, best guess, hi.
test_run!(
    interval_constructors,
    "",
    "Interval::pm(100.0, 0.1).lo" => "90",
    "Interval::pm(100.0, 0.1).hi" => "110",
    "Interval::within(5.0, 2.0).lo" => "3",
    "Interval::span(2.0, 6.0).mid" => "4",
    "Interval::of(1.0, 2.0, 4.0).hi" => "4",
    "Interval::exact(3.0).width()" => "0",
);

test_run!(
    interval_arithmetic_carries_the_bounds,
    "let g = Interval::pm(30.0, 0.1);",
    "(60.0 - g).lo" => "27",
    "(60.0 - g).hi" => "33",
    "(60.0 - g).mid" => "30",
    "(g * 2.0).hi" => "66",
    "(2.0 * g).lo" => "54",
    "(g / Interval::pm(2.0, 0.5)).lo" => "9",
    "(-g).hi" => "-27",
    "Interval::of(1.0, 2.0, 4.0).pow(2.0).hi" => "16",
    "g.contains(31.0)" => "true",
    "g.contains(40.0)" => "false",
);

// increasing functions map the ends straight through; `exp`/`ln` nudge
// the computed ends one ulp outward (no directed rounding in f64), so
// the true image sits strictly inside
test_run!(
    interval_functions_map_the_bounds,
    "let e = Interval::span(0.0, 1.0).exp();",
    "e.lo < 1.0" => "true",
    "e.lo > 0.99" => "true",
    "e.hi >= 2.718281828459045" => "true",
    "(e.hi - 2.718281828459045).abs() < 0.0000000000001" => "true",
    "(Interval::span(0.0, 1.0).exp().mid - 1.6487212707001282).abs() < 0.0000000000001" => "true",
    "Interval::span(1.0, 10.0).ln().lo <= 0.0" => "true",
    "Interval::span(1.0, 10.0).ln().lo > -0.0000001" => "true",
    "(Interval::span(1.0, 10.0).ln().hi - 2.302585092994046).abs() < 0.0000000000001" => "true",
    "Interval::of(0.0, 1.0, 9.0).sqrt().hi" => "3",
);

// `sqrt`/`ln` clamp to their domain, like `pow`: the part below zero
// is cut, and `ln` of a range reaching zero is unbounded below
test_run!(
    interval_functions_clamp_their_domains,
    "",
    "Interval::of(-4.0, 4.0, 9.0).sqrt().lo" => "0",
    "Interval::of(-4.0, 4.0, 9.0).sqrt().hi" => "3",
    "Interval::of(-4.0, 4.0, 9.0).sqrt().mid" => "2",
    "Interval::span(-2.0, -1.0).sqrt().hi" => "0",
    "Interval::span(-1.0, 4.0).ln().hi > 1.38" => "true",
    "Interval::span(-1.0, 4.0).ln().hi < 1.39" => "true",
    "Interval::span(-1.0, 4.0).ln().lo < -1000000.0" => "true",
);

// `min`/`max` take the pointwise envelopes — exact, even overlapping;
// the other side may be a plain number
test_run!(
    interval_min_max_take_the_envelopes,
    "let a = Interval::of(0.0, 1.0, 5.0); let b = Interval::of(3.0, 4.0, 8.0);",
    "a.min(b).lo" => "0",
    "a.min(b).hi" => "5",
    "a.max(b).lo" => "3",
    "a.max(b).hi" => "8",
    "a.min(b).mid" => "1",
    "a.min(Interval::of(2.0, 3.0, 4.0)).hi" => "4",
    "a.max(Interval::of(2.0, 3.0, 4.0)).lo" => "2",
    "a.min(4.5).hi" => "4.5",
    "a.max(7).lo" => "7",
    "a.max(7).mid" => "7",
);

// ordering is *certain*: true only if it holds for every value in both ranges
test_run!(
    interval_comparisons_are_certain,
    "let a = Interval::pm(10.0, 0.1); let b = Interval::pm(20.0, 0.1);",
    "a < b" => "true",
    "b < a" => "false",
    "a < 11.5" => "true",
    "a < 10.5" => "false",
    "a > 5.0" => "true",
    "5.0 < a" => "true",
    "a <= 11.0" => "true",
    "a >= 9.0" => "true",
    "a > b" => "false",
    // overlapping ranges: neither is certain, and `overlaps` says why
    "Interval::pm(10.0, 0.5) < Interval::pm(12.0, 0.5)" => "false",
    "Interval::pm(10.0, 0.5).overlaps(Interval::pm(12.0, 0.5))" => "true",
    "a.overlaps(b)" => "false",
);

test_run!(
    interval_sampling_is_a_triangular_draw,
    "let x = Interval::of(0.0, 2.0, 10.0);",
    "x.sample(0.0)" => "0",
    "x.sample(1.0)" => "10",
    "x.sample(0.2)" => "2",
);

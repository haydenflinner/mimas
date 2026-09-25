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

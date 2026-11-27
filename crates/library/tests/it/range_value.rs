//! Ranges as values (L7).

test_run!(
    // r[verify mimas.range.value]
    a_range_is_a_value,
    "let r = 0..5;",
    "r.map(|i| i * 2).to_list() == [0, 2, 4, 6, 8]" => "true",
    "r.len()" => "5",
    "(1..=3).to_list()" => "[1, 2, 3]",
    "3 in r" => "true",
    "{ let t = 0; for i in r { t += i; } t }" => "10",
    // a `for` head keeps its counted loop
    "{ let t = 0; for i in 0..4 { t += i; } t }" => "6",
);

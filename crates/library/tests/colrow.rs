#[macro_use]
mod test_runner;

// `Col*`/`Row*` — oriented vectors: a `Col` is a coordinate list read
// top-to-bottom, a `Row` is the transpose read left-to-right. They're
// nominal types, so a `Row` can never be passed where a `Col` is wanted
// — the only door between them is `v.t()`.
test_run!(
    colrow_constructors_and_fields,
    "let v = Col2 { x = 1.0, y = 2.0 };",
    "v.x" => "1",
    "v.y" => "2",
    "Col3::new(1.0, 2.0, 3.0).z" => "3",
    "Col4::new(1.0, 2.0, 3.0, 4.0).w" => "4",
    "Row2::new(5.0, 6.0).x" => "5",
);

test_run!(
    colrow_transpose_flips_orientation,
    "let v = Col3::new(1.0, 2.0, 3.0);",
    "v.t().x" => "1",
    "v.t().z" => "3",
    "v.t().t().y" => "2",
    "Row2::new(7.0, 8.0).t().y" => "8",
);

test_run!(
    colrow_arithmetic_is_component_wise,
    "let a = Col2::new(1.0, 2.0); let b = Col2::new(10.0, 20.0);",
    "(a + b).y" => "22",
    "(b - a).x" => "9",
    "(a * b).y" => "40",
    "(b / a).x" => "10",
    "(a + 1.0).x" => "2",
    "(1.0 + a).y" => "3",
    "(a * 2.0).x" => "2",
    "(10.0 - b).y" => "-10",
    "(-a).x" => "-1",
    "(+a).y" => "2",
);

// `row.inner(col)` is the inner product — a scalar. `col.outer(row)`
// is the outer product — a `[[float]]` matrix's rows.
test_run!(
    colrow_orientation_products,
    "let c = Col2::new(1.0, 2.0); let r = Row2::new(3.0, 4.0);",
    "r.inner(c)" => "11",
    "c.outer(r)[0][1]" => "4",
    "c.outer(r)[1][0]" => "6",
    "Row3::new(1.0, 0.0, 1.0).inner(Col3::new(2.0, 5.0, 3.0))" => "5",
);

test_run!(
    colrow_vector_methods,
    "let v = Col2::new(3.0, 4.0);",
    "v.len()" => "5",
    "v.norm().x" => "0.6",
    "v.dist(Col2::new(0.0, 0.0))" => "5",
    "v.dot(Col2::new(0.0, 2.0))" => "8",
    "v.lerp(Col2::new(0.0, 0.0), 0.5).x" => "1.5",
    "v.arr()[1]" => "4",
    "v.neg().y" => "-4",
);

// The whole point: a `Row` can't sneak into `Col` arithmetic, and a
// signature that wants a `Col` rejects a `Row` — compile- or run-time.
test_fail!(
    colrow_rejects_mixed_orientation,
    "Col2::new(1.0, 2.0) + Row2::new(3.0, 4.0)",
    "Col2::new(1.0, 2.0) - Row2::new(3.0, 4.0)",
    "Col2::new(1.0, 2.0) * Row2::new(3.0, 4.0)",
    "Row2::new(1.0, 2.0) * Col2::new(3.0, 4.0)",
    "fn f(p: Col2) -> float { p.x } let _y = f(Row2::new(1.0, 2.0));",
    "fn g(r: Row3) -> float { r.z } let _y = g(Col3::new(1.0, 2.0, 3.0));",
);

test_fail!(
    colrow_rejects_mismatched_arity,
    "Row2::new(1.0, 2.0).inner(Col3::new(1.0, 2.0, 3.0))",
    "Col2::new(1.0, 2.0).outer(Row3::new(1.0, 2.0, 3.0))",
    "Col2::new(1.0, 2.0) + Col4::new(1.0, 2.0, 3.0, 4.0)",
);

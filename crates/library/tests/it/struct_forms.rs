//! Struct forms agree (L3, C10): a pattern renames with `:` as a literal
//! does, a field may have a default, a `fn` parameter may be a pattern, and
//! a struct pattern binds in `let`.

const P: &str = "struct P { x: int, y: int }
let p = P { x = 3, y = 4 };";

test_run!(
    // r[verify mimas.struct.pattern-eq]
    // (a) `:` renames a field in a pattern, as in a literal
    a_pattern_renames_with_a_colon,
    P,
    "match p { P { x: a, y } => a * 10 + y }" => "34",
    "match p { P { x = a, y: b } => a + b }" => "7",
);

test_run!(
    // r[verify mimas.struct.let-pattern]
    // C10: a struct pattern binds in `let`
    a_struct_pattern_binds_in_let,
    P,
    "{ let P { x, y } = p; x * 10 + y }" => "34",
    "{ let P { x = a, y: b } = p; a - b }" => "-1",
);

const M: &str = "struct M { name: str, size: int = 10, tags: [str] = [\"new\"] }";

test_run!(
    // r[verify mimas.struct.field-default]
    // (b) a literal may leave out a field with a default; default() uses it
    a_field_default_fills_a_literal_and_default,
    M,
    "M { name = \"a\" }.size" => "10",
    "M { name = \"a\", size = 2 }.size" => "2",
    "M { name = \"a\" }.tags.len()" => "1",
    "M::default().size" => "10",
    "M::default().name" => "\"\"",
);

test_fail!(
    // r[verify mimas.struct.field-default]
    // a field without a default must still be written
    a_field_without_a_default_is_still_required,
    "struct M { name: str, size: int = 10 }\nlet TEST_VALUE = M { size = 1 };"
);

test_run!(
    // r[verify mimas.fn.pattern-param]
    // (c) a fn parameter may be a pattern, as a closure's may
    a_fn_parameter_may_be_a_pattern,
    "fn add((a, b): (int, int)) -> int { a + b }
fn first((a, _b): (int, int), k: int) -> int { a * k }",
    "add((2, 3))" => "5",
    "first((4, 9), 2)" => "8",
);

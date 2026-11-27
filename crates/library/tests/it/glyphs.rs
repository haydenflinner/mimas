//! Glyph operators lower to reserved names (L10): a page's own `fn sum`,
//! `fn quot_rem` or `fn compose` never changes what `∑`, `÷` or `∘` mean.

test_run!(
    // r[verify mimas.op.glyph-reserved]
    a_page_fn_does_not_capture_a_glyph,
    "fn sum(xs: [int]) -> int { 0 }
fn quot_rem(a: int, b: int) -> int { 0 }
fn compose(f: (int) -> int, g: (int) -> int) -> (int) -> int { |x: int| 0 }
fn inc(x: int) -> int { x + 1 }
fn dbl(x: int) -> int { x * 2 }",
    "∑ [1, 2, 3]" => "6",
    "(7 ÷ 2).0" => "3",
    "(7 ÷ 2).1" => "1",
    "(inc ∘ dbl)(5)" => "11",
    // the page's own fns still answer their own calls
    "sum([1, 2])" => "0",
    "quot_rem(7, 2)" => "0",
);

test_run!(
    // r[verify mimas.op.glyph-decl]
    // a page declares its own glyph; a use above the declaration reads too
    a_page_declares_a_glyph,
    "let early = 2 ⊛ 3;
fn mix(a: int, b: int) -> int { a * 10 + b }
infix ⊛ = mix;",
    "early" => "23",
    "1 ⊛ 2 + 3" => "15",
    "1 + 2 ⊛ 3" => "24",
);

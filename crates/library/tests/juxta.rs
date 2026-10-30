#[macro_use]
mod test_runner;

// `x y` applies: `f x` is `f(x)` — the same call `f (x)` always was,
// extended to atom arguments. It binds tighter than any infix (like
// unary minus), is left-associative, and only joins on the same line.

test_run!(
    juxtaposition_applies,
    "fn add1(x: int) -> int { x + 1 }
     let two = 2",
    // the atom cases: a name, a literal, a group
    "add1 1" => "2",
    "add1 two" => "3",
    "add1 (two + 1)" => "4",
    // a method value applies too: `(xs.len) 0` isn't the point —
    // `xs.map f` spelling is — a name with a postfix chain is the arg
    "[1, 2, 3].map add1" => "[2, 3, 4]",
);

test_run!(
    juxtaposition_binds_like_unary_minus,
    "fn add1(x: int) -> int { x + 1 }
     fn dbl(x: int) -> int { x * 2 }",
    // `add1 2 * 3` is `(add1 2) * 3`, not `add1 (2 * 3)`
    "add1 2 * 3" => "9",
    // left-associative: `f x y` is `(f x) y`
    "dbl (add1 2) + add1 1" => "8",
);

test_run!(
    juxtaposition_edges,
    "let f = 10
     let xs = [4, 5, 6]",
    // `f - x` is subtraction, not `f` applied to `-x`
    "f - 3" => "7",
    // `x [i]` stays indexing — `[` never means apply
    "xs [2]" => "6",
    // a newline is still a statement end — `f` and `x` on separate
    // lines never join
    "{\n let f = |x: int| x + 1\n f\n 5 }" => "5",
);

fn rejected(src: &str, needle: &str) {
    let err = test_runner::try_execute(src).expect_err(&format!("`{src}` should be rejected"));
    let text = format!("{err:?}");
    assert!(
        text.contains(needle),
        "`{src}`: expected `{needle}` in:\n{text}"
    );
}

#[test]
fn juxtaposition_errors_stay_errors() {
    // applying an int is a *type* error, not a parse one — the trade the
    // plan flags: `let a = b c` reports "cannot call `b`" rather than
    // a stray token, and only ever within a line.
    rejected("let a = 3 4;", "cannot be called");
    rejected("let a = 3 b;", "cannot be called");
    // `x if` keeps its stray-token error — keywords are not arguments
    rejected("let a = 1 if true { 2 } else { 3 };", "expected");
}

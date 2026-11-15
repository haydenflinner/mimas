// `expr?` passes a null or a raise on from the enclosing function, like the
// `?` of Rust. `let v = f()?;` gives the value of `f()`, or the function
// returns early. Its return type decides what it returns (`mimas.propagate`).
#[macro_use]
mod test_runner;

const FNS: &str = "
fn first_even(xs: [int]) -> int? {
    for x in xs { if x mod 2 == 0 { return x; } }
    null
}
fn parse_pos(s: str) -> int! {
    if s == \"\" { raise \"empty\"; }
    return 7;
}
// option in, option out: a null returns null
fn tenfold_first_even(xs: [int]) -> int? {
    let e = first_even(xs)?;
    e * 10
}
// raise in, raise out: the same payload
fn doubled(s: str) -> int! {
    let n = parse_pos(s)?;
    return n * 2;
}
// option in a raising function: the raise says what was null
fn need(d: {int}, k: str) -> int! {
    let v = d[k]?;
    return v + 1;
}
// raise in an option function: the error drops to null
fn quiet(s: str) -> int? { parse_pos(s)? + 100 }
// the rest of the function doesn't run
fn steps(d: {int}) -> [str]? {
    let a = d[\"a\"]?;
    let b = d[\"b\"]?;
    [f\"{a}\", f\"{b}\"]
}";

// r[verify mimas.propagate]
test_run!(
    propagate_returns_early,
    FNS,
    "tenfold_first_even([1, 3, 8, 5])" => "80",
    "tenfold_first_even([1, 3])" => "null",
    "doubled(\"x\")!" => "14",
    // the handler checks the payload it was handed
    "doubled(\"\") absolve |e| { if f\"{e}\" == \"empty\" { 1 } else { 0 } }" => "1",
    "need({ a: 4 }, \"a\")!" => "5",
    "need({ a: 4 }, \"b\") absolve |e| { if f\"{e}\" == \"`d[k]` was null\" { 1 } else { 0 } }" => "1",
    "quiet(\"x\")" => "107",
    "quiet(\"\")" => "null",
    "steps({ a: 1, b: 2 })!.len()" => "2",
    "steps({ a: 1 })" => "null",
);

// an infix after a block-ended `absolve` handler binds the absolved value:
// `f() absolve |e| { d } == v` is `(f() absolve |e| { d }) == v`
test_run!(
    absolve_then_infix,
    FNS,
    "parse_pos(\"\") absolve |e| { 0 } == 0" => "true",
    "parse_pos(\"x\") absolve |e| { 0 } == 7" => "true",
    // a block keeps a `let` to itself
    "{ let v = parse_pos(\"\") absolve |e| { 0 }; v }" => "0",
    // the handler's own block may hold a call; `==` still binds outside it
    "parse_pos(\"\") absolve |e| { need({ a: 4 }, \"a\")! } == 5" => "true",
    // `absolve` guards the whole expression to its left, like `??` —
    // the call's raise is what the handler recovers
    "doubled(\"\") absolve |e| { 0 }" => "0",
    "doubled(\"x\") absolve |e| { 0 } == 14" => "true",
);

// r[verify mimas.propagate]
test_run!(
    propagate_in_closures,
    // an annotated closure returns from itself, not from the function around it
    "fn total(ds: [{int}], k: str) -> int? {
        let get = |d: {int}| -> int? { let v = d[k]?; v * 2 };
        let a = get(ds[0])?;
        let b = get(ds[1])?;
        a + b
    }
    // with no annotation, `?` makes the closure return an option
    fn plus_one(d: {int}) -> int? {
        let f = |e: {int}| { let v = e[\"q\"]?; v + 1 };
        f(d)
    }",
    "total([{ x: 1 }, { x: 2 }], \"x\")" => "6",
    "total([{ x: 1 }, { y: 2 }], \"x\")" => "null",
    "plus_one({ q: 3 })" => "4",
    "plus_one({ r: 3 })" => "null",
);

// r[verify mimas.propagate.where]
test_fail!(
    propagate_needs_somewhere_to_go,
    // no function to return from
    "let d = { a: 1 }; let v = d[\"a\"]?;",
    // a function that returns a plain value can't carry the null
    "fn f(d: {int}) -> int { let v = d[\"a\"]?; v }",
    // a plain value is never null
    "fn g(x: int) -> int? { let v = x?; v }",
);

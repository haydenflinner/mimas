#[macro_use]
mod test_runner;

// `+`/`-` on a sequence is the Z-set surface — `xs + x` appends one
// `x`, `xs - x` drops the first `==` match, `xs + ys` unions (concat),
// `xs - ys` removes each `y`. Weighted multiplicity keeps the `Change`
// spelling (`xs + add(3)`); these are the singleton reads. `xs += x`
// and `xs -= x` come through the same rule.

test_run!(
    zset_arithmetic,
    "[1, 2] + 3" => "[1, 2, 3]",
    "[1, 2, 1] - 1" => "[2, 1]",
    "[1, 2] - 9" => "[1, 2]",
    "[1, 2] + [3, 4]" => "[1, 2, 3, 4]",
    "[1, 2, 3] - [3, 9]" => "[1, 2]",
    "[\"a\"] + \"b\"" => "[\"a\", \"b\"]",
    "[[1]] + [2]" => "[[1], [2]]",
);

test_run!(
    zset_assign,
    "let xs = [1, 2];
     xs += 3;
     xs -= 1;
     let ys = [4];
     ys += [5, 6];
     ys -= [6, 9];",
    "xs" => "[2, 3]",
    "ys" => "[4, 5]",
);

test_fail!(
    zset_rejects,
    // element and list types don't mix — `[int] + float` is not a thing
    "[1, 2] + 2.5",
    "[1, 2] + \"x\"",
    // and a `[T; n]` place can't grow or shrink — same rule as `push`
    "let xs: [int; 2] = [1, 2]; xs += 3;",
);

// A spaced `!` never reads as postfix unwrap (it hugs like `x%`), so a
// fresh element starting with `!` parses as a negation, not as
// `(!a)!`-applied-to-`b`. The Zig rule: adjacency with two readings
// gets neither.
test_run!(
    spaced_bang_is_not_unwrap,
    "let o: int? = 7;",
    "[!true, !false]" => "[false, true]",
    "[!true !false]" => "[false, true]",
    "o!" => "7",
);

test_fail!(
    spaced_bang_is_not_apply,
    // `f !x` is neither `f(!x)` nor `(f!)(x)` — say which
    "fn f(b: bool) -> bool { b }
     let _v = f !true;",
    "let xs = [1, 2];
     xs !true",
);

// `?` demotes with the same hugging rule — `f ?x` was `(f?)(x)`.
test_run!(
    spaced_hook_is_not_demote,
    "fn r() -> int! { 2 } fn boom() -> int! { raise(\"b\") }
     fn d() -> int? { r()? } fn e() -> int? { boom()? }",
    "d()" => "2",
    "e()" => "null",
    "[d()! d()!]" => "[2, 2]",
);

test_fail!(
    spaced_hook_is_not_apply,
    // `f ?x` is neither `f(x?)` nor `(f?)(x)`
    "fn f(o: int?) -> int? { o }
     let _v = f ?r();",
);

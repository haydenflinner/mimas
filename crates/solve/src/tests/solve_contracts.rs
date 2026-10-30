//! `where` clauses and `[T; n]` sized arrays: what solves, what fails statically.
//! Runtime fallbacks live in `library/tests/contracts.rs` -- everything here is
//! about what the checker can prove (see `lens.rs`). The test session has no std
//! library, so predicates stick to operators and `.len()` stays in those tests.

// --- where predicates ---

test_success!(
    where_literal_passes,
    "fn f(n: int) -> int where n > 0 { n }
     f(2);"
);

test_fail!(
    where_literal_fails,
    // `n` binds to the caller's literal -- `-1 > 0` folds to false
    "fn f(n: int) -> int where n > 0 { n }
     f(-1);",
    // both predicates must hold; the second one can't
    "fn f(a: int, b: int) -> int where a > 0, b > 5 { 0 }
     f(1, 3);"
);

test_success!(
    where_unknown_defers,
    // `g()`'s return value isn't const -- no static answer, no error; the
    // callee's entry check carries it
    "fn f(n: int) -> int where n > 0 { n }
     fn g() -> int { 0 }
     f(g());"
);

test_fail!(
    where_defaults_evaluated,
    // `n` defaults to 3 -- `n > 5` on a call without it is provably false
    "fn f(n = 3) -> int where n > 5 { n }
     f();"
);

test_fail!(
    where_must_be_bool,
    "fn f(xs: [int]) -> int where xs { 0 }",
    "fn f(n: int) -> int where n { 0 }"
);

// --- [T; n] contracts ---

test_success!(
    sized_literal_match,
    "fn f(m: [int; 3]) -> int { 0 }
     f([1, 2, 3]);"
);

test_fail!(
    sized_literal_mismatch,
    "fn f(m: [int; 3]) -> int { 0 }
     f([1, 2]);"
);

test_success!(sized_annotation_match, "let m: [int; 3] = [1, 2, 3];");

test_fail!(sized_annotation_mismatch, "let m: [int; 4] = [1, 2, 3];");

test_success!(
    sized_widens_to_unsized,
    "let m: [int; 3] = [1, 2, 3];
     fn f(xs: [int]) -> int { 0 }
     f(m);"
);

test_success!(
    unsized_defers_to_runtime,
    // `xs` is `[int]` -- nothing to prove against, so the `[int; 3]` contract
    // lowers to a runtime check rather than an error here
    "let xs: [int] = [1, 2];
     fn f(m: [int; 3]) -> int { 0 }
     f(xs);"
);

// --- multidimensional ---

test_success!(
    matrix_literal_match,
    "fn f(m: [float; 2, 3]) -> int { 0 }
     f([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);"
);

test_fail!(
    matrix_literal_mismatch,
    // outer dim: 1 vs 2
    "fn f(m: [float; 2, 3]) -> int { 0 }
     f([[1.0, 2.0, 3.0]]);",
    // inner dim: every element is a uniform 3-long literal vs a pinned 2
    "fn f(m: [float; 2, 2]) -> int { 0 }
     f([[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]]);"
);

test_success!(
    matrix_ragged_defers,
    // a ragged literal doesn't *prove* inner dims either way -- the inner `2`
    // becomes a runtime check on each row, not a compile error
    "fn f(m: [float; 2, 2]) -> int { 0 }
     f([[1.0, 2.0], [3.0, 4.0, 5.0]]);"
);

//! `where` clauses and `[T; n]` arrays end to end: what passes, what panics at
//! runtime, and what the checker rejects outright (see `solve/src/lens.rs`).

#[macro_use]
mod test_runner;

// --- where predicates ---

test_run!(
    where_len_known_ok,
    "fn f(xs: [int]) -> int where xs.len() > 2 { xs.len() }",
    "f([1, 2, 3])" => "3"
);

test_run!(
    where_len_unknown_ok,
    // `xs`'s length isn't provable at the call site -- the entry check runs it
    "fn f(xs: [int]) -> int where xs.len() > 1 { xs[0] }
     fn g() -> [int] { [9, 8] }",
    "f(g())" => "9"
);

test_fail!(
    where_len_unknown_fails,
    // same unknown shape, wrong answer at runtime -- the entry check fires
    "fn f(xs: [int]) where xs.len() > 2 { 0 }
     fn g() -> [int] { [1] }
     f(g());"
);

test_fail!(
    where_scalar_runtime_fails,
    // `n > 0` on a computed argument -- unprovable statically, false at runtime
    "fn f(n: int) where n > 0 { n }
     fn g() -> int { -2 }
     f(g());"
);

// --- [T; n] contracts ---

test_run!(
    sized_param_ok,
    "fn f(m: [int; 3]) -> int { m[0] + m[2] }",
    "f([10, 20, 30])" => "40"
);

test_fail!(
    sized_param_runtime_fail,
    // `m` is `[int]` at the call site -- the compiler can't prove its length, so
    // the callee's entry check does it
    "fn f(m: [int; 4]) -> int { 0 }
     fn make() -> [int] { [1, 2, 3] }
     f(make());"
);

test_fail!(
    sized_let_runtime_fail,
    // a `[int; 3]` binding from a value the checker can't size -- the
    // lowered check sits on the rhs expr
    "fn make() -> [int] { [1, 2] }
     let m: [int; 3] = make();"
);

test_run!(
    sized_matrix_ok,
    "fn f(m: [float; 2, 3]) -> int { m.len() }",
    "f([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])" => "2"
);

test_fail!(
    sized_matrix_runtime_fail,
    // outer len is right, a row's isn't -- the per-element check catches it
    "fn f(m: [float; 2, 2]) -> int { 0 }
     fn make() -> [[f32]] { [[1.0, 2.0], [3.0, 4.0, 5.0]] }
     f(make());"
);

test_run!(
    sized_after_where,
    // sized param and a where predicate compose -- both checked at entry
    "fn f(m: [int; 2], n: int) -> int where n > 0 { m[0] + n }",
    "f([10, 20], 5)" => "15"
);

test_fail!(
    sized_after_where_runtime,
    "fn f(m: [int; 2], n: int) -> int where n > 0 { m[0] + n }
     fn z() -> int { 0 }
     f([10, 20], z());"
);

test_fail!(
    sized_push_rejected,
    // length-changing methods on a sized array fail at compile time
    "let m: [int; 3] = [1, 2, 3];
     m.push(4);",
    "let m: [int; 3] = [1, 2, 3];
     m.pop();",
    "let m: [int; 3] = [1, 2, 3];
     m.extend([4]);"
);

test_run!(
    sized_inplace_ok,
    // length-preserving mutation is fine -- the pin is about size, not contents
    "let m: [int; 3] = [3, 1, 2];
     m.shuffle();",
    "m.len()" => "3"
);

#[test]
fn contract_failure_renders_the_arg_span() {
    // `__contract_fail` carries RtErr::Contract -- the message is the diagnostic
    // title itself, and LocatedRtErr underlines the throwing inst's span
    let src = "fn f(m: [int; 4]) -> int { 0 }\nfn make() -> [int] { [1, 2, 3] }\nf(make());";
    let Err(err) = vm::Vm::execute(src, library::std) else {
        panic!("expected a contract failure")
    };
    let rendered = format!("{err:?}");
    eprintln!("{rendered}");
    // the arriving length rides along in the message -- `got` is the runtime len
    assert!(
        rendered.contains("× expected an array of length 4, got 3"),
        "{rendered}"
    );
    assert!(rendered.contains("the contract fails here"), "{rendered}");
}

#[test]
fn contract_failure_names_the_inner_dimension() {
    // a matrix whose outer len is right but a row is wrong: the walk reaches the
    // row's len check at dimension 1 and says so
    let src = "fn f(m: [float; 2, 2]) -> int { 0 }\nfn make() -> [[float]] { [[1.0, 2.0], [3.0, 4.0, 5.0]] }\nf(make());";
    let Err(err) = vm::Vm::execute(src, library::std) else {
        panic!("expected a contract failure")
    };
    let rendered = format!("{err:?}");
    eprintln!("{rendered}");
    assert!(
        rendered.contains("× expected an array of length 2 at dimension 1, got 3"),
        "{rendered}"
    );
}

#[test]
fn where_failure_renders_the_predicate_span() {
    let src = "fn f(n: int) -> int where n > 0 { n }\nfn z() -> int { -2 }\nf(z());";
    let Err(err) = vm::Vm::execute(src, library::std) else {
        panic!("expected a contract failure")
    };
    let rendered = format!("{err:?}");
    eprintln!("{rendered}");
    assert!(rendered.contains("× where `n > 0` failed"), "{rendered}");
    assert!(rendered.contains("the contract fails here"), "{rendered}");
}

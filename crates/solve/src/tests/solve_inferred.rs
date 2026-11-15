use crate::components::Ty::*;

// -- inferred signatures -------------------------------------------------------
// `fn` holes (an un-annotated parameter, a missing `->`) are Vids the body's
// check settles: one monotype per hole, shared across call sites. Declared
// `<T>` stays the spelling for "reusable at any type".
// r[verify mimas.fn.inferred]

test_ty!(
    inferred_param_pins_at_call,
    "fn id(x) { x }",
    "id(1)" => Int,
    "id(2)" => Int,
);

test_ty!(
    inferred_signature_shows_open_vids,
    "fn id(x) { x }",
    "id" => func!((unknown!()) -> unknown!()),
);

// one monotype per hole: the second use at another type rejects -- write `<T>`
// to be called at two types
// r[verify mimas.fn.inferred.mono]
test_fail!(
    inferred_fn_rejects_second_type,
    "fn id(x) { x }
     let a = id(1);
     let b = id(\"s\");"
);

// h's body yields nothing for its `->`: it settles to () while `x` stays
// annotated -- and `f(4)`/`g("a")` show calls still pin open holes
// r[verify mimas.fn.inferred.unit]
test_ty!(
    inferred_return_from_body,
    "fn f(x: int) { x + 1 }
     fn g(x) { 3 }
     fn h(x: int) { let y = x; }",
    "f" => func!((Int) -> Int),
    "g" => func!((unknown!()) -> Int),
    "h" => func!((Int) -> Unit),
    "f(4)" => Int,
    "g(\"a\")" => Int,
    "h(1)" => Unit,
);

// an inferred `->` that reads a declared param instantiates with it per use
test_ty!(
    inferred_return_reads_declared_param,
    "fn id<T>(x: T) { x }",
    "id(1)" => Int,
    "id(\"s\")" => Str,
);

test_ty!(
    inferred_independent_params,
    "fn pair(a, b) { (a, b) }",
    "pair(1, \"s\")" => tuple!(Int, Str),
);

test_ty!(
    inferred_coupled_params_share,
    "fn add(a, b) { a + b }",
    "add(1, 2)" => Int,
);

// `a + b` wants Self operands: the holes merge before any call, so the first
// mixed use already rejects
// r[verify mimas.fn.inferred.mono]
test_fail!(
    inferred_coupled_params_reject_mixed,
    "fn add(a, b) { a + b }
     let x = add(1, \"s\");"
);

test_ty!(
    inferred_unused_param_accepts_arg,
    "fn f(x) { 3 }",
    "f(42)" => Int,
);

test_ty!(
    inferred_recursive_fn,
    "fn fib(n) { if n < 2 { n } else { fib(n - 1) + fib(n - 2) } }",
    "fib(10)" => Int,
);

test_ty!(
    inferred_return_via_return_stmt,
    "fn f(x) { if x > 0 { return 3 } return 4 }",
    "f(0)" => Int,
);

// an inferred `->` pinned by one `return` is like a written `-> int`: the
// paths that yield nothing reject
// r[verify mimas.fn.inferred.unit]
test_fail!(
    inferred_return_partial_paths_reject,
    "fn f(x) { if x { return 3 } }"
);

// a pact signature is a contract, not a body: no `->` still means `-> ()`
// r[verify mimas.fn.inferred.pact-sig]
test_fail!(
    pact_member_no_arrow_stays_unit,
    "pact P { fn m(self); }
     struct S;
     impl P for S { fn m(self) { 3 } }"
);

// recursion at a different instantiation (Bush's (T, T)) can't share one Vid --
// the explicit `<T>` is what buys polymorphic recursion
test_fail!(
    inferred_nonuniform_recursion_needs_params,
    "enum Bush<T> { BLeaf, BNode(T, Bush<(T, T)>) }
     fn size(b) { match b { Bush::BLeaf => 0, Bush::BNode(x, r) => 1 + size(r) } }
     let n = size(Bush::BLeaf);"
);

test_ty!(
    declared_params_still_polymorphic_recursion,
    "enum Bush<T> { BLeaf, BNode(T, Bush<(T, T)>) }
     fn size<T>(b: Bush<T>) -> int {
         match b { Bush::BLeaf => 0, Bush::BNode(x, r) => 1 + size(r) }
     }",
    "size(Bush::BLeaf)" => Int,
);

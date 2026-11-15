// `multi fn` + `method`: opt-in multiple dispatch. Ordered arms over closed-world
// key sets, the generic's own body as the mandatory fallback.
// r[verify mimas.multi.decl] r[verify mimas.multi.method] r[verify mimas.multi.keys]
// r[verify mimas.multi.dispatch] r[verify mimas.multi.dominated] r[verify mimas.multi.ambiguous]
#[macro_use]
mod test_runner;

// first-match order across a pact's whole implementer set; the generic's body
// is the last arm. A pact key is a union of implementer dispatch ids, so the
// arm fires for A *or* B. r[verify mimas.multi.dispatch]
test_run!(
    pact_key_dispatches_per_implementer,
    concat!("pact P { fn tag(self) -> int { 0 } } struct A { v: int } impl P for A { fn tag(self) -> int { 1 } } struct B { v: int } impl P for B { fn tag(self) -> int { 2 } } multi fn f(x) -> int { -1 } method f(x: P) -> int { x.tag() }"),
    "f(A { v = 0 })" => "1",
    "f(B { v = 0 })" => "2",
    "f(9)" => "-1",
);

// a more specific earlier arm wins inside a broader later one — containment,
// not ambiguity; a tuple matching no arm runs the generic body.
// r[verify mimas.multi.dispatch]
test_run!(
    specific_first_arm_wins,
    concat!("pact P { fn tag(self) -> int { 0 } } struct A { v: int } impl P for A { fn tag(self) -> int { 1 } } struct B { v: int } impl P for B { fn tag(self) -> int { 2 } } multi fn f(a, b) -> int { 0 } method f(a: int, b: int) -> int { 1 } method f(a: P, b: P) -> int { 2 }"),
    "f(1, 2)" => "1",
    // partial key match is not a match: `A` is a `P` but `3` is not, so the
    // generic body runs
    "f(A { v = 0 }, 3)" => "0",
    "f(A { v = 0 }, B { v = 0 })" => "2",
);

// enum keys expand to the variant layouts: `q: Q` catches any variant while a
// `Q::Len` arm only catches Len — variant arms don't over-match, enum arms do.
// r[verify mimas.multi.keys]
test_run!(
    enum_and_variant_keys,
    "enum Q { Len(int), Wid(int) } multi fn f(q) -> int { -1 } method f(q: Q::Len) -> int { 2 } method f(q: Q) -> int { 1 }",
    "f(Q::Wid(3))" => "1",
    "f(Q::Len(3))" => "2",
    "f(9)" => "-1",
);

// overlapping but incomparable keys -- `(P, A)` and `(A, P)` both match
// `(A, A)`, neither covers the other -- are a hard error at the later arm,
// never silent ordering. r[verify mimas.multi.ambiguous]
test_fail!(
    ambiguous_arms_rejected,
    r#"
    pact P { fn tag(self) -> int { 0 } }
    struct A { v: int }
    impl P for A { fn tag(self) -> int { 1 } }
    multi fn f(a, b) -> int { 0 }
    method f(a: P, b: A) -> int { 1 }
    method f(a: A, b: P) -> int { 2 }
    let _ = f(A { v = 0 }, A { v = 0 });
    "#,
);

// an arm an earlier arm already covers can never run. r[verify mimas.multi.dominated]
test_fail!(
    dominated_arm_rejected,
    r#"
    pact P { fn tag(self) -> int { 0 } }
    struct A { v: int }
    impl P for A { fn tag(self) -> int { 1 } }
    multi fn f(x) -> int { 0 }
    method f(x: P) -> int { 1 }
    method f(x: A) -> int { 2 }
    let _ = f(A { v = 0 });
    "#,
);

// the host's `use "kit"` splice is text assembly: kit decls and page arms land
// in one compilation, so a page's `method` sees the kit's `multi fn` and its
// calls dispatch over both files' arms. r[verify mimas.multi.method]
test_run!(
    use_spliced_multi_sees_both_sides,
    r#"
    use "multi_kit"
    struct KitT { v: int }
    multi fn describe(x) -> str { "fallback" }
    method describe(x: KitT) -> str { "kit arm" }
    struct PageT { v: int }
    method describe(x: PageT) -> str { "page arm" }
    "#,
    "describe(KitT { v = 1 })" => "\"kit arm\"",
    "describe(PageT { v = 1 })" => "\"page arm\"",
    "describe(3)" => "\"fallback\"",
);

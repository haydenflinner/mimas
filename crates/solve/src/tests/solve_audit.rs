// Audit against counterexamples.org glossary categories.
// Each test asserts the *sound* outcome; a failure marks a real hole.

// ---- 2. Covariant containers ---------------------------------------------------
// `[int]` must NOT be accepted where `[int?]` is expected: the alias shares the
// same mutable Vec, so a null written through the `[int?]` view lands in an
// `[int]` cell.
// NB: xs.push(null) on an [int?] is itself rejected — use index assignment as
// the write instead.
test_fail!(
    #[ignore = "known unsound: [int] as [int?] + arr[0]=null writes null into int array"]
    covariant_array_write_null,
    "let xs: [int] = [1, 2];
     fn poison(arr: [int?]) { arr[0] = null; }
     poison(xs);"
);
test_fail!(
    #[ignore = "known unsound: mutable [int] widens to [int?] -- unify.rs Option arm recurses under Array"]
    covariant_array_alias_only,
    "let xs: [int] = [1, 2];
     fn take(arr: [int?]) { }
     take(xs);"
);

// ---- 3. Incomplete variance: function parameters --------------------------------
// `(int) -> int` must NOT be accepted where `(int?) -> int` is expected — the
// stored function will be *called with* null.
test_fail!(
    #[ignore = "known unsound: T->T? coercion fires inside fn param (contravariant) position -- unify.rs"]
    contravariant_fn_param,
    "fn need_int(x: int) -> int { x }
     let f: (int?) -> int = need_int;
     let bad = f(null);"
);

// same shape through Result (`T` fulfills `T!` is also stripped inside args)
test_fail!(
    #[ignore = "known unsound: T->T! coercion inside fn param position -- unify.rs"]
    contravariant_fn_param_result,
    "fn need_int(x: int) -> int { x }
     let f: (int!) -> int = need_int;
     let bad = f(1);"
);

// ---- existential extraction ------------------------------------------------------
// a `Named`-typed value (some implementer) must NOT flow into a concrete `Foo`
// slot — unify(Pacts, Adt) currently passes when Foo merely implements the pact.
test_fail!(
    #[ignore = "known unsound: symmetric (Pacts, Adt) unify arm extracts concrete from existential"]
    pact_extracts_concrete,
    "pact Named { fn name(self) -> str; }
     struct Foo { x: int }
     struct Bar { y: str }
     impl Named for Foo { fn name(self) -> str { \"f\" } }
     impl Named for Bar { fn name(self) -> str { \"b\" } }
     let n: Named = Bar { y = \"hi\" };
     let f: Foo = n;
     let z = f.x;"
);

// ---- scope escape ----------------------------------------------------------------
// `x.id()` on a bound returns the implementer's Self — outside the pact it must
// widen back to the bound, not leak as `int`.
test_fail!(
    #[ignore = "known unsound: x.id() on pact bound returns bound, unifies with concrete via symmetric arm"]
    skolem_escapes_through_bound_call,
    "pact Ident { fn id(self) -> Self; }
     struct Wrap { v: int }
     impl Ident for Wrap { fn id(self) -> Wrap { self } }
     fn leak(x: Ident) -> Wrap { x.id() }"
);

// ---- 1. Polymorphic references ----------------------------------------------------
// value bindings are monomorphic, so both of these must fail. If the second one
// compiles, generalization + mutation combine into a polymorphic reference.
test_fail!(
    mono_let_no_generalize,
    "let id = |x| x;
     let a = id(1);
     let b = id(\"s\");"
);
test_fail!(
    polymorphic_reference,
    "let r = [];
     r.push(1);
     r.push(\"s\");"
);

// ---- 28. Under false pretenses / equality ------------------------------------------
// T? and T are different runtime layouts; a `T?` must not be usable as `T`.
test_fail!(
    option_into_plain,
    "let a: int? = null;
     let b: int = a;"
);
// cross-type equality must be rejected, not silently false
test_fail!(
    equality_across_types,
    "let bad = 1 == \"s\";"
);

// ---- 21. Recursive types / occurs ---------------------------------------------------
test_fail!(
    occurs_self_containing_array,
    "let r = [];
     r.push(r);"
);

// ---- covariant containers, take 2: dict + generic adt -------------------------------
test_fail!(
    #[ignore = "known unsound: ~{int} widens to ~{int?} -- dict value is invariant, unify recurses covariantly"]
    covariant_dict,
    "let d: ~{int} = ~{ x = 1 };
     fn poison(m: ~{int?}) { m[\"k\"] = null; }
     poison(d);"
);
test_fail!(
    #[ignore = "known unsound: Box<int> widens to Box<int?> -- adt arg position must be invariant"]
    covariant_generic_field,
    "struct Box<T> { v: T }
     let b: Box<int> = Box { v = 1 };
     fn poke(x: Box<int?>) { x.v = null; }
     poke(b);"
);

// ---- polymorphic references, take 2 -------------------------------------------------
// `let f = |x| x` used at both int and str — if the binding's vids get
// freshened per use and a captured mutable cell's vid is freshened too, `f`
// returns two differently-typed views of the SAME array.
test_fail!(
    polymorphic_reference_via_capture,
    "let cell = [];
     let f = || cell;
     let a: [int] = f();
     let b: [str] = f();
     a.push(1);
     b.push(\"x\");"
);

// same probe without annotations — does the second push see the first's binding?
test_fail!(
    polymorphic_reference_via_capture_infer,
    "let cell = [];
     let f = |x| cell.push(x);
     f(1);
     f(\"s\");"
);

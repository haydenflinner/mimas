#[macro_use]
mod test_runner;

// The writes a function's parameters take are derived from its body and its
// call graph (spec `mimas.fn.writes`): member and element writes on a plain
// parameter reach the caller's shared storage with no `&`, re-storing an
// `inout` parameter writes the caller's slot, and a call hands the callee's
// writes through to whichever arguments root them.

// ordinary member writes through a plain parameter need no `&` -- the same
// handle-mutation model Java, Python and JavaScript have.
// r[verify mimas.fn.writes]
test_run!(
    plain_param_member_writes_reach_caller,
    "struct Point { x: int, y: int }
     fn shift(p: Point) { p.x += 10; p.y = 0; }
     let pt = Point { x = 1, y = 2 }
     shift(pt);
     let got = pt.x * 100 + pt.y;
     fn grow(xs: [int]) { xs.push(9); }
     let xs = [1, 2];
     grow(xs);
     let grew = xs.len();",
    "got" => "1100",
    "grew" => "3",
);

// re-storing an `inout` parameter's binding writes the caller's slot -- the
// same act that on a plain parameter only rebinds the local copy.
// r[verify mimas.fn.writes]
test_run!(
    inout_param_restore_writes_caller,
    "fn set(n: inout int) { n = 42; }
     let v = 1;
     set(&v);
     let got = v;",
    "got" => "42",
);

// writes derived through a call chain feed the in-place loop's freeze: a
// helper whose body pushes the collection is the same violation a direct
// `xs.push` is.
// r[verify mimas.for.inplace-freeze]
test_fail!(
    derived_writes_freeze_helper_calls,
    "fn poke(xs: [int]) { xs.push(9); }
     let xs = [1, 2, 3];
     for x in xs {
         poke(xs);
     }",
    "fn poke(xs: [int]) { xs.push(9); }
     fn call_poke(xs: [int]) { poke(xs); }
     let xs = [1, 2, 3];
     for x in xs {
         call_poke(xs);
     }",
);

// a helper that writes a *different* collection stays legal inside the
// frozen loop.
// r[verify mimas.for.inplace-freeze]
test_run!(
    derived_writes_disjoint_helper_ok,
    "fn poke(ys: [int]) { ys.push(9); }
     let xs = [1, 2, 3];
     let ys = [0];
     for x in xs {
         poke(ys);
     }
     let got = ys.len();",
    "got" => "4",
);

// mutating the iterated collection through an alias slips the static guard;
// the loop's latch faults deterministically instead of walking forever.
// r[verify mimas.for.iter-mutate]
test_fail!(
    iter_mutate_alias_faults,
    "let xs = [1, 2, 3];
     let ys = xs;
     for x in xs {
         ys.push(9);
     }",
);

// a lent or kept place outliving a shape change of its own container faults
// on the next use.
// r[verify mimas.place.stale]
test_fail!(
    stale_place_faults,
    "let xs = [1, 2, 3];
     fn bump(p: inout int) { xs.push(9); p = 99; }
     bump(&xs[0]);",
    "let xs = [1, 2, 3];
     let p = &xs[0];
     xs.push(9);
     p = 99;",
    "struct Row { v: int }
     struct Store { items: [Row] }
     impl Store {
         subscript live() -> Row {
             for i in 0 .. self.items.len() { yield &self.items[i]; }
         }
     }
     let s = Store { items = [Row { v = 1 }] };
     let t = s;
     for v in s.live {
         let f = || v.v = 9;
         t.items.push(Row { v = 7 });
         f();
     }",
);

// rebinding the binding the place was minted from detaches it -- the kept
// place still writes the old container.
// r[verify mimas.place.stale]
test_run!(
    stale_place_detach_on_rebind_ok,
    "let xs = [1, 2, 3];
     let p = &xs[0];
     xs = [50, 51];
     p = 67;
     let got = xs[0];",
    "got" => "50",
);

// re-storing a by-value parameter's binding warns that the caller never
// sees it -- `n = 0` and `p = T{..}` alike -- while member writes and
// `inout` re-stores stay quiet.
// r[verify mimas.lint.param-write]
#[test]
fn param_write_dies_warns() {
    let files = [(
        "main",
        "struct P { x: int }
         fn bump(n: int) { n = 0; }
         fn reset(p: P) { p = P { x = 0 }; }
         let pt = P { x = 1 };
         bump(1);
         reset(pt);",
    )];
    let audit = vm::Vm::audit_files(&files, library::std).unwrap();
    let hits: Vec<_> = audit
        .warnings
        .iter()
        .filter(|w| w.to_string().contains("stays inside the call"))
        .collect();
    assert_eq!(hits.len(), 2, "{:?}", audit.warnings);
}

// r[verify mimas.lint.param-write]
#[test]
fn param_writes_that_reach_do_not_warn() {
    let files = [(
        "main",
        "struct P { x: int }
         fn ok(p: P) { p.x += 1; }
         fn set(n: inout int) { n = 5; }
         let pt = P { x = 1 };
         ok(pt);
         let v = 1;
         set(&v);",
    )];
    let audit = vm::Vm::audit_files(&files, library::std).unwrap();
    assert!(
        !audit
            .warnings
            .iter()
            .any(|w| w.to_string().contains("stays inside the call")),
        "{:?}",
        audit.warnings
    );
}

// r[verify mimas.fn.writes]
test_run!(
    sig_reports_derived_writes,
    "struct W { tour: int, xs: [int] }
     fn step(w: W) { w.tour += 1 }
     fn outer(w: W) { step(w); w.xs.push(9) }
     let TEST_VALUE = sig!(outer).writes;",
    "TEST_VALUE" => r#"["w.xs", "w.tour"]"#,
);

test_run!(
    sig_reports_inout_whole_writes,
    "fn rebind(n: inout int) { n = 0 }
     let TEST_VALUE = sig!(rebind).writes;",
    "TEST_VALUE" => r#"["n"]"#,
);

// r[verify mimas.place.field]
test_run!(
    place_fields_write_through,
    "struct Cur { p: __RemovablePlace<int> }
     let xs = [10, 20, 30]
     let c = Cur { p = &xs[0] }
     let seen = c.p
     c.p = 99
     let TEST_VALUE = (seen, xs[0]);",
    "TEST_VALUE" => "[10, 99]",
);

test_run!(
    place_fields_member_write_through,
    "struct Thing { id: int, speed: float }
     struct Hand { gripped: __RemovablePlace<Thing> }
     let things = [Thing { id = 1, speed = 0.0 }, Thing { id = 2, speed = 0.0 }]
     let h = Hand { gripped = &things[1] }
     h.gripped.speed = 2.5
     let TEST_VALUE = things[1].speed;",
    "TEST_VALUE" => "2.5",
);

test_fail!(
    a_place_field_stales_when_its_collection_grows,
    "struct Cur { p: __RemovablePlace<int> }
     let xs = [10]
     let c = Cur { p = &xs[0] }
     xs.push(11)
     print(c.p)",
    "changed since it was lent"
);

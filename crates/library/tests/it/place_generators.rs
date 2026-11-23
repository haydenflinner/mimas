// `yield &e` in a generator body hands out element places across resumptions:
// `for v in s.live` binds `&items[i]` per pull, so `v.f =` writes through by
// construction rather than through heap sharing. `v =` stays the loop-variable
// error, and the iterated store's root freezes like `for r in &xs` does.

// r[verify mimas.generator.place]
// r[verify mimas.for.place-iter]
test_run!(
    place_generator_writes_through,
    "struct Row { v: int }
     struct Store {
         items: [Row],
         gone: [bool],
     }
     impl Store {
         subscript live() -> Row {
             for i in 0 .. self.items.len() {
                 if !self.gone[i] { yield &self.items[i]; }
             }
         }
     }
     let s = Store { items: [Row { v: 1 }, Row { v: 2 }, Row { v: 3 }], gone: [false, true, false] };
     for v in s.live { v.v += 10; }
     let got = s.items;",
    "got" => "[{ 11 }, { 2 }, { 13 }]",
);

// a member read through the pull's place reads the stored slot.
// r[verify mimas.for.place-iter]
test_run!(
    place_generator_member_reads,
    "struct Row { v: int }
     struct Store {
         items: [Row],
         gone: [bool],
     }
     impl Store {
         subscript live() -> Row {
             for i in 0 .. self.items.len() {
                 if !self.gone[i] { yield &self.items[i]; }
             }
         }
     }
     let s = Store { items: [Row { v: 4 }, Row { v: 5 }], gone: [false, false] };
     let out: [int] = [];
     for v in s.live { out.push(v.v * 2); }
     let got = out;",
    "got" => "[8, 10]",
);

// `for v in &s.live` affirms the same binding -- the pull already hands out
// places, so the `&` documents intent rather than minting anything.
// r[verify mimas.for.place-iter]
test_run!(
    place_generator_borrow_iterator,
    "struct Row { v: int }
     struct Store {
         items: [Row],
         gone: [bool],
     }
     impl Store {
         subscript live() -> Row {
             for i in 0 .. self.items.len() {
                 if !self.gone[i] { yield &self.items[i]; }
             }
         }
     }
     let s = Store { items: [Row { v: 1 }], gone: [false] };
     for v in &s.live { v.v = 9; }
     let got = s.items[0].v;",
    "got" => "9",
);

// an empty store pulls null on the first resume and the loop never runs.
// r[verify mimas.for.place-iter]
test_run!(
    place_generator_empty_store,
    "struct Row { v: int }
     struct Store {
         items: [Row],
         gone: [bool],
     }
     impl Store {
         subscript live() -> Row {
             for i in 0 .. self.items.len() {
                 if !self.gone[i] { yield &self.items[i]; }
             }
         }
     }
     let s = Store { items: [], gone: [] };
     let n = 0;
     for v in s.live { n += 1; }
     let got = n;",
    "got" => "0",
);

// a generator body can itself loop over a place iterator (`yield &v`
// re-hands the pulled place), and a nested `for` inside a generator body
// walks deeper storage (`self.bags[i].xs[j]`).
// r[verify mimas.for.place-iter]
test_run!(
    place_generator_nested,
    "struct Inner { v: int }
     struct Bag { xs: [Inner] }
     struct Store { bags: [Bag] }
     impl Store {
         subscript live() -> Bag {
             for i in 0 .. self.bags.len() { yield &self.bags[i]; }
         }
         subscript nums() -> Inner {
             for i in 0 .. self.bags.len() {
                 for j in 0 .. self.bags[i].xs.len() { yield &self.bags[i].xs[j]; }
             }
         }
         subscript relay() -> Inner {
             for v in self.nums { yield &v; }
         }
     }
     let s = Store { bags: [Bag { xs: [Inner { v: 1 }, Inner { v: 2 }] }, Bag { xs: [Inner { v: 3 }] }] };
     for b in s.live { b.xs.push(Inner { v: 9 }); }
     let a = s.bags;
     let s2 = Store { bags: [Bag { xs: [Inner { v: 1 }, Inner { v: 2 }] }, Bag { xs: [Inner { v: 3 }] }] };
     for n in s2.nums { n.v += 10; }
     let b2 = s2.bags;
     let s3 = Store { bags: [Bag { xs: [Inner { v: 5 }] }] };
     for n in s3.relay { n.v += 1; }
     let c = s3.bags[0].xs[0].v;",
    "a" => "[{ [{ 1 }, { 2 }, { 9 }] }, { [{ 3 }, { 9 }] }]",
    "b2" => "[{ [{ 11 }, { 12 }] }, { [{ 13 }] }]",
    "c" => "6",
);

// a manual `it()` pull hands the descriptor, not the element -- `.read()` is
// the explicit deref a kept `let p = &xs[i]` already spells.
// r[verify mimas.generator.place]
test_run!(
    place_generator_manual_pull,
    "struct Row { v: int }
     struct Store {
         items: [Row],
         gone: [bool],
     }
     impl Store {
         subscript live() -> Row {
             for i in 0 .. self.items.len() {
                 if !self.gone[i] { yield &self.items[i]; }
             }
         }
     }
     let s = Store { items: [Row { v: 1 }, Row { v: 2 }], gone: [false, false] };
     let it = s.live;
     let p = it();
     let got = p!.read().v;",
    "got" => "1",
);

// `v = other` is still the loop-variable error even though `v` is place-typed:
// the binding is the loop's, not a slot the body may rebind.
// r[verify mimas.for.elem-bind]
test_fail!(
    place_generator_loopvar_assign_fails,
    "struct Row { v: int }
     struct Store { items: [Row] }
     impl Store { subscript live() -> Row { yield &self.items[0]; } }
     let s = Store { items: [Row { v: 1 }] };
     for v in s.live { v = Row { v: 9 }; }",
);

// mixing `yield v` and `yield &e` in one body can't pick a pull type.
// r[verify mimas.generator.place]
test_fail!(
    place_generator_mixed_yields_fail,
    "struct Store { items: [int] }
     impl Store {
         subscript live() -> int {
             yield self.items[0];
             yield &self.items[1];
         }
     }",
    "struct Store { items: [int] }
     impl Store {
         subscript live() -> int {
             yield &self.items[1];
             yield self.items[0];
         }
     }",
);

// a place yield can't hand out a bare binding (`yield &c`), only a
// field/slot/accessor place.
// r[verify mimas.generator.place]
test_fail!(
    place_generator_bad_place_operand_fails,
    "struct Store { items: [int] }
     impl Store {
         subscript live() -> int {
             let c = 3;
             yield &c;
         }
     }",
);

// `v.remove()` isn't on a yielded place: removal mid-walk would shift the
// rows under the suspended generator.
// r[verify mimas.generator.place]
test_fail!(
    place_generator_no_remove_fails,
    "struct Store { items: [int] }
     impl Store { subscript live() -> int { yield &self.items[0]; } }
     let s = Store { items: [1] };
     for v in s.live { v.remove(); }",
);

// mutating the walked store mid-iteration is the same refusal `for r in &xs`
// gets: pushes grow what the walk indexes, a field rebind stales open pulls,
// and a slot write replaces what a pull names.
// r[verify mimas.for.place-iter-freeze]
test_fail!(
    place_generator_freeze_fails,
    "struct Row { v: int }
     struct Store { items: [Row], gone: [bool] }
     impl Store { subscript live() -> Row { for i in 0 .. self.items.len() { yield &self.items[i]; } } }
     let s = Store { items: [Row { v: 1 }], gone: [false] };
     for v in s.live { s.items.push(Row { v: 9 }); }",
    "struct Row { v: int }
     struct Store { items: [Row], gone: [bool] }
     impl Store { subscript live() -> Row { for i in 0 .. self.items.len() { yield &self.items[i]; } } }
     let s = Store { items: [Row { v: 1 }], gone: [false] };
     for v in s.live { s.items = []; }",
    "struct Row { v: int }
     struct Store { items: [Row], gone: [bool] }
     impl Store { subscript live() -> Row { for i in 0 .. self.items.len() { yield &self.items[i]; } } }
     let s = Store { items: [Row { v: 1 }], gone: [false] };
     for v in s.live { s.gone[0] = true; }",
    "struct Row { v: int }
     struct Store { items: [Row], gone: [bool] }
     impl Store { subscript live() -> Row { for i in 0 .. self.items.len() { yield &self.items[i]; } } }
     let s = Store { items: [Row { v: 1 }], gone: [false] };
     for v in s.live { s.items[0] = Row { v: 9 }; }",
);

// `for v in it` where `it` was bound from `s.live` keeps the producer freeze:
// the let records which store the pulls reach into.
// r[verify mimas.for.place-iter-freeze]
test_fail!(
    place_generator_bound_iterator_freeze_fails,
    "struct Row { v: int }
     struct Store { items: [Row] }
     impl Store { subscript live() -> Row { yield &self.items[0]; } }
     let s = Store { items: [Row { v: 1 }] };
     let it = s.live;
     for v in it { s.items.push(Row { v: 9 }); }",
);

// a bound pull name iterates the same places `s.live` would inline.
// r[verify mimas.for.place-iter]
test_run!(
    place_generator_bound_iterator_writes,
    "struct Row { v: int }
     struct Store {
         items: [Row],
         gone: [bool],
     }
     impl Store {
         subscript live() -> Row {
             for i in 0 .. self.items.len() {
                 if !self.gone[i] { yield &self.items[i]; }
             }
         }
     }
     let s = Store { items: [Row { v: 1 }, Row { v: 2 }], gone: [false, false] };
     let it = s.live;
     for v in it { v.v += 1; }
     let got = s.items;",
    "got" => "[{ 2 }, { 3 }]",
);

// a `&` lend into the walked store is the same overlap `for r in &xs` refuses.
// r[verify mimas.for.place-iter-freeze]
test_fail!(
    place_generator_lend_overlap_fails,
    "struct Row { v: int }
     struct Store { items: [Row] }
     impl Store { subscript live() -> Row { yield &self.items[0]; } }
     fn bump(p: inout Row) { p.v += 1; }
     let s = Store { items: [Row { v: 1 }] };
     for v in s.live { bump(&s.items[0]); }",
    "struct Row { v: int }
     struct Store { items: [Row] }
     impl Store { subscript live() -> Row { yield &self.items[0]; } }
     let s = Store { items: [Row { v: 1 }] };
     for v in s.live { let p = &s.items[0]; }",
);

// `for r in &s.items` nested inside `for v in s.live` is two live loans on
// the same store.
// r[verify mimas.for.place-iter-freeze]
test_fail!(
    place_generator_nested_overlap_fails,
    "struct Row { v: int }
     struct Store { items: [Row] }
     impl Store { subscript live() -> Row { yield &self.items[0]; } }
     let s = Store { items: [Row { v: 1 }] };
     for v in s.live { for r in &s.items { r.v += 1; } }",
);

// a `for .. in &` on a value pull can't lend -- the generator yields copies.
// r[verify mimas.for.place-iter]
test_fail!(
    place_generator_borrow_of_value_iter_fails,
    "struct Store { items: [int] }
     impl Store { subscript live() -> int { yield self.items[0]; } }
     let s = Store { items: [1] };
     for v in &s.live { v += 1; }",
);

// `v += n` on a scalar place element is still the loop-variable error --
// whole-element writes go through member syntax (`v.f =`) since `v =`/`v +=`
// would rebind the loop's own binding.
// r[verify mimas.for.elem-bind]
test_fail!(
    place_generator_scalar_compound_assign_fails,
    "struct Store { items: [int] }
     impl Store { subscript live() -> int { yield &self.items[0]; } }
     let s = Store { items: [1] };
     for v in s.live { v += 10; }",
);

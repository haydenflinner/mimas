// `for r in &xs` iterates element *places*: `r` reads and writes `xs[i]`
// itself, where `for x in xs` binds a copy each pass.

// r[verify mimas.for.inplace]
test_run!(
    inplace_for_writes_elements_back,
    "let xs = [1, 2, 3];
     for r in &xs { r += 10; }
     let got = xs;",
    "got" => "[11, 12, 13]",
);

// `r = v` replaces the whole slot, not the binding. r[verify mimas.for.inplace.use]
test_run!(
    inplace_for_whole_element_assign,
    "struct Box { v: int }
     let xs = [Box { v = 1 }, Box { v = 2 }];
     for r in &xs { r = Box { v = r.v * 10 }; }
     let got = xs[0].v + xs[1].v;",
    "got" => "30",
);

// a field write through the place lands in the element. r[verify mimas.for.inplace.use]
test_run!(
    inplace_for_field_write_through,
    "struct Box { v: int }
     let xs = [Box { v = 1 }, Box { v = 2 }];
     for r in &xs { r.v += 100; }
     let got = xs[0].v + xs[1].v;",
    "got" => "203",
);

// `for x in xs` binds each element's value, not its slot: a whole-element
// write is rejected, and `&` is the opt-in for slot writes. Field writes
// through the copy alias heap storage today either way.
// r[verify mimas.for.inplace]
test_fail!(
    plain_for_bars_whole_element_writes,
    "let xs = [1, 2];
     for x in xs { x = 9; }",
);

// a subscript write through the place lands in the element.
// r[verify mimas.for.inplace.use]
test_run!(
    inplace_for_subscript_write_through,
    "let xs = [[1, 2], [3, 4]];
     for r in &xs { r[0] += 10; }
     let got = xs;",
    "got" => "[[11, 2], [13, 4]]",
);

// reads of `r` see the slot's current value. r[verify mimas.for.inplace.use]
test_run!(
    inplace_for_reads_the_live_element,
    "let xs = [1, 2, 3];
     let got = 0;
     for r in &xs { got += r; r = 0; }
     let done = xs;",
    "got" => "6",
    "done" => "[0, 0, 0]",
);

// an empty list gives the body no element places. r[verify mimas.for.inplace.operand]
test_run!(
    inplace_for_empty_list,
    "let xs: [int] = [];
     let n = 0;
     for r in &xs { n += 1; }
     let got = n;",
    "got" => "0",
);

// `f(&r)` re-lends the element place through the call boundary, like a
// struct field's place today. r[verify mimas.for.inplace.use]
test_run!(
    inplace_for_relends_into_inout,
    "fn bump(x: inout int) { x += 1; }
     let xs = [1, 2, 3];
     for r in &xs { bump(&r); }
     let got = xs;",
    "got" => "[2, 3, 4]",
);

// `break` and `continue` behave as in any `for`.
// r[verify mimas.for.inplace.use]
test_run!(
    inplace_for_break_and_continue,
    "let xs = [1, 2, 3, 4];
     for r in &xs {
         if r == 3 { break; }
         if r == 2 { continue; }
         r += 10;
     }
     let got = xs;",
    "got" => "[11, 2, 3, 4]",
);

// `collect r` gathers the elements' values -- the place reads on the way
// out, so collected results are copies like any `for` yields.
// r[verify mimas.for.inplace.use]
test_run!(
    inplace_for_collect_reads_elements,
    "let xs = [1, 2];
     let got = for r in &xs { collect r * 10 };",
    "got" => "[10, 20]",
);

// element places iterate a field's list too. r[verify mimas.for.inplace.operand]
test_run!(
    inplace_for_over_a_field,
    "struct Bag { xs: [int] }
     let b = Bag { xs = [1, 2] };
     for r in &b.xs { r += 1; }
     let got = b.xs;",
    "got" => "[2, 3]",
);

// `&xs[i]` iterates the nested list's own element places.
// r[verify mimas.for.inplace.operand]
test_run!(
    inplace_for_over_an_index,
    "let xs = [[1, 2], [3, 4]];
     for r in &xs[1] { r *= 10; }
     let got = xs;",
    "got" => "[[1, 2], [30, 40]]",
);

// the operand is evaluated once, before the loop -- a call operand is
// legal, it just has no name to freeze.
// r[verify mimas.for.inplace.operand]
test_run!(
    inplace_for_evaluates_operand_once,
    "let n = 0;
     fn next() -> [int] { n += 1; [1, 2] }
     for r in &next() { r += 1; }
     let got = n;",
    "got" => "1",
);

// the iterated root freezes for the loop's extent. r[verify mimas.for.inplace-freeze]
test_fail!(
    inplace_for_rejects_push,
    "let xs = [1, 2];
     for r in &xs { xs.push(3); }",
);
test_fail!(
    inplace_for_rejects_clear,
    "let xs = [1, 2];
     for r in &xs { xs.clear(); }",
);
test_fail!(
    inplace_for_rejects_index_write,
    "let xs = [1, 2];
     for r in &xs { xs[0] = 9; }",
);
test_fail!(
    inplace_for_rejects_reassign,
    "let xs = [1, 2];
     for r in &xs { xs = []; }",
);
test_fail!(
    inplace_for_rejects_compound_reassign,
    "let xs = [1, 2];
     for r in &xs { xs += [3]; }",
);

// `xs[i].remove()` shifts the iterated collection -- the same freeze
// `xs.push` gets, whichever side of the lent path it lands on.
// r[verify mimas.for.inplace-freeze]
test_fail!(
    inplace_for_rejects_slot_removal,
    "let xs = [1, 2];
     for r in &xs { xs[0].remove(); }",
);
test_fail!(
    inplace_for_rejects_removal_above_the_loan,
    "let xs = [[1], [2]];
     for r in &xs[0] { xs[0].remove(); }",
);

// an element place can't remove its own slot mid-loop -- the length
// mutation is the freeze's, not the descriptor's to offer.
// r[verify mimas.for.inplace.use]
test_fail!(
    inplace_for_element_has_no_remove,
    "let xs = [1, 2];
     for r in &xs { r.remove(); }",
);

// no `&` lends of the iterated root or any slot under it.
// r[verify mimas.for.inplace-freeze]
test_fail!(
    inplace_for_rejects_lending_the_root,
    "fn f(x: inout int) { x += 1; }
     let xs = [1, 2];
     for r in &xs { f(&xs[0]); }",
);
test_fail!(
    inplace_for_rejects_lending_the_whole,
    "fn f(x: inout [int]) { }
     let xs = [1, 2];
     for r in &xs { f(&xs); }",
);

// two in-place loops over one root would hold two sets of live element
// places on it -- the second is the lend the freeze refuses.
// r[verify mimas.for.inplace-freeze]
test_fail!(
    inplace_for_rejects_nested_same_root,
    "let xs = [1, 2];
     for r in &xs { for q in &xs { q += 1; } }",
);
// overlapping paths count too: a wider loan reaches the frozen slots, a
// narrower one lends inside the frozen region. r[verify mimas.for.inplace-freeze]
test_fail!(
    inplace_for_rejects_nested_wider_loan,
    "let xs = [[1], [2]];
     for r in &xs[0] { for q in &xs { q += [0]; } }",
);
test_fail!(
    inplace_for_rejects_nested_narrower_loan,
    "let xs = [[1], [2]];
     for r in &xs { for q in &xs[0] { q += 1; } }",
);
// a write at-or-above the lent place stales the loan; writes under it
// reach the iterated storage itself and are refused the same way.
// r[verify mimas.for.inplace-freeze]
test_fail!(
    inplace_for_rejects_slot_replacement,
    "let xs = [[1], [2]];
     for r in &xs[0] { xs[0] = [9]; }",
);
test_fail!(
    inplace_for_rejects_root_reassign_under_index,
    "let xs = [[1], [2]];
     for r in &xs[0] { xs = []; }",
);
test_fail!(
    inplace_for_rejects_lending_above_the_loan,
    "fn f(x: inout [int]) { }
     let xs = [[1], [2]];
     for r in &xs[0] { f(&xs); }",
);
test_fail!(
    inplace_for_rejects_lending_the_field,
    "struct Bag { xs: [int] }
     fn f(x: inout int) { }
     let b = Bag { xs = [1, 2] };
     for r in &b.xs { f(&b.xs[0]); }",
);
test_run!(
    inplace_for_nests_other_roots,
    "let xs = [1];
     let ys = [10];
     for r in &xs { for q in &ys { q += r; } }
     let got = ys[0];",
    "got" => "11",
);

// a by-value loop reading the frozen list stays legal -- only writes and
// lends freeze. r[verify mimas.for.inplace-freeze]
test_run!(
    inplace_for_allows_by_value_inner_loop,
    "let xs = [1, 2];
     let t = 0;
     for r in &xs { for x in xs { t += x; } }
     let got = t;",
    "got" => "6",
);

// `for r in &p` on a kept place re-lends the place instead of lending
// list elements -- rejected rather than writing a read-copy.
// r[verify mimas.for.inplace.operand]
test_fail!(
    inplace_for_rejects_a_place_operand,
    "let xs = [1, 2];
     let p = &xs[0];
     for r in &p { r += 1; }",
);
// an `inout` param's real slots have no name in the callee, so
// `for r in &g` is fully guarded there -- the loop writes the caller's
// list in place. r[verify mimas.for.inplace.operand]
test_run!(
    inplace_for_over_an_inout_operand,
    "fn f(g: inout [int]) { for r in &g { r += 1; } }
     let xs = [1, 2];
     f(&xs);
     let got = xs;",
    "got" => "[2, 3]",
);
test_run!(
    inplace_for_over_an_inout_field,
    "struct Bag { xs: [int] }
     fn f(g: inout Bag) { for r in &g.xs { r += 1; } }
     let b = Bag { xs = [1, 2] };
     f(&b);
     let got = b.xs;",
    "got" => "[2, 3]",
);
// the freeze holds there too: `g = v` and `g.xs = v` are the same
// slot replacement the loan can't survive.
// r[verify mimas.for.inplace-freeze]
test_fail!(
    inplace_for_inout_rejects_param_reassign,
    "fn f(g: inout [int]) { for r in &g { g = [9]; } }
     let xs = [1];
     f(&xs);",
);
test_fail!(
    inplace_for_inout_rejects_field_reassign,
    "struct Bag { xs: [int] }
     fn f(g: inout Bag) { for r in &g.xs { g.xs = [9]; } }
     let b = Bag { xs = [1] };
     f(&b);",
);

// destructuring can't lend one place, and a dict has no element places
// to lend. r[verify mimas.for.inplace.operand]
test_fail!(
    inplace_for_rejects_a_pattern,
    "let xs = [(1, 2), (3, 4)];
     for (a, b) in &xs { }",
);
test_fail!(
    inplace_for_rejects_a_dict,
    "let d = {a = 1};
     for r in &d { }",
);

// a `const` has no place to lend. r[verify mimas.for.inplace.operand]
test_fail!(
    inplace_for_rejects_a_const,
    "const XS = [1, 2];
     for r in &XS { r += 1; }",
);

// a subscript call's place is one its `inout` arm *yielded*, not a
// list's own slots -- lending elements out of it waits on machinery
// that doesn't exist yet, so the error is explicit.
// r[verify mimas.for.inplace.operand]
test_fail!(
    inplace_for_rejects_a_computed_subscript,
    "struct G { cells: [int] }
     impl G { subscript live() -> [int] { inout { yield self.cells; } } }
     let g = G { cells = [1] };
     for r in &g.live { r += 1; }",
);

// a `for` variable still names shared storage, so its element places
// are as lendable as `let p = &x[i]` already was.
// r[verify mimas.for.inplace.operand]
test_run!(
    inplace_for_over_a_for_variable,
    "let xss = [[1], [2]];
     for x in xss { for r in &x { r += 1; } }
     let got = xss;",
    "got" => "[[2], [3]]",
);

// `for r in &xs` reads through closures: the element places write the
// shared list, like `f(&xs[0])` inside one does.
// r[verify mimas.for.inplace.use]
test_run!(
    inplace_for_writes_through_a_closure_capture,
    "let xs = [1, 2];
     let go = || { for r in &xs { r += 1; } };
     go();
     let got = xs;",
    "got" => "[2, 3]",
);

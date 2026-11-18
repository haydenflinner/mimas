#[macro_use]
mod test_runner;

// `x: inout T` lends the caller's place to the callee: reads and writes
// reach the slot the `&` argument projected.

// r[verify mimas.inout.param]
test_run!(
    inout_writes_back_through_struct_field,
    "struct Box { value: int }
     fn bump(x: inout int) { x += 1; }
     let b = Box { value = 1 };
     bump(&b.value);
     let got = b.value;",
    "got" => "2",
);

// r[verify mimas.inout.param]
test_run!(
    inout_reads_the_lent_place,
    "struct Box { value: int }
     fn peek(x: inout int) -> int { x }
     let b = Box { value = 41 };
     let got = peek(&b.value);",
    "got" => "41",
);

// r[verify mimas.inout.param]
test_run!(
    inout_writes_back_through_subscript_place,
    "struct Grid { cells: [int] }
     impl Grid {
        subscript(i: int) -> int {
            inout { yield self.cells[i]; }
        }
     }
     fn bump(x: inout int) { x += 1; }
     let g = Grid { cells = [1] };
     bump(&g[0]);
     let got = g.cells[0];",
    "got" => "2",
);

// the accessor's side effects run when the place is lent, and the loan
// ends with the call -- the callee keeps no handle it could touch later.
// r[verify mimas.inout.param]
test_run!(
    inout_loan_ends_at_the_call,
    "struct Grid { cells: [int], log: [str] }
     impl Grid {
        subscript(i: int) -> int {
            inout {
                self.log.push(\"lend\");
                yield self.cells[i];
            }
        }
     }
     fn bump(x: inout int) { x += 1; }
     let g = Grid { cells = [3], log = [] };
     bump(&g[0]);
     let log = g.log.len();
     let got = g.cells[0];",
    "log" => "1",
    "got" => "4",
);

// a builtin list slot is a place too -- `&xs[i]` lends it, removable
// descriptor and all (the param only asks for read/write).
// r[verify mimas.inout.param]
test_run!(
    inout_writes_back_through_list_slot,
    "fn bump(x: inout int) { x += 1; }
     let xs = [1, 2];
     bump(&xs[0]);
     let got = xs[0];",
    "got" => "2",
);

// `x` reads as the element, so methods on the lent place's value act on
// the caller's storage -- `xs` here is the shared list, not a copy.
test_run!(
    inout_method_writes_through,
    "struct Box { xs: [int] }
     fn app(xs: inout [int]) { xs.push(4); }
     let b = Box { xs = [1] };
     app(&b.xs);
     let got = b.xs.len();",
    "got" => "2",
);

// an inout param lends onward: `g(&x)` passes the same place through.
// r[verify mimas.inout.param]
test_run!(
    inout_lends_onward,
    "fn inner(x: inout int) { x += 1; }
     fn outer(x: inout int) { inner(&x); }
     let xs = [5];
     outer(&xs[0]);
     let got = xs[0];",
    "got" => "6",
);

// `let p = &x` re-lends the parameter's place to a local alias.
test_run!(
    inout_let_alias_writes_back,
    "fn bump(x: inout int) { let p = &x; p += 5; }
     let xs = [1];
     bump(&xs[0]);
     let got = xs[0];",
    "got" => "6",
);

// named-argument calls take the `&` the same way.
test_run!(
    inout_named_argument,
    "fn scale(x: inout int, k: int) { x *= k; }
     let xs = [3];
     scale(k = 4, x = &xs[0]);
     let got = xs[0];",
    "got" => "12",
);

// r[verify mimas.inout.param]
test_fail!(
    inout_rejects_temporary_argument,
    "fn bump(x: inout int) { x += 1; }
     bump(1);",
);

// a plain value has no slot to write back through -- the error must
// point at the missing `&`. r[verify mimas.inout.param]
test_fail!(
    inout_rejects_missing_ampersand,
    "struct Box { value: int }
     fn bump(x: inout int) { x += 1; }
     let b = Box { value = 1 };
     bump(b.value);",
);

test_fail!(
    inout_rejects_non_place_borrow,
    "fn bump(x: inout int) { x += 1; }
     let xs = [1];
     bump(&(xs[0] + 1));",
);

// disjoint field lends under one root are legal -- exclusivity compares
// paths, not roots. r[verify mimas.inout.exclusive]
test_run!(
    inout_disjoint_field_loans_pass,
    "struct Box { a: int, b: int }
     fn swap(x: inout int, y: inout int) { let t = x; x = y; y = t; }
     let s = Box { a = 1, b = 2 };
     swap(&s.a, &s.b);
     let got = s.a * 10 + s.b;",
    "got" => "21",
);

// the same path lent twice still collides.
// r[verify mimas.inout.exclusive]
test_fail!(
    inout_rejects_same_path_loans,
    "struct Box { a: int, b: int }
     fn swap(x: inout int, y: inout int) { let t = x; x = y; y = t; }
     let s = Box { a = 1, b = 2 };
     swap(&s.a, &s.a);",
);

// two `&` args that never share a root are fine even inside one call --
// this one must fail for a different reason, so check the distinct-root
// shape passes first by way of the pair below.
test_run!(
    inout_distinct_roots_pass,
    "fn bump(x: inout int, y: inout int) { x += 1; y += 1; }
     let xs = [1, 2];
     let ys = [3];
     bump(&xs[0], &ys[0]);
     let got = xs[0] + ys[0];",
    "got" => "6",
);

// r[verify mimas.inout.exclusive]
test_fail!(
    inout_rejects_shared_root_loans,
    "fn both(x: inout int, y: inout int) { x = y; }
     let xs = [1, 2];
     both(&xs[0], &xs[1]);",
);

// a sibling argument is only dangerous where the callee writes: `other.n`
// never touches the lent `s.xs[]` path, so the loan is safe.
// r[verify mimas.inout.exclusive]
test_run!(
    inout_disjoint_write_alongside_loan,
    "struct Box { xs: [int], n: int }
     fn f(p: inout int, other: Box) { p += other.n; other.n = 0; }
     let s = Box { xs = [1], n = 5 };
     f(&s.xs[0], s);
     let got = s.xs[0] * 10 + s.n;",
    "got" => "60",
);

// when the callee writes a path that overlaps the lend, the sibling still
// collides. r[verify mimas.inout.exclusive]
test_fail!(
    inout_rejects_overlapping_write_alongside_loan,
    "struct Box { xs: [int], n: int }
     fn f(p: inout int, other: Box) { other.xs.push(9); }
     let s = Box { xs = [1], n = 1 };
     f(&s.xs[0], s);",
);

// r[verify mimas.inout.escape]
test_fail!(
    inout_param_escapes_through_closure,
    "fn f(x: inout int) { let g = || x; }
     let xs = [1];
     f(&xs[0]);",
);

// r[verify mimas.inout.escape]
test_fail!(
    inout_param_escapes_through_return,
    "fn f(x: inout int) -> int { return &x; }
     let xs = [1];
     f(&xs[0]);",
);

// a `let` alias of the lent place keeps the same rules: `return &p`
// hands the caller's loan back out. r[verify mimas.inout.escape]
test_fail!(
    inout_rebound_place_escapes_through_return,
    "fn f(x: inout int) -> int { let p = &x; return &p; }
     let xs = [1];
     f(&xs[0]);",
);

// `inout` is a call convention, not a type. r[verify mimas.inout.param]
test_fail!(
    inout_rejected_in_let_annotation,
    "let x: inout int = 1;",
);
test_fail!(
    inout_rejected_in_return_type,
    "fn f() -> inout int { 1 }",
);
test_fail!(
    inout_rejected_in_field_type,
    "struct Box { value: inout int }",
);
test_fail!(
    inout_param_takes_no_default,
    "fn bump(x: inout int = 0) { x += 1; }",
);

// an `inout` type may be any name the param scope resolves -- a sibling
// file's decl (the `use "…"` splice shape), a `use`-imported module member,
// or a path -- not only a local or builtin. r[verify mimas.inout.param]
#[test]
fn inout_param_type_from_host_spliced_file() {
    let kit = r#"
struct Tour { step: int }
fn tour() -> Tour { Tour { step = 0 } }
"#;
    let main = r#"
use "tour-kit";
fn next(b: inout Tour) { b.step += 1; }
struct W { tour: Tour }
let w = W { tour = tour() };
next(&w.tour);
"#;
    vm::Vm::execute_files(&[("tour-kit", kit), ("main", main)], library::std)
        .expect("spliced-file inout param type resolves");
}

// r[verify mimas.inout.param]
#[test]
fn inout_param_type_from_module_uses() {
    let kit = r#"
module kit;
pub struct Tour { step: int }
pub fn tour() -> Tour { Tour { step = 0 } }
"#;
    // `use kit::Tour` then `inout Tour`; `use kit` then `inout kit::Tour`
    let named = r#"
use kit::Tour;
use kit::tour;
fn next(b: inout Tour) { b.step += 1; }
struct W { tour: Tour }
let w = W { tour = tour() };
next(&w.tour);
"#;
    let pathed = r#"
use kit;
fn next(b: inout kit::Tour) { b.step += 1; }
struct W { tour: kit::Tour }
let w = W { tour = kit::tour() };
next(&w.tour);
"#;
    for main in [named, pathed] {
        vm::Vm::execute_files(&[("kit", kit), ("main", main)], library::std)
            .expect("module-imported inout param type resolves");
    }
}

// r[verify mimas.inout.param]
#[test]
fn inout_param_type_across_page_marks() {
    // the host splices `use "…"` pages in under `//# page:` marks -- the
    // marks bound operator shadowing, never name resolution
    let src = r#"//# page: tour-kit
struct Tour { step: int }
fn tour() -> Tour { Tour { step = 0 } }
fn to(b: inout Tour, step: int) { b.step = step; }

//# page:
use "tour-kit";
fn next(b: inout Tour) { to(&b, 7); }
struct W { tour: Tour }
let w = W { tour = tour() };
next(&w.tour);
"#;
    vm::Vm::execute(src, library::std).expect("page-spliced inout param resolves");
}

// `x: inout s` lends a place of `s`: the callee's writes check against
// the element dimension, like `x: s` checks its assignments.
// r[verify mimas.inout.param]
test_fail!(
    inout_param_writes_check_the_element_dim,
    "fn f(x: inout s) { x += 1.0 }",
);

// r[verify mimas.inout.param]
test_run!(
    inout_param_writes_keep_the_element_dim,
    "fn f(x: inout s) { x += 1.0s; }
     struct B { t: s }
     let b = B { t = 1.0s };
     f(&b.t);
     let got = b.t.to(1s);",
    "got" => "2.0",
);

// `&c` lends a whole `let` binding: the descriptor reads and writes a
// one-slot cell, and the call stores the cell back into `c` when it
// returns -- the tour kit's `let c = tour(); to(&c, 3)` shape.
// r[verify mimas.inout.binding]
test_run!(
    inout_lends_a_whole_binding,
    "struct Tour { step: int }
     fn tour() -> Tour { Tour { step = 0 } }
     fn to(b: inout Tour, step: int) { b.step = step; }
     let c = tour();
     to(&c, 3);
     let got = c.step;",
    "got" => "3",
);

// a scalar binding lends the same way -- `&n` on a bare `int`.
// r[verify mimas.inout.binding]
test_run!(
    inout_lends_a_scalar_binding,
    "fn bump(x: inout int) { x += 1; }
     let n = 0;
     bump(&n);
     bump(&n);
     bump(&n);
     let got = n;",
    "got" => "3",
);

// `&w` on a function's own param -- the `tick(w) -> W` game-loop shape
// where helpers take `inout` of the world struct.
// r[verify mimas.inout.binding]
test_run!(
    inout_lends_a_param_binding,
    "struct Tour { step: int }
     struct W { tour: Tour }
     fn bump(w: inout W) { w.tour.step += 1; }
     fn tick(w: W) -> W { bump(&w); bump(&w); w }
     let w = tick(W { tour = Tour { step = 0 } });
     let got = w.tour.step;",
    "got" => "2",
);

// a top-level `let` lends from inside a function's body.
// r[verify mimas.inout.binding]
test_run!(
    inout_lends_a_global_binding,
    "fn bump(x: inout int) { x += 1; }
     let g = 0;
     fn f() { bump(&g); }
     f();
     f();
     let got = g;",
    "got" => "2",
);

// a whole `[T]` binding lends; the callee's mutation lands in the list.
// r[verify mimas.inout.binding]
test_run!(
    inout_lends_a_list_binding,
    "fn push(l: inout [int]) { l.push(9); }
     let xs = [1, 2];
     push(&xs);
     let got = xs.len();",
    "got" => "3",
);

// writes land when the call returns: a later read sees them, and a
// later loan starts from them. r[verify mimas.inout.binding]
test_run!(
    inout_writebacks_are_sequential,
    "fn bump(x: inout int) { x += 10; }
     fn read(x: int) -> int { x }
     let n = 1;
     bump(&n);
     let r = read(n);
     bump(&n);
     let got = n + r;",
    "got" => "32",
);

// `let p = &c` on a bare binding has no call boundary to write back at.
// r[verify mimas.inout.binding]
test_fail!(
    inout_binding_lend_is_call_scoped,
    "fn bump(x: inout int) { x += 1; }
     let n = 0;
     let p = &n;",
);

// `&c` of a captured binding would write the closure's copy back, never
// the original. r[verify mimas.inout.escape]
test_fail!(
    inout_binding_lend_rejects_captures,
    "fn bump(x: inout int) { x += 1; }
     fn f() {
         let n = 0;
         let g = || bump(&n);
         g();
     }",
);

// a `for` variable's binding is the loop's, not the body's.
// r[verify mimas.inout.binding]
test_fail!(
    inout_binding_lend_rejects_loop_vars,
    "fn bump(x: inout int) { x += 1; }
     for n in [1, 2] { bump(&n); }",
);

// two whole-binding lends of the same root alias one slot.
// r[verify mimas.inout.exclusive]
test_fail!(
    inout_binding_lends_stay_exclusive,
    "fn swap(a: inout int, b: inout int) { let t = a; a = b; b = t; }
     let n = 1;
     swap(&n, &n);",
);

// `&c` of a plain `c` can't fill an `inout s` slot any more than `c`
// could fill `x: s` -- the place clashes on its element.
// r[verify mimas.inout.binding]
test_fail!(
    inout_binding_lend_checks_the_element_dim,
    "fn f(x: inout s) { x += 1.0s; }
     let c = 0.0;
     f(&c);",
);

// `fn reset(inout self)` lends the caller's receiver slot: `b.reset()` is
// `reset(&b)` -- `self = next` inside replaces the caller's variable.
// r[verify mimas.inout.self]
test_run!(
    inout_self_rebinds_the_receiver,
    "struct Box { v: int }
     impl Box {
         fn reset(inout self) { self = Box { v = 0 } }
     }
     let b = Box { v = 42 };
     b.reset();
     let got = b.v;",
    "got" => "0",
);

// partial writes through `inout self` reach the caller's slot too --
// `self` is the same `__Place` an `x: inout T` param holds.
// r[verify mimas.inout.self]
test_run!(
    inout_self_member_writes_still_apply,
    "struct Box { v: int }
     impl Box {
         fn bump(inout self) { self.v += 10 }
     }
     let b = Box { v = 1 };
     b.bump();
     let got = b.v;",
    "got" => "11",
);

// the receiver lend is path-granular like a `&` argument: `p.a.merge(&p.b)`
// loans two disjoint fields of one root.
// r[verify mimas.inout.self]
// r[verify mimas.inout.exclusive]
test_run!(
    inout_self_lends_disjoint_sibling_paths,
    "struct Box { v: int }
     impl Box {
         fn merge(inout self, other: inout Box) { self = other }
     }
     struct Pair { a: Box, b: Box }
     let p = Pair { a = Box { v = 1 }, b = Box { v = 2 } };
     p.a.merge(&p.b);
     let got = p.a.v;",
    "got" => "2",
);

// the lent receiver overlaps a `&` argument from the same root.
// r[verify mimas.inout.self]
// r[verify mimas.inout.exclusive]
test_fail!(
    inout_self_overlaps_same_root_args,
    "struct Box { v: int }
     impl Box {
         fn merge(inout self, other: inout Box) { self = other }
     }
     let b = Box { v = 1 };
     b.merge(&b);",
);

// a temporary has no slot to write back.
// r[verify mimas.inout.self]
test_fail!(
    inout_self_rejects_a_temporary_receiver,
    "struct Box { v: int }
     impl Box {
         fn reset(inout self) { self = Box { v = 0 } }
     }
     fn mk() -> Box { Box { v = 9 } }
     mk().reset();",
);

// `xs[0].reset()` lends the element place, which the iteration freeze
// counts like any other write.
// r[verify mimas.inout.self]
test_fail!(
    inout_self_element_lend_hits_the_loop_freeze,
    "struct Box { v: int }
     impl Box {
         fn reset(inout self) { self = Box { v = 0 } }
     }
     let xs = [Box { v = 1 }];
     for x in xs { xs[0].reset() }",
);

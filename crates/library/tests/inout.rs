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

// r[verify mimas.inout.exclusive]
test_fail!(
    inout_rejects_aliased_loans,
    "struct Box { a: int, b: int }
     fn swap(x: inout int, y: inout int) { let t = x; x = y; y = t; }
     let s = Box { a = 1, b = 2 };
     swap(&s.a, &s.b);",
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

// a sibling argument reading the lent root would see the loan mid-write.
// r[verify mimas.inout.exclusive]
test_fail!(
    inout_rejects_value_alongside_loan,
    "struct Box { xs: [int], n: int }
     fn f(p: inout int, other: Box) { p += other.n; }
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

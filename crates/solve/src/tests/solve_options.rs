use super::solve_test_utils::*;
use crate::components::Ty::*;

test_ty!(
    explicate,
    "let a: int? = null;
     let b: bool? = false;",
    "a" => option!(Int),
    "b" => option!(Bool),
);
test_ty!(
    option_array,
    "let a: [int]? = null;",
    "a" => option!(array!(Int)),
);
test_ty!(
    option_dict,
    "let a: {int}? = null;",
    "a" => option!(dictionary!(Int)),
);
test_ty!(
    array_of_optional,
    "let a: [int?] = [0, null, 1];",
    "a" => array!(option!(Int)),
);
test_ty!(
    dict_of_optional,
    "let a: {int?} = { x : 0, y : null };",
    "a" => dictionary!(option!(Int)),
);
test_ty!(
    coerce_from_if,
    "if true { 0 } else { null }" => option!(Int)
);
// an else arm that fails to *solve* must surface its error -- the
// option-coercion retry used to re-query the half-solved node, get a
// memoized Ok, and swallow it, leaving unresolved idents that panicked
// the emitter (fuzz: `x.year year < y.year` parses `year < y.year{...}`
// as the if's unbraced body).
test_fail!(
    else_arm_solve_error_not_swallowed,
    "struct S { year: int }
     fn f(first: S) -> S? {
         if true {
             null
         } else {
             let s: S? = null;
             if s!.year year < first.year { s } else { first }
         }
     }"
);
test_ty!(
    coerce_from_loop,
    "loop { break 0; break null; }" => option!(Int)
);
test_ty!(
    flattening,
    "let a = if true { 0 } else { null };
    let b = if true { a } else { null };
    let c = if true { b } else { null };",
    "c" => option!(Int)
);
test_ty!(unwrap, "let a: int? = 0;", "a!" => Int);

// A trailing `?` after a call or square access is `mimas.propagate`: inside a
// `T?` function it is the option's value, and the function returns null early.
test_ty!(
    postfix_hook_after_square,
    "let a: [int?] = [1, null, 3];
     fn second() -> int? { let v = a[1]?; v }",
    "second()" => option!(Int),
);
test_ty!(
    postfix_hook_after_call,
    "fn maybe() -> int? { 5 }
     fn plus() -> int? { maybe()? + 1 }",
    "plus()" => option!(Int),
);
test_ty!(
    unwrap_through_dot,
    "struct Foo { x: int }
     let foo: Foo? = Foo { x = 0 };",
    "foo!.x" => Int,
);

test_ty!(
    t_against_option,
    "let a: int = 0;
     let b: int? = a;",
    "b" => option!(Int)
);
test_fail!(unwrap_non_option, "let a = 0; let b = a!;");
test_fail!(
    option_against_t,
    "let a: int? = null;
     let b: int = a;"
);
test_fail!(
    coalesce_mismatched_types,
    "let a: int? = null; let b = a ?? \"default\";"
);

test_fail!(
    null_is_permanent,
    "let a = null;
    if true {
        a = 1;
    } else {
        a = null;
    };"
);

test_fail!(
    option_ordering_error,
    "let a: int = 0;
    let b: int? = null;
    let c = a > b;"
);

test_fail!(
    option_ordering_error_reversed,
    "let a: int = 0;
    let b: int? = null;
    let c = b > a;"
);

test_ty!(
    match_option_null_and_bind_exhaustive,
    "let x: int? = 0;",
    "match x { null => 0, n? => n }" => Int,
);

test_ty!(
    match_option_null_and_wildcard_exhaustive,
    "let x: int? = 0;",
    "match x { null => 0, _ => 1 }" => Int,
);

test_ty!(
    match_option_with_panic_terminator,
    "let x: int? = 0;",
    "match x { n? => n, ! }" => Int,
);

test_fail!(
    match_option_only_bind_non_exhaustive,
    "let x: int? = 0; let y = match x { n? => n };"
);

test_fail!(
    match_option_only_null_non_exhaustive,
    "let x: int? = 0; let y = match x { null => 0 };"
);

test_fail!(coalesce_assign_on_non_option, "let a = 0; a ??= 1;");

// `x == null` where x can never hold null names the missing `?` on the
// declaration rather than reporting a bare mismatch on the `null`.
#[test]
fn never_null_eq_message() {
    let _t = TestResetter;
    let e = TEST_SESSION
        .with(|s| {
            s.borrow_mut().run(
                "struct N { v: int }\nlet n = N { v = 0 };\nlet c = n == null;",
                "test",
            )
        })
        .unwrap_err();
    let rendered = format!("{e:?}");
    assert!(rendered.contains("`N` can never be `null`"), "{rendered}");
    assert!(rendered.contains("always `false`"), "{rendered}");
    assert!(rendered.contains("`N?`"), "{rendered}");
}

#[test]
fn never_null_ne_message() {
    let _t = TestResetter;
    let e = TEST_SESSION
        .with(|s| s.borrow_mut().run("let x = 3;\nlet c = x != null;", "test"))
        .unwrap_err();
    let rendered = format!("{e:?}");
    assert!(rendered.contains("`int` can never be `null`"), "{rendered}");
    assert!(rendered.contains("always `true`"), "{rendered}");
}

#[test]
fn never_null_eq_reversed() {
    let _t = TestResetter;
    let e = TEST_SESSION
        .with(|s| s.borrow_mut().run("let x = 3;\nlet c = null == x;", "test"))
        .unwrap_err();
    let rendered = format!("{e:?}");
    assert!(rendered.contains("`int` can never be `null`"), "{rendered}");
}

// the field-declaration case: `self.head == null` where head was
// declared `Node` instead of `Node?`.
test_fail!(
    never_null_eq_method_field,
    "struct L { head: int }
     impl L {
         fn is_empty(self) -> bool { self.head == null }
     }"
);

// `null` in a non-compare position gets the same `?` advice through
// the general unification error.
#[test]
fn null_mismatch_let_message() {
    let _t = TestResetter;
    let e = TEST_SESSION
        .with(|s| s.borrow_mut().run("let a: int = null;", "test"))
        .unwrap_err();
    let rendered = format!("{e:?}");
    assert!(rendered.contains("`int` is never `null`"), "{rendered}");
    assert!(rendered.contains("`int?`"), "{rendered}");
}

test_fail!(null_mismatch_argument, "fn f(a: int) {} f(null);");

// controls: options still compare to `null`, and `null == null` is fine.
test_success!(null_eq_option_ok, "let a: int? = null; let c = a == null;");
test_success!(null_eq_null_ok, "let c = null == null;");

// a `.member` on a `T?` names the unwrap spellings, not "has no fields"
// r[verify mimas.option.access]
#[test]
fn option_member_access_names_the_unwrap() {
    let _t = TestResetter;
    let e = TEST_SESSION
        .with(|s| {
            s.borrow_mut().run(
                "fn find_all(xs: [str]) -> [[str]]? { [] }\nlet m = find_all([\"a\"]).flatten();",
                "test",
            )
        })
        .unwrap_err();
    let rendered = format!("{e:?}");
    assert!(rendered.contains("this is an option"), "{rendered}");
    assert!(rendered.contains("unwrap it with `!` or `?`"), "{rendered}");
}

// a `T?` where a settled type is wanted names the unwrap spellings
#[test]
fn option_used_as_settled_names_the_unwrap() {
    let _t = TestResetter;
    let e = TEST_SESSION
        .with(|s| {
            s.borrow_mut()
                .run("let d = { a : 1 };\nlet x: int = d[\"a\"];", "test")
        })
        .unwrap_err();
    let rendered = format!("{e:?}");
    assert!(rendered.contains("this is an option"), "{rendered}");
    assert!(rendered.contains("unwrap it with `!` or `?`"), "{rendered}");
}

// controls: `?` access still unwraps, and option-to-option binds fine.
test_success!(
    option_member_after_unwrap_ok,
    "let d = { a : [1] };\nlet x = d[\"a\"]?[0];"
);
test_success!(option_into_option_ok, "let a: int? = 1; let b: int? = a;");

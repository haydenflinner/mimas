#[macro_use]
mod vm_test_utils;

use vm::Captured::*;
use vm::Vm;

test_vm!(
    coalesce,
    "let present: int? = 3;
     let absent: int? = null;
     let s: str? = null;",
    "present ?? 99" => Int(3),
    "absent ?? 99" => Int(99),
    r#"s ?? "default""# => str!("default"),
);

test_vm!(
    coalesce_chained,
    "let a: int? = null;
     let b: int? = null;
     let c: int? = 7;",
    "a ?? b ?? 0" => Int(0),
    "a ?? (c ?? -1)" => Int(7),
);

// `??` binds looser than `+`, so `a ?? 2 + 3` is `a ?? (2 + 3)`
test_vm!(
    coalesce_precedence,
    "let a: int? = null;",
    "a ?? 2 + 3" => Int(5),
    "(a ?? 5) * 2" => Int(10),
    "(a ?? 5) + 3" => Int(8),
);

test_vm!(
    coalesce_result_is_non_optional,
    "let a: int? = null;
     let r: int = a ?? 7;",
    "r + 1" => Int(8),
);

// T? assigned-through stays T?, no nesting
test_vm!(
    option_assign_through_flattens,
    "let a: int? = 5;
     let b: int? = a;
     let spaced: int ? ? = a;",
    "b ?? -1" => Int(5),
    "spaced ?? -1" => Int(5),
);

test_vm!(
    option_flatten_null_rides_through,
    "let a: int? = null;
     let b: int? = a;",
    "b ?? -1" => Int(-1),
);

test_vm!(
    option_fn_return_flatten,
    "fn maybe() -> int? { 5 }
     let x: int? = maybe();",
    "x ?? -1" => Int(5),
);

test_vm!(
    index_option,
    "let foo: [int]? = [0];
    let bar: [int]? = null;",
    "foo?[0]" => Int(0),
    "bar?[0]" => Null,
);

test_vm!(
    index_unwrap,
    "let foo: [int]? = [0];",
    "foo![0]" => Int(0),
);

// the `?` taints the trailing array drills, so a null from the `?[0]` short-circuit rides through
// the plain `[0]` and comes out null rather than faulting
test_vm!(
    taint_ride_null_short_circuits,
    "let a: [~{[[int]]}] = [~{ foo = [[7]] }];
     let b: [~{[[int]]}] = [~{}];",
    "a[0][\"foo\"]?[0][0]" => Int(7),
    "b[0][\"foo\"]?[0][0]" => Null,
);

// taint through a `.field` drill: present rides to the value, null short-circuits the chain
test_vm!(
    taint_through_dot_field,
    "struct Row { cells: [int] }
     let rows: [Row]? = [Row { cells = [4, 5] }];
     let empty: [Row]? = null;",
    "rows?[0].cells[1]" => Int(5),
    "empty?[0].cells[1]" => Null,
);

test_vm!(
    dot_chain_deep,
    "struct A { b: B? }
     struct B { c: C? }
     struct C { v: int }
     let inner = C { v = 9 };
     let mid = B { c = inner };
     let present = A { b = mid };
     let nulled = A { b = null };
     let midnull = A { b = B { c = null } };",
    "present.b?.c?.v ?? -1" => Int(9),
    "nulled.b?.c?.v ?? -1" => Int(-1),
    "midnull.b?.c?.v ?? -1" => Int(-1),
);

test_vm!(
    index_chain,
    "let m: [[int]?]? = [[1, 2], null];",
    "m?[0]?[1] ?? -1" => Int(2),
    "m?[1]?[0] ?? -1" => Int(-1),
);

test_vm!(
    mixed_dot_index,
    "struct Box { items: [int] }
     let present: Box? = Box { items = [10, 20] };
     let absent: Box? = null;",
    "present?.items[0] ?? -1" => Int(10),
    "absent?.items[0] ?? -1" => Int(-1),
);

// optional-on-the-left runs null-safely against a non-optional rhs
test_vm!(
    eq_opt_lhs,
    "let a: int = 5;
     let present: int? = 5;
     let absent: int? = null;",
    "present == a" => Bool(true),
    "absent == a" => Bool(false),
    "absent != a" => Bool(true),
);

test_vm!(
    eq_opt_both_sides,
    "let five: int? = 5;
     let other: int? = 5;
     let n1: int? = null;
     let n2: int? = null;",
    "five == other" => Bool(true),
    "n1 == n2" => Bool(true),
    "five == n1" => Bool(false),
);

// `a == b` (int vs int?) and `b == a` are both legal -- equality is symmetric, and
// comparing null against a concrete value is just false.
test_vm!(
    eq_asymmetry_soundness_guard,
    "let a: int = 5;
     let b: int? = null;",
    "a == b" => Bool(false),
);

test_vm!(
    result_auto_wrap_then_unwrap,
    "fn foo() -> int! { 42 }",
    "foo()!" => Int(42),
);

test_vm!(
    result_let_annotation_unwrap,
    "let a: int! = 7;",
    "a!" => Int(7),
);

test_vm!(
    result_absolve_passes_ok_through,
    "fn foo() -> int! { 7 }",
    "foo() absolve |_| 0" => Int(7),
);

test_vm!(
    result_absolve_handles_raise,
    "fn foo() -> int! { raise \"bad\" }",
    "foo() absolve |_| 99" => Int(99),
);

test_vm!(
    result_absolve_reads_error,
    "fn foo() -> str! { raise \"oops\" }",
    "foo() absolve |e| f\"saw: {e}\"" => str!("saw: oops"),
);

test_fail!(
    result_unwrap_raised_errors,
    "fn foo() -> int! { raise \"bad\" }\nlet TEST_VALUE = foo()!;"
);

// `raise` carries any value — the payload stays a `Val` until something asks
// for text. A raised call bound without `!`/`?`/`absolve` keeps its wrapper.
test_vm!(
    raise_any_value_keeps_payload,
    "fn die() -> int! { raise 42 }
     fn die_list() -> int! { raise [1, 2] }
     fn die_dict() -> int! { raise ~{x = 9} }",
    "die()" => Raised(Box::new(Int(42))),
    "die_list()" => Raised(Box::new(array!(Int(1), Int(2)))),
    "die_dict()" => Raised(Box::new(dict! { ~{x = Int(9)} })),
);

// `?` demotes a structured raise to null without touching the payload;
// `absolve`'s str handler gets the display string, produced cold.
test_vm!(
    raise_any_value_boundaries,
    "fn die() -> str! { raise 42 }
     fn die_list() -> str! { raise [1, 2] }",
    "die() absolve |e| f\"saw: {e}\"" => str!("saw: 42"),
    "die_list() absolve |e| e" => str!("[1, 2]"),
    "die()?" => Null,
    "die_list()?" => Null,
);

test_vm!(
    raise_nested_payload,
    "fn inner() -> int! { raise 7 }
     fn outer() -> str! { raise inner() }",
    "outer() absolve |e| e" => str!("<raised 7>"),
);

test_fail!(
    unwrap_raised_int_errors,
    "fn foo() -> int! { raise 42 }\nlet TEST_VALUE = foo()!;"
);

test_fail!(
    unwrap_raised_list_errors,
    "fn foo() -> int! { raise [1, 2] }\nlet TEST_VALUE = foo()!;"
);

// a raised structured payload survives the heap snapshot intact — the
// `SnapVal` tree recurses through the payload like any other value.
#[test]
fn raised_payload_survives_snapshot() {
    let source = r#"
fn die() -> int! { raise [1, 2] }
let held = die();
let TEST_VALUE = 0;
"#;
    let mut vm = Vm::execute(source, |_| {}).expect("compiled and ran");
    let snap = vm.snapshot().expect("snapshot");
    vm.restore(&snap).expect("restore");
    assert_eq!(
        vm.resolve_name("held"),
        Some(Raised(Box::new(array!(Int(1), Int(2))))),
    );
}

//! Macros: `macro name(pattern) { template }` and `name!(…)` calls, expanded
//! by the parser before the checker sees the program (see
//! `parse::parser::macros`).
use crate::test_runner;

test_run!(
    // r[verify mimas.macro.declare]
    // r[verify mimas.macro.expr]
    macro_expands_in_an_expression,
    "macro twice($e:expr) { $e * 2 }",
    "twice!(21)" => "42",
    // a fragment keeps its own precedence: (1 + 2) * 2, not 1 + 2 * 2
    "twice!(1 + 2)" => "6",
    // the expansion is one value inside a bigger expression
    "1 + twice!(3) * 10" => "61",
    "twice!(twice!(5))" => "20",
);

const SQUARE_OR_SUM: &str = r#"
macro pick {
    (square $x:expr) => { $x * $x }
    (sum $($x:expr),*) => { 0 $(+ $x)* }
}
"#;

test_run!(
    // r[verify mimas.macro.match]
    // r[verify mimas.macro.repeat]
    macro_rules_pick_the_first_match,
    SQUARE_OR_SUM,
    "pick!(square 7)" => "49",
    "pick!(sum 1, 2, 3, 4)" => "10",
    "pick!(sum)" => "0",
);

const MAX_OF: &str = r#"
macro max_of {
    ($x:expr) => { $x }
    ($x:expr, $($rest:expr),+) => {
        let a = $x
        let b = max_of!($($rest),+)
        if a > b { a } else { b }
    }
}
"#;

test_run!(
    // r[verify mimas.macro.recursion]
    // r[verify mimas.macro.hygiene]
    macro_recurses,
    MAX_OF,
    "max_of!(3)" => "3",
    "max_of!(3, 9, 4)" => "9",
    "max_of!(10, 2, 8, 1, 7)" => "10",
    // every level binds its own `a` and `b`
    "{ let a = 1; let b = 2; max_of!(b, a, 0) }" => "2",
);

const DOUBLE: &str = r#"
macro double($e:expr) {
    let x = $e
    x + x
}
let x = 5
"#;

test_run!(
    // r[verify mimas.macro.hygiene]
    macro_locals_dont_capture_the_call_site,
    DOUBLE,
    // the template's `x` is not the page's `x`
    "double!(x + 1)" => "12",
    "double!(x)" => "10",
    "x" => "5",
);

test_run!(
    // r[verify mimas.macro.hygiene]
    macro_names_from_the_call_site_bind_at_the_call_site,
    r#"
macro define($name:ident, $value:expr) {
    let $name = $value
}
define!(answer, 6 * 7)
"#,
    "answer" => "42",
);

test_run!(
    // r[verify mimas.macro.hygiene]
    macro_fstring_names_the_template_local,
    r#"
macro show($e:expr) {
    let v = $e
    f"v = {v}"
}
let v = 100
"#,
    "show!(v + 1)" => r#""v = 101""#,
);

// A page author's record: a struct, its constructor, and a getter per field
// that reads the struct back as a tuple.
const RECORD: &str = r#"
macro record($name:ident { $($field:ident: $ty:ty),* }) {
    struct $name { $($field: $ty),* }
    impl $name {
        fn new($($field: $ty),*) -> $name {
            $name { $($field),* }
        }
        fn parts(self) -> ($($ty),*) {
            ($(self.$field),*)
        }
    }
}

record!(Point { x: float, y: float })
record!(Pet { name: str, legs: int })
"#;

test_run!(
    // r[verify mimas.macro.stmt]
    macro_writes_items,
    RECORD,
    "Point::new(1.5, 2.0).y" => "2.0",
    "Point::new(1.5, 2.0).parts()" => "[1.5, 2.0]",
    r#"Pet::new("Rex", 4).legs"# => "4",
    r#"Pet { name = "Tweety", legs = 2 }.parts()"# => r#"["Tweety", 2]"#,
);

// A `check`-style helper that names both sides when they differ.
const EXPECT: &str = r#"
macro expect($got:expr, $want:expr) {
    let got = $got
    let want = $want
    if got == want { "ok" } else { f"expected {want}, got {got}" }
}
let got = 1
let want = 2
"#;

test_run!(
    // r[verify mimas.macro.hygiene]
    macro_check_helper,
    EXPECT,
    "expect!(2 + 2, 4)" => r#""ok""#,
    // the call site's own `got` and `want` reach the template as fragments
    "expect!(got, want)" => r#""expected 2, got 1""#,
);

/// The first error a failing source reports, with the source text under
/// its first label.
fn first_error(src: &str) -> (String, String) {
    let err = test_runner::try_execute(src).expect_err("the source should not compile");
    let report = err.0;
    let message = report.to_string();
    let label = report
        .labels()
        .and_then(|mut labels| labels.next())
        .map(|l| src[l.offset()..l.offset() + l.len()].to_string())
        .unwrap_or_default();
    (message, label)
}

#[test]
// r[verify mimas.macro.no-match]
fn macro_no_rule_matches() {
    let src = "macro pair($a:expr, $b:expr) { ($a, $b) }\nlet p = pair!(1);";
    let (message, label) = first_error(src);
    assert_eq!(message, "no rule of `pair!` matches this call");
    assert_eq!(label, "pair!(1)");
}

#[test]
// r[verify mimas.macro.call-site-errors]
fn macro_type_error_points_into_the_call() {
    let src = "macro inc($e:expr) { $e + 1 }\nlet n = inc!(\"one\");";
    let (_, label) = first_error(src);
    let call = src.find("inc!(").unwrap();
    let at = src.find(&label).unwrap();
    assert!(
        at >= call && !label.is_empty(),
        "the error should point into the call, not at {label:?}"
    );
}

#[test]
// r[verify mimas.macro.hygiene]
fn macro_locals_stay_inside() {
    let src = "macro hide() { let secret = 3 }\nhide!()\nlet n = secret;";
    let (message, label) = first_error(src);
    assert_eq!(message, "undefined variable");
    assert_eq!(label, "secret");
}

#[test]
// r[verify mimas.macro.recursion-limit]
fn macro_endless_recursion_stops() {
    let src = "macro forever($e:expr) { forever!($e) }\nlet n = forever!(1);";
    let (message, label) = first_error(src);
    assert!(message.contains("expands too deep"), "{message}");
    assert_eq!(label, "forever!(1)");
}

#[test]
// r[verify mimas.macro.declare]
fn macro_template_names_unknown_fragment() {
    let src = "macro bad($a:expr) { $b }\nlet n = 1;";
    let (message, _) = first_error(src);
    assert_eq!(message, "`$b` is not in the pattern");
}

test_run!(
    // r[verify mimas.macro.hygiene]
    macro_hygiene_keeps_variants_fields_and_shorthand,
    r#"
struct Pair { left: int, right: int }
enum Slot { Empty, Full(int) }
macro or_zero($e:expr) {
    match $e {
        Slot::Empty => 0,
        Slot::Full(v) => v,
    }
}
let v = 99
macro pair_of($e:expr) {
    let left = $e
    let right = left * 2
    Pair { left, right }
}
macro split($p:expr) {
    match $p {
        Pair { left, right } => left + right,
    }
}
let left = 7
"#,
    "or_zero!(Slot::Full(4))" => "4",
    "or_zero!(Slot::Full(v))" => "99",
    "or_zero!(Slot::Empty)" => "0",
    "pair_of!(left).right" => "14",
    "split!(Pair { left = 1, right = 2 })" => "3",
    "left" => "7",
);

test_run!(
    // r[verify mimas.macro.hygiene]
    macro_nested_expansions_keep_their_own_locals,
    r#"
macro add_one($e:expr) {
    let n = $e
    n + 1
}
macro add_two($e:expr) {
    let n = $e
    add_one!(n) + 1
}
let n = 40
"#,
    "add_two!(n)" => "42",
    "add_two!(add_one!(n))" => "43",
);

test_run!(
    // r[verify mimas.macro.stmt]
    macro_statements_join_the_block,
    r#"
macro bump($name:ident) {
    $name = $name + 1
}
fn count() -> int {
    let total = 0
    bump!(total)
    bump!(total)
    total
}
"#,
    "count()" => "2",
    "{ let k = 10; bump!(k); k }" => "11",
);

test_run!(
    // r[verify mimas.macro.call]
    macro_calls_take_any_bracket_and_leave_unwrap_alone,
    r#"
macro list_of($($x:expr),*) { [$($x),*] }
let maybe: int? = 4
let macro = 3
"#,
    "list_of![1, 2, 3]" => "[1, 2, 3]",
    "list_of!{4, 5}" => "[4, 5]",
    "list_of!()" => "[]",
    // not a macro: `maybe!` is still the unwrap
    "maybe! + 1" => "5",
    // and `macro` is still a name
    "macro * 2" => "6",
);

#[test]
// r[verify mimas.macro.call]
fn macro_from_an_earlier_file() {
    let kit = "macro squared($e:expr) {\n    let v = $e\n    v * v\n}\n";
    let page = "let v = 3\nlet TEST_VALUE = squared!(v + 1) + v;";
    let mut vm = vm::Vm::execute_files(&[("kit", kit), ("page", page)], library::std)
        .expect("the page compiles with the kit's macro");
    let value = vm.resolve_name("TEST_VALUE").expect("TEST_VALUE was bound");
    assert_eq!(format!("{value}"), "19");
}

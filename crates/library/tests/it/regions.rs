//! Each `//# page: NAME` region of a spliced source is a namespace (C2):
//! a kit and the page may each declare a name, the use's own region and then
//! the page win, `kit_a::name` reaches a region's item, and a bare name two
//! kits define is an error at the use.

use crate::test_runner::try_execute;

// k13: a kit and the page both mark `#[on] fn click`
const K13: &str = "//# page: button-kit
#[on]
fn click(me: str) -> str { \"kit handled \" + me }
fn kit_click(me: str) -> str { click(me) }
//# page: 
#[on]
fn click(me: str) -> str { \"page handled \" + me }
let ONS = fns!(on);
";

test_run!(
    // r[verify mimas.region.bare]
    // r[verify mimas.region.qualified]
    a_kit_and_the_page_each_hold_a_click,
    K13,
    "click(\"a\")" => "\"page handled a\"",
    "kit_click(\"a\")" => "\"kit handled a\"",
    "button_kit::click(\"b\")" => "\"kit handled b\"",
    "ONS.map(|s| s.name)" => "[\"click\", \"click\"]",
    "ONS.map(|s| s.page)" => "[\"button-kit\", \"\"]",
);

// k10: the page's own `fn label` and a kit's: each side calls its own
test_run!(
    // r[verify mimas.region.bare]
    each_region_calls_its_own,
    "//# page: kit-a
fn label() -> str { \"kit\" }
fn hello() -> str { \"hi \" + label() }
//# page: 
fn label() -> str { \"page\" }
",
    "hello()" => "\"hi kit\"",
    "label()" => "\"page\"",
);

// k07b: the page binds `let scale` before the kit's text; the kit's own
// call still reaches the kit's `fn scale`
test_run!(
    // r[verify mimas.region.own-first]
    a_kit_resolves_its_own_names_before_a_page_let,
    "//# page: 
let scale = 10.0
//# page: kit-a
fn scale() -> float { 2.0 }
fn doubled(x: float) -> float { x * scale() }
//# page: 
",
    "doubled(1.0)" => "2.0",
    "scale" => "10.0",
);

// k11: a kit's struct and the page's struct of one name stay apart
test_run!(
    // r[verify mimas.region.bare]
    a_kit_struct_and_a_page_struct_stay_apart,
    "//# page: kit-a
struct Card { id: str }
fn first_id(cs: [Card]) -> str { cs[0].id }
fn kit_card() -> Card { Card { id = \"k\" } }
//# page: 
struct Card { id: str, x: int }
",
    "first_id([kit_card()])" => "\"k\"",
    "Card { id = \"p\", x = 1 }.x" => "1",
);

#[test]
// r[verify mimas.region.bare]
fn a_bare_name_two_kits_define_is_an_error_naming_both() {
    let err = try_execute(
        "//# page: img
fn rect() -> int { 1 }
//# page: man-kit
fn rect() -> int { 2 }
//# page: 
let TEST_VALUE = rect();",
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("`img::rect`") && err.contains("`man_kit::rect`"), "{err}");
}

test_run!(
    // r[verify mimas.region.qualified]
    two_kits_reach_by_qualified_name,
    "//# page: img
fn rect() -> int { 1 }
//# page: man-kit
fn rect() -> int { 2 }
//# page: 
",
    "img::rect() + man_kit::rect()" => "3",
);

test_fail!(
    // r[verify mimas.region.type-dup]
    // a duplicate type inside one region is an error; so is a second fn
    duplicates_inside_one_region,
    "//# page: kit-a
struct Btn { label: str }
struct Btn { label: str, w: int }
//# page: 
let TEST_VALUE = 1;",
    "//# page: kit-a
fn f() -> int { 1 }
fn f() -> int { 2 }
//# page: 
let TEST_VALUE = 1;",
);

// C13, k03: two kits' macros of one name are a pair, each region using its
// own, and the page reaching the one kit that declares it
test_run!(
    // r[verify mimas.region.macro]
    two_kits_macros_of_one_name_are_a_pair,
    "//# page: kit-a
macro twice($e:expr) { $e * 2 }
fn a_twice(x: int) -> int { twice!(x) }
//# page: kit-b
macro twice($e:expr) { $e + $e + 100 }
fn b_twice(x: int) -> int { twice!(x) }
//# page: kit-c
macro thrice($e:expr) { $e * 3 }
//# page: 
",
    "a_twice(1)" => "2",
    "b_twice(1)" => "102",
    "thrice!(2)" => "6",
);

#[test]
// r[verify mimas.region.macro]
fn a_macro_two_kits_declare_is_an_error_at_a_page_call() {
    let err = format!(
        "{:?}",
        try_execute(
            "//# page: kit-a
macro twice($e:expr) { $e * 2 }
//# page: kit-b
macro twice($e:expr) { $e + 1 }
//# page: 
let TEST_VALUE = twice!(1);",
        )
        .unwrap_err()
    );
    assert!(err.contains("more than one included page"), "{err}");
}

#[test]
// r[verify mimas.region.macro]
fn a_second_notation_in_one_region_says_notation() {
    let err = format!(
        "{:?}",
        try_execute(
            "notation n(s: str) -> int { s.len() }
notation n(s: str) -> int { 0 }
let TEST_VALUE = n\"abc\";",
        )
        .unwrap_err()
    );
    assert!(err.contains("`notation n` is already defined"), "{err}");
}

// k04 across kits: two kits' notations of one tag are a pair too
test_run!(
    // r[verify mimas.region.macro]
    two_kits_notations_of_one_tag,
    "//# page: kit-a
notation n(s: str) -> int { s.len() }
fn a_n() -> int { n\"abc\" }
//# page: kit-b
notation n(s: str) -> int { 0 }
fn b_n() -> int { n\"abc\" }
//# page: 
",
    "a_n() + b_n()" => "3",
);

// a `::` path never reads a value: a `let dict` beside the std `dict`
// module (the host's `let page` beside `page::`)
test_run!(
    // r[verify mimas.path.not-value]
    a_path_never_reads_a_value,
    "let dict = 5;\nlet D: {str: int} = dict::new();",
    "D.len() + dict" => "5",
);

// a kit's `fn` of a std name still reaches the page when no other kit
// declares it (std ranks after every page region), and std keeps its own
test_run!(
    // r[verify mimas.region.bare]
    a_kit_name_beats_std_for_the_page,
    "//# page: kit-a
fn clamp01(x: float) -> float { 0.5 }
//# page: 
",
    "clamp01(3.0)" => "0.5",
    "[1, 2].len()" => "2",
);

// A byte offset orders positions within one source only. std's
// protocols.mim declares `struct Range` near byte ~7000; a page that
// declares its own `Range` near byte 0 and uses it past that offset used to
// resolve the use to std's `Range` ("expected Range but found Range"),
// because reading order compared the page's offset with the library file's.
// The pad sweeps the use across std's offset.
// r[verify mimas.decl.source-order]
#[test]
fn a_page_type_beats_a_library_type_at_any_offset() {
    let lib_at = library::PROTOCOLS_SRC.find("struct Range").expect("std declares Range");
    for pad in (0..=lib_at + 400).step_by(97) {
        let src = format!(
            "struct Range {{ lo: int }}\n//{}\nfn f(r: Range) -> int {{ r.lo }}\nlet TEST_VALUE = f(Range {{ lo: 3 }});",
            "x".repeat(pad)
        );
        assert!(
            crate::test_runner::try_execute(&src).is_ok(),
            "pad {pad} (std's Range at {lib_at}) failed"
        );
    }
}

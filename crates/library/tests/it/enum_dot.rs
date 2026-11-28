//! `.Down`: a variant of whatever enum the slot expects, the Swift-style
//! shorthand for `Dir::Down` (`mimas.enum.dot`).

use crate::test_runner;

const DIR: &str = "enum Dir { Up, Down, Left(int), Jump { h: int } }
enum Arrow { Up, Down }
struct Move { d: Dir, n: int }
fn f(d: Dir) -> int { match d { .Up => 1, .Down => 2, .Left(k) => k, .Jump { h } => h * 10 } }
fn opt(d: Dir?) -> int { match d { x? => f(x), _ => 0 } }
fn back() -> Dir { .Down }";

fn run(src: &str) -> String {
    test_runner::render(DIR, src)
}

fn rejected(src: &str, needle: &str) {
    let err = test_runner::try_execute(src).expect_err(&format!("`{src}` should be rejected"));
    let text = format!("{err:?}");
    assert!(text.contains(needle), "`{src}`: expected `{needle}` in:\n{text}");
}

// r[verify mimas.enum.dot]
#[test]
fn a_dot_variant_fills_the_slot_its_enum_expects() {
    // an argument, with a payload, with named fields
    assert_eq!(run("f(.Up)"), "1");
    assert_eq!(run("f(.Left(7))"), "7");
    assert_eq!(run("f(.Jump(h: 3))"), "30");
    // a return, a struct field, a typed `let`, a list element, an option
    assert_eq!(run("f(back())"), "2");
    assert_eq!(run("f(Move { d: .Left(5), n: 1 }.d)"), "5");
    assert_eq!(run("{ let d: Dir = .Down; f(d) }"), "2");
    assert_eq!(run("{ let ds: [Dir] = [.Up, .Down]; f(ds[1]) }"), "2");
    assert_eq!(run("opt(.Left(9))"), "9");
    // beside `==` against a typed side, either way round
    assert_eq!(run("{ let d = back(); d == .Down }"), "true");
    assert_eq!(run("{ let d = back(); .Up == d }"), "false");
    // two enums share a variant name: the slot picks
    assert_eq!(run("{ let a: Arrow = .Up; a == Arrow::Up }"), "true");
    // the full name still works, and means the same value
    assert_eq!(run("{ let d: Dir = .Up; d == Dir::Up }"), "true");
}

// r[verify mimas.enum.dot.no-slot]
#[test]
fn a_dot_variant_with_no_enum_expected_names_the_enums() {
    rejected(&format!("{DIR}\nlet d = .Up;"), "`Arrow`, `Dir` have `Up`");
    rejected(&format!("{DIR}\nlet d = .Left(3);"), "write `Dir::Left`");
    rejected(&format!("{DIR}\nlet d = .Sideways;"), "no declared enum has this variant");
    // a name the expected enum lacks lists its variants
    rejected(&format!("{DIR}\nf(.Sideways);"), "`.Sideways` is not a variant of `Dir`");
    // a match over a non-enum cannot take a dot arm
    rejected(&format!("{DIR}\nmatch 3 {{ .Up => 1, _ => 0 }};"), "needs a place that expects an enum");
}

// r[verify mimas.enum.dot.line]
#[test]
fn a_dot_variant_opening_a_line_is_a_new_value() {
    // the tail `.Down` is the block's value, not a field of the line above
    assert_eq!(run("f({ let n = 1\n.Down })"), "2");
    // a lowercase name after the dot still continues the line above
    assert_eq!(run("{ let s = \"abc\"\n    .len()\n  s }"), "3");
}

// a pattern inside a payload reads its own slot's enum
// r[verify mimas.enum.dot]
#[test]
fn a_dot_variant_nests_in_a_payload_pattern() {
    let src = format!("{DIR}\nenum Wrap {{ One(Dir), Two(Dir, Arrow) }}
fn w(x: Wrap) -> int {{ match x {{ .One(.Up) => 1, .One(_) => 2, .Two(.Left(k), .Down) => k, .Two(_, _) => 0 }} }}");
    assert_eq!(test_runner::render(&src, "w(.One(.Up))"), "1");
    assert_eq!(test_runner::render(&src, "w(.One(.Down))"), "2");
    assert_eq!(test_runner::render(&src, "w(.Two(.Left(6), .Down))"), "6");
    assert_eq!(test_runner::render(&src, "w(.Two(.Up, .Up))"), "0");
}

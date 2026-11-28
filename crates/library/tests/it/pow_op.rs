//! `a ^ b`, the power operator, through the `Pow` pact.

use crate::test_runner;

fn run(src: &str) -> String {
    test_runner::render("let n = 3\nlet k = 2\nlet x = 2.0", src)
}

fn rejected(src: &str, needle: &str) {
    let err = test_runner::try_execute(src).expect_err(&format!("`{src}` should be rejected"));
    let text = format!("{err:?}");
    assert!(text.contains(needle), "`{src}`: expected `{needle}` in:\n{text}");
}

// r[verify mimas.op.pow]
#[test]
fn power_binds_tightest_and_groups_right() {
    assert_eq!(run("2^3^2"), "512");
    assert_eq!(run("n^k^2"), "81");
    assert_eq!(run("-n^2"), "-9");
    assert_eq!(run("-2^2"), "-4");
    assert_eq!(run("(-n)^2"), "9");
    assert_eq!(run("2 * n^2"), "18");
    assert_eq!(run("n^2 * 2"), "18");
    assert_eq!(run("n^2 + 1"), "10");
    // glued or spaced, the same operator
    assert_eq!(run("n ^ 2"), "9");
}

// r[verify mimas.op.pow.types]
#[test]
fn int_and_float_powers() {
    assert_eq!(run("n^2"), "9");
    assert_eq!(run("n^k"), "9");
    assert_eq!(run("x^2"), "4.0");
    assert_eq!(run("x^k"), "4.0");
    assert_eq!(run("n^0.5 > 1.7"), "true");
    assert_eq!(run("n^-1 * 3.0"), "1.0");
    assert_eq!(run("2^-1"), "0.5");
    // the method spelling stays
    assert_eq!(run("n.pow(2)"), "9");
    // an int result is an int
    assert_eq!(run("{ let m: int = n^2; m }"), "9");
}

// r[verify mimas.op.pow.units]
#[test]
fn a_quantity_squares_its_unit() {
    assert_eq!(test_runner::render("let w = 3.0m", "w ^ 2"), test_runner::render("", "9.0m^2"));
    assert_eq!(test_runner::render("let w = 3.0m", "w^2 + 1.0m^2"), test_runner::render("", "10.0m^2"));
    // a quantity literal keeps its own unit exponent: `2.0m^2` is two square metres
    assert_eq!(test_runner::render("", "2.0m^2 + 1.0m^2"), test_runner::render("", "3.0m^2"));
    rejected("let w = 3.0m\nlet e = 2.0\nw ^ e", "whole-number literal exponent");
}

// r[verify mimas.op.pow]
#[test]
fn a_user_type_implements_pow() {
    let src = "struct V { x: float }
impl Pow<int> for V {
    type Out = V;
    fn pow(self, k: int) -> V { V { x: self.x.pow(k * 1.0) } }
}
fn sq<T: Pow<int, Out = T>>(v: T) -> T { v ^ 2 }
let v = V { x: 3.0 }";
    assert_eq!(test_runner::render(src, "(v ^ 2).x"), "9.0");
    assert_eq!(test_runner::render(src, "sq(v).x"), "9.0");
    assert_eq!(test_runner::render(src, "sq(5)"), "25");
}

// r[verify mimas.ident.symbols]
#[test]
fn a_caret_is_never_part_of_a_name() {
    // `x^2` was one name; now it is `x ^ 2`
    rejected("let x^2 = 4", "");
    assert_eq!(test_runner::render("let a = 5", "a^2"), "25");
}

//! std's operator pacts: `Add`, `Sub`, `Mul`, `Div`, `Rem` and `Neg`, each
//! with `type Out`, and `Rhs = Self` on the binary ones.

use crate::test_runner;

const V: &str = "struct V { x: float, y: float }
impl Add for V { type Out = V; fn add(self, o: V) -> V { V { x: self.x + o.x, y: self.y + o.y } } }
impl Sub for V { type Out = V; fn sub(self, o: V) -> V { V { x: self.x - o.x, y: self.y - o.y } } }
impl Mul for V { type Out = V; fn mul(self, o: V) -> V { V { x: self.x * o.x, y: self.y * o.y } } }
impl Mul<float> for V { type Out = V; fn mul(self, k: float) -> V { V { x: self.x * k, y: self.y * k } } }
impl Mul<V> for float { type Out = V; fn mul(self, v: V) -> V { V { x: self * v.x, y: self * v.y } } }
impl Div<float> for V { type Out = V; fn div(self, k: float) -> V { V { x: self.x / k, y: self.y / k } } }
impl Rem<float> for V { type Out = V; fn rem(self, k: float) -> V { V { x: self.x mod k, y: self.y mod k } } }
impl Neg for V { type Out = V; fn neg(self) -> V { V { x: -self.x, y: -self.y } } }
fn sum<T: Add<Out = T>>(xs: [T]) -> T {
    let acc = xs[0]
    for i in 1..xs.len() { acc = acc + xs[i] }
    acc
}
fn scale<T: Mul<float, Out = T>>(x: T) -> T { x * 2.0 }
fn flip<T: Neg<Out = T>>(x: T) -> T { -x }
let a = V { x: 1.0, y: 2.0 }
let b = V { x: 3.0, y: 5.0 }";

fn run(src: &str) -> String {
    test_runner::render(V, src)
}

fn rejected(src: &str, needle: &str) {
    let err = test_runner::try_execute(src).expect_err(&format!("`{src}` should be rejected"));
    let text = format!("{err:?}");
    assert!(text.contains(needle), "`{src}`: expected `{needle}` in:\n{text}");
}

// r[verify mimas.op.pact]
#[test]
fn an_operator_resolves_through_the_rhs_type() {
    assert_eq!(run("(a + b).y"), "7.0");
    assert_eq!(run("(b - a).x"), "2.0");
    assert_eq!(run("(a * b).y"), "10.0");
    assert_eq!(run("(a * 3.0).x"), "3.0");
    assert_eq!(run("(2.0 * b).y"), "10.0");
    assert_eq!(run("(b / 2.0).x"), "1.5");
    assert_eq!(run("(b mod 2.0).y"), "1.0");
    assert_eq!(run("(-a).y"), "-2.0");
    // `+=` takes the same road
    assert_eq!(run("{ let c = a; c += b; c.x }"), "4.0");
}

// r[verify mimas.std.ops]
// r[verify mimas.op.pact-std]
#[test]
fn a_bound_takes_the_numbers_and_a_user_type() {
    assert_eq!(run("sum([1, 2, 3])"), "6");
    assert_eq!(run("sum([0.5, 0.25])"), "0.75");
    assert_eq!(run("sum([a, b, a]).x"), "5.0");
    assert_eq!(run("scale(1.5)"), "3.0");
    assert_eq!(run("scale(a).y"), "4.0");
    assert_eq!(run("flip(3)"), "-3");
    assert_eq!(run("flip(a).x"), "-1.0");
    // a quantity is a float with a dimension, so it meets the bound
    assert_eq!(run("{ let m: float = sum([1.0, 2.0]); m }"), "3.0");
}

// r[verify mimas.std.ops]
#[test]
fn a_bound_is_held_to_the_operator() {
    // `int` implements `Mul<Self>`, not `Mul<float>`
    rejected(&format!("{V}\nscale(3)"), "Mul<float>");
    // `str` has `+` built in but no `Add` impl
    rejected(&format!("{V}\nsum([\"a\"])"), "Add");
    // `int / int` is a `float`, so `Div<Out = int>` does not hold
    rejected(
        "fn half<T: Div<Out = T>>(x: T) -> T { x / x }\nhalf(4)",
        "Div",
    );
}

// r[verify mimas.op.method]
#[test]
fn an_inherent_operator_method_still_works() {
    let src = "struct M { n: int }
impl M { fn add(self, o: M) -> M { M { n: self.n + o.n } } }
let m = M { n: 2 }";
    assert_eq!(test_runner::render(src, "(m + m).n"), "4");
}

// r[verify mimas.pact.params.pick]
#[test]
fn an_open_operand_takes_the_left_type() {
    // an operator's open operand takes the other side's type, as it always
    // did, so `a * o` is `Mul<V>`; a method call has no such rule and is
    // the ambiguity error
    assert_eq!(test_runner::render(&format!("{V}\nfn f(o) {{ a * o }}"), "f(b).y"), "10.0");
    rejected(&format!("{V}\nfn g(o) {{ a.mul(o) }}"), "could be any of 2 impls");
}

//! Pacts with type parameters, `pact Plus<Rhs = Self> { type Out; … }`:
//! one pact implemented several times with different arguments, bounds
//! that name the arguments and pin associated types.

use crate::test_runner;

const V: &str = "pact Plus<Rhs = Self> {
    type Out;
    fn plus(self, o: Rhs) -> Out;
}
struct V { x: float, y: float }
impl Plus for V {
    type Out = V;
    fn plus(self, o: V) -> V { V { x: self.x + o.x, y: self.y + o.y } }
}
impl Plus<float> for V {
    type Out = V;
    fn plus(self, k: float) -> V { V { x: self.x + k, y: self.y + k } }
}
impl Plus<V> for float {
    type Out = V;
    fn plus(self, v: V) -> V { V { x: self + v.x, y: self + v.y } }
}
impl Plus for int {
    type Out = int;
    fn plus(self, o: int) -> int { self + o }
}
fn total<T: Plus<Out = T>>(xs: [T]) -> T {
    let acc = xs[0]
    for i in 1..xs.len() { acc = acc.plus(xs[i]) }
    acc
}
fn bump<T: Plus<float, Out = T>>(v: T) -> T { v.plus(0.5) }
let a = V { x: 1.0, y: 2.0 }
let b = V { x: 10.0, y: 20.0 }";

fn rejected(src: &str, needle: &str) {
    let err = test_runner::try_execute(src).expect_err(&format!("`{src}` should be rejected"));
    let text = format!("{err:?}");
    assert!(text.contains(needle), "`{src}`: expected `{needle}` in:\n{text}");
}

fn run(src: &str) -> String {
    test_runner::render(V, src)
}

// r[verify mimas.pact.params]
// r[verify mimas.pact.params.impls]
#[test]
fn one_pact_takes_several_impls_by_argument() {
    assert_eq!(run("a.plus(b).x"), "11.0");
    assert_eq!(run("a.plus(0.5).y"), "2.5");
    assert_eq!(run("2.0.plus(a).x"), "3.0");
    assert_eq!(run("3.plus(4)"), "7");
}

// r[verify mimas.pact.params.default]
// r[verify mimas.pact.assoc-bound]
#[test]
fn a_bound_takes_defaults_and_pins_an_associated_type() {
    // `T: Plus<Out = T>` is `Plus<T>`, and its `Out` reads as `T`
    assert_eq!(run("total([1, 2, 3])"), "6");
    assert_eq!(run("total([a, b, a]).y"), "24.0");
    // `T: Plus<float, Out = T>` picks the `float` impl
    assert_eq!(run("bump(a).x"), "1.5");
    // two instances of one pact on one param: the argument picks, at
    // compile time and at run time
    let both = "fn twice<T: Plus<Out = T> + Plus<float, Out = T>>(v: T) -> T { v.plus(v).plus(1.0) }";
    assert_eq!(test_runner::render(&format!("{V}\n{both}"), "twice(a).y"), "5.0");
}

// r[verify mimas.pact.params.bound]
#[test]
fn a_bound_is_held_to_its_arguments_and_pins() {
    // `str` has no `Plus` impl at all
    rejected(&format!("{V}\ntotal([\"a\"])"), "Plus");
    // `int` implements `Plus<Self>`, not `Plus<float>`
    rejected(&format!("{V}\nbump(3)"), "Plus<float>");
    // the impl's `Out` is not the pinned type
    rejected(
        "pact Plus<Rhs = Self> { type Out; fn plus(self, o: Rhs) -> Out; }
struct W { n: int }
impl Plus for W { type Out = int; fn plus(self, o: W) -> int { self.n + o.n } }
fn total<T: Plus<Out = T>>(xs: [T]) -> T { xs[0].plus(xs[1]) }
total([W { n: 1 }, W { n: 2 }])",
        "Plus",
    );
}

// r[verify mimas.pact.params.pick]
#[test]
fn a_call_picks_the_impl_by_argument_type() {
    // an open argument type picks nothing: the error lists the impls
    rejected(
        &format!("{V}\nfn f(o) {{ a.plus(o) }}"),
        "could be any of 2 impls",
    );
}

// r[verify mimas.pact.params]
#[test]
fn pact_arguments_must_fit_the_parameters() {
    rejected(
        "pact P { fn f(self) -> int; }\nstruct S {}\nimpl P<int> for S { fn f(self) -> int { 1 } }",
        "takes no type arguments",
    );
    rejected(
        "pact P<A> { fn f(self, a: A) -> int; }\nstruct S {}\nimpl P for S { fn f(self, a: int) -> int { 1 } }",
        "needs an argument for `A`",
    );
    rejected(
        "pact P<A> { fn f(self, a: A) -> int; }\nstruct S {}\nimpl P<int, int> for S { fn f(self, a: int) -> int { 1 } }",
        "takes 1 type argument",
    );
    // a second impl of the same instance is still a duplicate
    rejected(
        "pact P<A> { fn f(self, a: A) -> int; }\nstruct S {}\nimpl P<int> for S { fn f(self, a: int) -> int { 1 } }\nimpl P<int> for S { fn f(self, a: int) -> int { 2 } }",
        "P<int>",
    );
}

// r[verify mimas.pact.params.generic-target]
#[test]
fn a_generic_target_names_its_params_in_the_arguments() {
    let src = "pact Plus<Rhs = Self> { type Out; fn plus(self, o: Rhs) -> Out; }
struct Box<T> { v: T }
impl Plus<T> for Box<T> {
    type Out = Box<T>;
    fn plus(self, o: T) -> Box<T> { Box { v: o } }
}
fn put<B: Plus<int>>(b: B) -> B::Out { b.plus(7) }
let b = Box { v: 1 }";
    assert_eq!(test_runner::render(src, "b.plus(5).v"), "5");
    // through the bound: the checker and the run both find the impl
    assert_eq!(test_runner::render(src, "put(b).v"), "7");
    rejected(
        "pact Plus<Rhs = Self> { fn plus(self, o: Rhs) -> int; }
struct Box<T> { v: T }
impl Plus<U> for Box<U> { fn plus(self, o: U) -> int { 1 } }",
        "declares <T>",
    );
}

// r[verify mimas.pact.params]
#[test]
fn a_member_type_parameter_and_an_associated_type_work_with_pact_params() {
    let src = "pact Conv<To> {
    type Tag;
    fn conv(self) -> To;
    fn conv_with<U>(self, f: (To) -> U) -> U { f(self.conv()) }
}
struct C { n: int }
impl Conv<str> for C { type Tag = int; fn conv(self) -> str { f\"{self.n}\" } }
impl Conv<float> for C { type Tag = int; fn conv(self) -> float { self.n * 1.0 } }
fn as_str<T: Conv<str>>(x: T) -> str { x.conv() }
let c = C { n: 4 }";
    assert_eq!(test_runner::render(src, "as_str(c)"), "\"4\"");
}

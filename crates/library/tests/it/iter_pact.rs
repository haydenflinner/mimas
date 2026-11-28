//! `pact Iter { type Item; fn iter(self) -> () -> Item?; }`: the protocol
//! `for` and the combinators walk, with an associated type each impl fixes.

use crate::test_runner;

const DECK: &str = "struct Deck { cards: [str] }
impl Iter for Deck {
    type Item = str;
    fn iter(self) -> () -> Item? { __list_pull(self.cards) }
}
let d = Deck { cards: [\"ace\", \"two\", \"three\"] }
fn count<T: Iter>(xs: T) -> int {
    let n = 0
    for _x in xs { n += 1 }
    n
}";

fn rejected(src: &str, needle: &str) {
    let err = test_runner::try_execute(src).expect_err(&format!("`{src}` should be rejected"));
    let text = format!("{err:?}");
    assert!(text.contains(needle), "`{src}`: expected `{needle}` in:\n{text}");
}

// r[verify mimas.std.iter]
// r[verify mimas.protocol.iter]
#[test]
fn a_user_type_walks_through_iter() {
    let run = |src: &str| test_runner::render(DECK, src);
    // `for` reads `Item` as the impl's `str`
    assert_eq!(run("{ let t = 0; for c in d { t += c.len() }; t }"), "11");
    // a `T: Iter` bound takes it, and std's own impls
    assert_eq!(run("count(d)"), "3");
    assert_eq!(run("count(2..6)"), "4");
    assert_eq!(run("count(Stream { items: [1, 2] })"), "2");
    // the combinators walk it; a closure reads `str` methods on each item
    assert_eq!(run("d.map(|c| c.len()).to_list()"), "[3, 3, 5]");
    assert_eq!(run("d.filter(|c| c.len() == 3).to_list()"), "[\"ace\", \"two\"]");
}

// r[verify mimas.pact.assoc-type]
#[test]
fn an_impl_fixes_every_associated_type() {
    rejected(
        "struct S {}\nimpl Iter for S {\n    fn iter(self) -> () -> int? { || -> int? { null } }\n}",
        "type Item",
    );
    rejected(
        "struct S {}\nimpl Iter for S {\n    type Item = int;\n    type Elem = int;\n    fn iter(self) -> () -> int? { || -> int? { null } }\n}",
        "Elem",
    );
    // the signature must agree with the binding
    rejected(
        "struct S {}\nimpl Iter for S {\n    type Item = int;\n    fn iter(self) -> () -> str? { || -> str? { null } }\n}",
        "str",
    );
}

// r[verify mimas.pact.assoc-type]
#[test]
fn a_bounded_params_item_is_opaque() {
    // a generic body cannot assume the item is an `int`
    rejected(
        "fn total<T: Iter>(xs: T) -> int {\n    let t = 0\n    for x in xs { t += x }\n    t\n}",
        "T::Item",
    );
    // and a type without `Iter` misses the bound
    rejected(&format!("{DECK}\nlet n = count(5);"), "Iter");
}

// r[verify mimas.pact.assoc-type]
#[test]
fn a_default_body_names_the_associated_type() {
    let pre = "pact Source {
        type Item;
        fn first(self) -> Item;
        fn pair(self) -> (Item, Item) { (self.first(), self.first()) }
    }
    struct Dice { faces: [int] }
    impl Source for Dice {
        type Item = int;
        fn first(self) -> Item { let f: Item = self.faces[0]; f }
    }";
    // through a concrete receiver, `Item` is the impl's `int`
    assert_eq!(test_runner::render(pre, "Dice { faces: [6, 5] }.pair().0 + 1"), "7");
}

// r[verify mimas.std.iter.builtin]
// r[verify mimas.impl.builtin-shape]
#[test]
fn the_built_in_shapes_meet_the_bound() {
    let run = |src: &str| test_runner::render(DECK, src);
    assert_eq!(run("count([1, 2, 3])"), "3");
    // a `str` walks by character, a dict by `(key, value)` pair
    assert_eq!(run("count(\"héllo\")"), "5");
    assert_eq!(run("count({\"a\": 1, \"b\": 2})"), "2");
    // the impl's `iter` is a method of its own on the shape
    assert_eq!(run("{ let p = [5, 6].iter(); p() }"), "5");
    // a list keeps its own `map`; a `str` walks through `iter`
    assert_eq!(run("[1, 2].map(|x| x + 1)"), "[2, 3]");
    assert_eq!(run("\"ab\".map(|c| c + \"!\").to_list()"), "[\"a!\", \"b!\"]");
}

const PATHS: &str = "fn first<T: Iter>(xs: T) -> T::Item? { let p = xs.iter(); p() }
fn last<T: Iter>(xs: T) -> T::Item? {
    let out: T::Item? = null
    for x in xs { out = x }
    out
}
fn has<T: Iter>(xs: T, want: T::Item) -> bool {
    for x in xs { if x == want { return true } }
    false
}";

// r[verify mimas.pact.assoc-path]
#[test]
fn an_associated_type_path_reads_in_a_signature() {
    let run = |src: &str| test_runner::render(PATHS, src);
    assert_eq!(run("first([3, 4])"), "3");
    assert_eq!(run("first(\"xy\")"), "\"x\"");
    // the caller reads the item's own type: `str` methods on a list's item
    assert_eq!(run("first([\"ab\", \"c\"])!.len()"), "2");
    assert_eq!(run("last(2..7)"), "6");
    assert_eq!(run("has([1, 2], 2)"), "true");
    assert_eq!(run("has(\"abc\", \"z\")"), "false");
    // a parameter typed `T::Item` checks against the list's item
    rejected(&format!("{PATHS}\nlet a = has([1, 2], \"x\");"), "int");
}

// r[verify mimas.pact.member-generic]
#[test]
fn a_pact_member_takes_its_own_type_parameters() {
    let pre = "pact Boxy {
        fn get(self) -> int;
        fn with<U>(self, u: U) -> (int, U) { (self.get(), u) }
    }
    struct B { n: int }
    impl Boxy for B { fn get(self) -> int { self.n } }
    let b = B { n: 4 }";
    // each call picks its own `U`
    assert_eq!(test_runner::render(pre, "b.with(\"s\").1.len() + b.with(2).1"), "3");
    // in the default body `U` is rigid: it is not an `int`
    rejected(
        "pact P {\n    fn get(self) -> int;\n    fn bad<U>(self, u: U) -> int { u + 1 }\n}",
        "U",
    );
    // a member's own bound is owed at the call
    rejected(
        "struct S { cards: [str] }\nimpl Iter for S {\n    type Item = str;\n    fn iter(self) -> () -> str? { __list_pull(self.cards) }\n}\nlet z = S { cards: [] }.zip(5);",
        "Iter",
    );
}

// r[verify mimas.iter.pact-combinators]
#[test]
fn the_combinators_are_iter_methods() {
    let pre = format!(
        "{DECK}
struct Echo {{ xs: [int] }}
impl Iter for Echo {{
    type Item = int;
    fn iter(self) -> () -> int? {{ __list_pull(self.xs) }}
    fn map<U>(self, f: (int) -> U) -> () -> U? {{ __pull_map(__list_pull([0]), f) }}
}}
fn sevens<T: Iter>(xs: T) -> [int] {{ xs.map(|_x| 7).to_list() }}"
    );
    let run = |src: &str| test_runner::render(&pre, src);
    assert_eq!(run("d.fold(0, |a, c| a + c.len())"), "11");
    // `zip`'s other side is any `Iter`, each call its own
    assert_eq!(run("d.zip([1, 2]).to_list().len()"), "2");
    assert_eq!(run("d.zip([1, 2]).to_list()[1].0.len() + d.zip([1, 2]).to_list()[1].1"), "5");
    assert_eq!(run("d.zip(0..10).to_list().len()"), "3");
    assert_eq!(run("d.enumerate().to_list()[2].0"), "2");
    // an impl's own member answers, directly and through a bound
    assert_eq!(run("Echo { xs: [1, 2] }.map(|x| x + 1).to_list()"), "[1]");
    assert_eq!(run("sevens(Echo { xs: [1, 2] })"), "[7]");
    // a list keeps its own `map`, which answers a list; through a bound it walks
    assert_eq!(run("[1, 2].map(|x| x * 10)"), "[10, 20]");
    assert_eq!(run("sevens([1, 2, 3])"), "[7, 7, 7]");
    assert_eq!(run("\"abc\".to_list()"), "[\"a\", \"b\", \"c\"]");
}

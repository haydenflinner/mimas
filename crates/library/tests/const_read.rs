//! The value of a `const` is one value, read-only all the way down.
//!
//! A list, a struct or a record `const` is built once when the program starts,
//! frozen (`Ctx::freeze`), and every use reads that one value
//! (`compile::ir::Ir::store_const`). A read of a plain field is folded to the
//! field's own value (`Constant::field_of`, `column`). A write through any
//! value that came from a const is an error. `deep_clone` gives a copy that
//! can change.
#[macro_use]
mod test_runner;

const TABLE: &str = r#"
struct P { x: float, y: float }
struct B { a: int, name: str, on: bool, at: P, tags: [str] }
const O = P { x = 1, y = -2.5 };
const ONE = B { a = 1, name = "one", on = true, at = O, tags = ["t"] };
const MENU = [
    B { a = 1, name = "one", on = true, at = O, tags = [] },
    B { a = 2, name = "two", on = false, at = P { x = 3, y = 4 }, tags = [] },
    ONE,
];
const XS = [10, 20, 30];
fn total() -> int { let n = 0; for i in 0 .. 3 { n += MENU[i].a; } n }
fn whole() -> int { let n = 0; for i in 0 .. 3 { let b = MENU[i]; n += b.a; } n }
fn names() -> str { let s = ""; for i in 0 .. 3 { if MENU[i].on { s = s + MENU[i].name; } } s }
fn bump(b: B) { b.a += 1; }
let CALLS = [0];
fn noted() -> int { CALLS[0] += 1; 5 }
// a struct written out in ordinary code is not a const: every field runs
fn literal_field() -> int { let b = B { a = noted(), name = "n", on = true, at = O, tags = [] }.name; CALLS[0] }
// a copy can change, and the constant does not
fn copied() -> str {
    let mine = deep_clone(ONE);
    mine.a = 7;
    mine.at.x = 9.0;
    mine.tags.push("u");
    f"{mine.a} {mine.at.x} {mine.tags.len()} / {ONE.a} {ONE.at.x} {ONE.tags.len()}"
}
"#;

test_run!(
    // r[verify mimas.const.read]
    // r[verify mimas.const.shared]
    a_const_reads_as_its_value,
    TABLE,
    "O.x" => "1.0",
    "O.y" => "-2.5",
    "ONE.name" => "\"one\"",
    "(ONE).a" => "1",
    // a field that is a const, and a field of that
    "ONE.at.x + ONE.at.y" => "-1.5",
    // one field of each element, by an index that runs
    "total()" => "4",
    "whole()" => "4",
    "names()" => "\"oneone\"",
    "MENU[1].at.y" => "4.0",
    "MENU[2].name" => "\"one\"",
    "XS.len()" => "3",
    // one value: every use is the same thing
    "MENU[2] == ONE" => "true",
    "literal_field()" => "1"
);

test_run!(
    // r[verify mimas.const.frozen]
    a_copy_of_a_const_can_change,
    TABLE,
    "copied()" => "\"7 9.0 2 / 1 1.0 1\""
);

// A write through a value that came from a const is an error, however deep
// and however it got there.
test_fail!(
    // r[verify mimas.const.frozen]
    a_const_cannot_be_changed,
    &format!("{TABLE}\nlet x = ONE; x.a = 5;"),
    &format!("{TABLE}\nlet p = ONE.at; p.x = 5.0;"),
    &format!("{TABLE}\nlet t = MENU[2].tags; t.push(\"z\");"),
    &format!("{TABLE}\nbump(MENU[1]);"),
    &format!("{TABLE}\nlet xs = XS; xs.push(1);"),
    &format!("{TABLE}\nlet xs = XS; xs[0] = 1;"),
    &format!("{TABLE}\nfor b in MENU {{ b.at.y = 0.0; }}"),
    &format!("{TABLE}\nfn f() {{ const LOCAL = [1, 2]; let l = LOCAL; l.push(3); }} f();")
);

//! Units through closures, generics and std combinators (combo item C8).
//! Each probe from the combo doc must refuse its `+ 1.0s` with the usual
//! dimension message, and the same programs with matching units compile.

fn refused(src: &str) -> String {
    match vm::Vm::compile(src, library::std) {
        Ok(_) => panic!("compiled: {src}"),
        Err(e) => format!("{e:?}"),
    }
}

fn compiles(src: &str) {
    if let Err(e) = vm::Vm::compile(src, library::std) {
        panic!("refused: {src}\n{e:?}");
    }
}

const ADD_PX_S: &str = "cannot add pixels (px) and time (s)";

// leak07, d12: a closure with a declared parameter unit returns its
// declared (or computed) unit at the call.
// r[verify mimas.unit.closure]
#[test]
fn a_typed_closure_keeps_its_return_unit() {
    assert!(refused("let f = |w: px| -> px { w }\nlet bad = f(1.0px) + 1.0s\n").contains(ADD_PX_S));
    assert!(refused("let pad = 4.0px\nlet grow = |w: px| w + pad\nlet bad = grow(10.0px) + 1.0s\n")
        .contains(ADD_PX_S));
    let err = refused("let f = |w: px| -> px { w }\nlet x = f(1.0s)\n");
    assert!(err.contains("argument 1 of `f`"), "{err}");
    compiles("let f = |w: px| w + 1.0px\nlet ok = f(2.0px) + 1.0px\n");
}

// leak08: a closure passed where `(px) -> px` is declared learns `px`
// for its parameter, so a body that adds seconds is refused.
// r[verify mimas.unit.closure]
#[test]
fn a_closure_argument_learns_its_parameter_unit() {
    let src = "fn ap(f: (px) -> px, x: px) -> px { f(x) }\nlet bad = ap(|w| w + 1.0s, 1.0px)\n";
    assert!(refused(src).contains(ADD_PX_S));
    compiles("fn ap(f: (px) -> px, x: px) -> px { f(x) }\nlet ok = ap(|w| w * 2.0, 1.0px) + 1.0px\n");
    // a fn-typed parameter called inside the body gives its declared unit
    let src = "fn ap(f: (px) -> px, x: px) -> px { f(x) + 1.0s }\n";
    assert!(refused(src).contains(ADD_PX_S));
    // a returned closure called at once
    compiles(
        "fn twice(f: (px) -> px) -> (px) -> px { |x| f(f(x)) }\n\
         let ok = twice(|w| w + 1.0px)(1.0px) + 1.0px\n",
    );
}

// d14, d11: a type parameter takes the unit of the first argument it
// meets; a second argument of another unit is refused, and the return
// keeps the unit.
// r[verify mimas.unit.generic]
#[test]
fn a_type_parameter_carries_one_unit() {
    let big = "fn biggest<T>(a: T, b: T) -> T { if a > b { a } else { b } }\n";
    let err = refused(&format!("{big}let x = biggest(3.0px, 5.0s)\n"));
    assert!(err.contains("argument `b` of `biggest`"), "{err}");
    assert!(refused(&format!("{big}let z = biggest(3.0px, 5.0px)\nlet bad = z + 1.0s\n")).contains(ADD_PX_S));
    compiles(&format!(
        "{big}fn first<T>(xs: [T]) -> T {{ xs[0] }}\n\
         fn pair<A, B>(a: A, b: B) -> (A, B) {{ (a, b) }}\n\
         let a = biggest(3.0px, 5.0px) + 1.0px\nlet b = first([1.0s, 2.0s]) + 1.0s\n\
         let p = pair(1.0px, 2.0s)\nlet c = p.1 + 1.0s\nlet d = biggest(1, 2)\nlet e = biggest(\"a\", \"b\")\n"
    ));
}

// leak03, leak10: `map` gives a list of what its closure returns, and
// `fold`'s closure learns the accumulator's and the items' units.
// r[verify mimas.unit.combinator]
#[test]
fn combinators_keep_units() {
    let xs = "let xs: [px] = [2.0px, 3.0px]\n";
    assert!(refused(&format!("{xs}let bad = xs.map(|w| w * 2.0)[0] + 1.0s\n")).contains(ADD_PX_S));
    let err = refused("let bad = [1.0px, 2.0px].fold(0.0s, |a, w| a + w)\n");
    assert!(err.contains("cannot add time (s) and pixels (px)"), "{err}");
    compiles(&format!(
        "{xs}let a = xs.map(|w| w * 2.0)[0] + 1.0px\n\
         let b = xs.fold(0.0px, |a, w| a + w) + 1.0px\n\
         let c = xs.filter(|w| w > 2.5px)[0] + 1.0px\n"
    ));
}

// x08, u06: `⨅`/`⨆` keep the items' unit; `÷` gives a quotient worth
// `a / b` and a remainder worth `a`.
// r[verify mimas.unit.combinator]
#[test]
fn glyph_reductions_keep_units() {
    let src = "let ws: [px] = [3.0px, 1.0px]\nlet m = (⨅ ws)!\nlet bad = m + 1.0s\n";
    assert!(refused(src).contains(ADD_PX_S));
    let src = "let (q, r) = 7.0px ÷ 2.0px\nlet bad = r + 1.0s\n";
    assert!(refused(src).contains(ADD_PX_S));
    compiles("let (q, r) = 7.0px ÷ 2.0px\nlet ok = r + 1.0px\nlet n = q + 1.0\n");
}

// leak05: a dict whose values share a unit keeps it through `d[k]`, as
// a `{px}` annotation does; a `for` over it walks `(key, value)` pairs.
// r[verify mimas.unit.dict]
#[test]
fn dict_values_keep_their_unit() {
    assert!(refused("let d = {\"a\": 2.0px}\nlet bad = d[\"a\"]! + 1.0s\n").contains(ADD_PX_S));
    assert!(refused("let d: {px} = {:}\nd[\"a\"] = 1.0px\nlet bad = d[\"a\"]! + 1.0s\n").contains(ADD_PX_S));
    compiles("let d = {\"a\": 2.0px}\nlet ok = d[\"a\"]! + 1.0px\nfor (k, v) in d { let w = v + 1.0px }\n");
}

// x05c: `for` over a generator declared `-> px` gives `px` items.
// r[verify mimas.unit.generator]
#[test]
fn generator_items_keep_their_unit() {
    let src = "subscript ws() -> px { yield 1.0px; yield 2.0px }\nfor w in ws() { let bad = w + 1.0s }\n";
    assert!(refused(src).contains(ADD_PX_S));
    compiles("subscript ws() -> px { yield 1.0px; yield 2.0px }\nfor w in ws() { let ok = w + 1.0px }\n");
}

// x05g: a member generator read as a field (`b.wide`) gives its
// declared item unit in `for`.
// r[verify mimas.unit.generator]
#[test]
fn a_member_generator_keeps_its_unit() {
    let bag = "struct Bag { ws: [px] }\n\
               impl Bag { subscript wide() -> px { for w in self.ws { if w > 1.0px { yield w } } } }\n\
               let b = Bag { ws: [1.0px, 2.0px] }\n";
    assert!(refused(&format!("{bag}for w in b.wide {{ let bad = w + 1.0s }}\n")).contains(ADD_PX_S));
    compiles(&format!("{bag}for w in b.wide {{ let ok = w + 1.0px }}\n"));
}

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

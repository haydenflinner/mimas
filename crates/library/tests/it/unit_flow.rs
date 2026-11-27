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

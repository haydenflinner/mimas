//! Runs each fixture twice — once on the plain interpreter, once with the
//! bcgen-specialized body table installed — and diffs `TEST_VALUE`, the `keep`
//! log, and the error outcome.

use bcgen_test::{generated, natives};
use vm::{Captured, Vm};

fn run(
    source: &str,
    bodies: Option<Vec<Option<vm::bc::BodyFn>>>,
) -> (Option<Captured>, Vec<Captured>, Option<String>) {
    let (program, sources) =
        Vm::compile_parts(&[("main", source)], natives::install).expect("fixture compiles");
    let mut vm = Vm::new();
    vm.load_prebuilt(program, sources, natives::install);
    if let Some(bodies) = bodies {
        vm.install_bc(bodies);
    }
    let err = vm.run().err().map(|e| e.to_string());
    let test_value = vm.resolve_name("TEST_VALUE");
    let kept = vm.fixture::<natives::Kept>().0.borrow().clone();
    (test_value, kept, err)
}

#[test]
fn cold_parity() {
    let source = include_str!("../fixtures/cold.mimas");
    assert_eq!(
        run(source, None),
        run(source, Some(generated::cold::bodies()))
    );
}

/// The emitter covers the entire op set — the only `step` call site left in
/// generated source should be the one `_ =>` catch-all per body fn. If this
/// fails, either a new op landed (emit an arm for it) or an arm silently fell
/// back to delegation.
#[test]
fn every_op_specialized() {
    for src in [
        include_str!(concat!(env!("OUT_DIR"), "/basic.rs")),
        include_str!(concat!(env!("OUT_DIR"), "/cold.rs")),
        include_str!(concat!(env!("OUT_DIR"), "/deep.rs")),
        include_str!(concat!(env!("OUT_DIR"), "/mixed.rs")),
    ] {
        // the inner-body signature is `-> u8` (the gout/tag call convention);
        // the outlined cold exit helper `gbail` shares it — don't count it
        let bodies =
            src.matches(") -> u8 {").count() - src.matches("fn gbail").count();
        // shadow-init bails also say `=> return step(`; the catch-all is the
        // only `_ =>` arm (it sets `*op_ip` first, then flushes shadows)
        let delegates = src.matches("_ => { *io.op_ip = code.ip;").count();
        assert_eq!(bodies, delegates, "an op arm delegated to the interpreter");
    }
}

#[test]
fn basic_parity() {
    let source = include_str!("../fixtures/basic.mimas");
    let bodies = generated::basic::bodies();
    assert!(!bodies.is_empty());
    assert_eq!(run(source, None), run(source, Some(bodies)));
}

/// Recursion past `INLINE_CALL_DEPTH` — the body fast-path hands off to
/// `Flow::Call` at the cap; both lanes must still land the same answer.
#[test]
fn deep_parity() {
    let source = include_str!("../fixtures/deep.mimas");
    let (v, k, e) = run(source, Some(generated::deep::bodies()));
    assert!(e.is_none(), "specialized deep run errored: {e:?}");
    assert!(matches!(v, Some(Captured::Int(400))));
    assert_eq!(run(source, None), (v, k, e));
}

#[test]
fn mixed_parity() {
    let source = include_str!("../fixtures/mixed.mimas");
    assert_eq!(
        run(source, None),
        run(source, Some(generated::mixed::bodies()))
    );
}

/// Sanity that the specialized path is actually being exercised, not silently
/// skipped — a bc-run fixture must produce the right answer, not just *an*
/// answer matching the interpreter's (which would also pass if `install_bc`
/// were a no-op and both runs ran plain).
#[test]
fn specialized_run_produces_values() {
    let source = include_str!("../fixtures/basic.mimas");
    let (test_value, kept, err) = run(source, Some(generated::basic::bodies()));
    assert!(err.is_none(), "specialized run errored: {err:?}");
    assert!(matches!(test_value, Some(Captured::Int(208))));
    assert_eq!(kept.len(), 2);
}

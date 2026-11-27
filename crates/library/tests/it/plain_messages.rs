//! Papercuts the lang2 lane fixed after its items: each mistake gets a
//! message that names the fix.

fn refused(src: &str) -> String {
    match vm::Vm::compile(src, library::std) {
        Ok(_) => panic!("compiled: {src}"),
        Err(e) => format!("{e:?}"),
    }
}

// `≈` checks dimensions as `==` does.
// r[verify mimas.unit.dimension]
#[test]
fn approx_compares_one_dimension() {
    let err = refused("check 1.0px ≈ 1.0s\n");
    assert!(err.contains("cannot compare pixels (px) and time (s)"), "{err}");
    let mut vm = vm::Vm::compile("check 1.0px ≈ 1.0px\ncheck 0.1 + 0.2 ≈ 0.3\n", library::std).unwrap();
    assert!(vm.run_tests().iter().all(|r| r.passed()));
}

// `B { ..b, up = false }` says the spread comes last.
// r[verify mimas.struct.update]
#[test]
fn a_spread_before_fields_says_it_comes_last() {
    let err = refused("struct B { up: bool, n: int }\nlet b = B { up = true, n = 1 }\nlet c = B { ..b, up = false }\n");
    assert!(err.contains("`..base` comes last"), "{err}");
    assert!(err.contains("B { up = false, ..b }"), "{err}");
}

// `1 < 2 ∧ 2 < 3`: `∧` is a named operator; the error says `&&`.
// r[verify mimas.op.named]
#[test]
fn a_logic_glyph_points_at_the_keyboard_form() {
    let err = refused("let ok = 1 < 2 ∧ 2 < 3\n");
    assert!(err.contains("not logic") && err.contains("`&&`"), "{err}");
    let err = refused("let ok = true ∨ false\n");
    assert!(err.contains("`||`"), "{err}");
}

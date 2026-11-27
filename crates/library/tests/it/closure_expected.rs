//! A closure learns its parameter types from the fn type its slot
//! expects: a struct field, a `let` annotation, a return value, as a
//! call argument already did (combo item C4: s08, s14, x12).

/// What each `print` recorded for the host (a string in quotes).
fn out(src: &str) -> Vec<String> {
    let vm = vm::Vm::execute(src, library::std).expect("runs");
    vm.fixture::<vm::fixtures::Prints>()
        .take()
        .into_iter()
        .map(|(_, _, text)| text)
        .collect()
}

fn refused(src: &str) -> String {
    match vm::Vm::compile(src, library::std) {
        Ok(_) => panic!("compiled: {src}"),
        Err(e) => format!("{e:?}"),
    }
}

// r[verify mimas.closure.expected]
#[test]
fn a_struct_field_closure_learns_its_parameter() {
    let src = "struct Btn { label: str, on: (Btn) -> str }\n\
               let b = Btn { label: \"ok\", on: |me| \"clicked \" + me.label }\n\
               print(b.on(b))\n";
    assert_eq!(out(src), ["\"clicked ok\""]);
}

// r[verify mimas.closure.expected]
#[test]
fn a_let_annotation_types_the_closure() {
    let src = "struct Btn { label: str }\n\
               let f: (Btn) -> str = |me| me.label\n\
               let h: (int, int) -> int = |a, b| a + b\n\
               print(f(Btn { label: \"x\" }))\nprint(h(2, 3))\n";
    assert_eq!(out(src), ["\"x\"", "5"]);
}

// r[verify mimas.closure.expected]
#[test]
fn a_returned_closure_and_a_branch_learn_from_the_slot() {
    let src = "fn mk() -> (int) -> int { |x| x * 2 }\n\
               print(mk()(4))\n\
               struct Btn { label: str, on: (Btn) -> str }\n\
               let b = Btn { label: \"b\", on: if true { |me| me.label } else { |me| \"\" } }\n\
               print(b.on(b))\n";
    assert_eq!(out(src), ["8", "\"b\""]);
}

// A closure annotated against its slot is still a mismatch.
// r[verify mimas.closure.expected]
#[test]
fn a_conflicting_annotation_still_fails() {
    let err = refused("let f: (int) -> int = |s: str| s.len()\n");
    assert!(err.contains("expected int but found str") || err.contains("mismatch"), "{err}");
}

// x12: `sort_by_int` takes the keys, not a key fn. The error says so
// instead of naming an unresolved internal type.
// r[verify mimas.closure.expected]
#[test]
fn a_closure_for_a_list_parameter_is_a_plain_mismatch() {
    let src = "struct P { n: str, age: int }\n\
               let ps = [P { n: \"b\", age: 3 }]\n\
               ps.sort_by_int(|p| p.age)\n";
    let err = refused(src);
    assert!(err.contains("a closure"), "{err}");
    assert!(!err.contains("?mimas"), "{err}");
}

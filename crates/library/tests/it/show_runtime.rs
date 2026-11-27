//! `show` at runtime: a value whose type implements `Show` writes itself
//! through its `show` wherever it prints, nested in lists, tuples, dicts
//! and struct fields, in echoes and in check messages. A `show` that prints
//! its own value, faults or runs too long falls back to the built-in form.

/// What the program wrote to its output, line by line.
fn printed(src: &str) -> Vec<String> {
    use std::{cell::RefCell, rc::Rc};
    let mut vm = vm::Vm::compile(src, library::std).expect("compiles");
    let lines = Rc::new(RefCell::new(Vec::new()));
    let sink = lines.clone();
    vm.fixture::<vm::fixtures::Out>().set(move |l| sink.borrow_mut().push(l.to_string()));
    vm.run().expect("runs");
    lines.borrow().clone()
}

const DOT: &str = "struct Dot { x: int }\nimpl Show for Dot { fn show(self) -> str { f\"dot {self.x}\" } }\n";

// r[verify mimas.show.runtime]
#[test]
fn nested_values_print_through_show() {
    let src = format!(
        "{DOT}struct Bag {{ d: Dot, n: int }}\n\
         let ds = [Dot {{ x: 1 }}, Dot {{ x: 2 }}]\n\
         print(ds)\nprint(f\"{{ds}}\")\nprint(Bag {{ d: Dot {{ x: 3 }}, n: 4 }})\n\
         print((Dot {{ x: 5 }}, \"a\"))\nprint({{\"k\": Dot {{ x: 6 }}}})\n"
    );
    assert_eq!(
        printed(&src),
        ["[dot 1, dot 2]", "[dot 1, dot 2]", "Bag { d: dot 3, n: 4 }", "(dot 5, \"a\")", "{k: dot 6}"]
    );
}

// r[verify mimas.show.runtime]
#[test]
fn echoes_and_check_messages_use_show() {
    let src = format!("{DOT}[Dot {{ x: 1 }}]\ncheck [Dot {{ x: 1 }}] == [Dot {{ x: 2 }}]\n");
    let mut vm = vm::Vm::compile(&src, library::std).unwrap();
    vm.fixture::<vm::fixtures::Echoes>().on.set(true);
    let results = vm.run_tests();
    let err = results[0].error.clone().expect("fails");
    assert!(err.contains("[dot 1] == [dot 2]"), "{err}");
    let echoes: Vec<String> = vm.fixture::<vm::fixtures::Echoes>().take().into_iter().map(|(_, t, _)| t).collect();
    assert!(echoes.iter().any(|t| t == "[dot 1]"), "{echoes:?}");
}

// A `show` that prints its own value gets the built-in form for it; one
// that faults or runs past its op budget leaves the built-in form and a
// note, and the run goes on.
// r[verify mimas.show.runtime]
#[test]
fn a_show_that_recurses_faults_or_loops_falls_back() {
    let src = "struct Me { x: int }\nimpl Show for Me { fn show(self) -> str { f\"me{[self]}\" } }\n\
               print([Me { x: 1 }])\n\
               struct Bad { x: int }\nimpl Show for Bad { fn show(self) -> str { let xs = [1]; f\"{xs[9]}\" } }\n\
               print([Bad { x: 2 }])\n\
               struct Slow { x: int }\nimpl Show for Slow { fn show(self) -> str { let n = 0; while true { n += 1 }; \"x\" } }\n\
               print([Slow { x: 3 }])\nprint(\"after\")\n";
    let out = printed(src);
    assert_eq!(out[0], "[me[Me { x: 1 }]]");
    assert!(out[1].starts_with("[Bad { x: 2 } (show failed: index out of bounds"), "{}", out[1]);
    assert!(out[2].starts_with("[Slow { x: 3 } (show failed: this program ran too long"), "{}", out[2]);
    assert_eq!(out[3], "after");
}

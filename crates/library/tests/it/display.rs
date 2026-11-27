//! One renderer for `print`, f-strings, echo and check messages (combo
//! item C12): a struct shows its field names and declared units, a tuple
//! its parentheses, a quantity its unit wherever the static unit is known.

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

// The C12 acceptance line: the three paths agree.
// r[verify mimas.display.one]
#[test]
fn a_struct_shows_field_names_and_units_on_every_path() {
    let src = "struct M { w: px, n: str }\n\
               let m = M { w: 3.0px, n: \"a\" }\n\
               print(m)\nprint(f\"{m}\")\nprint([m])\n";
    assert_eq!(
        printed(src),
        ["M { w: 3 px, n: \"a\" }", "M { w: 3 px, n: \"a\" }", "[M { w: 3 px, n: \"a\" }]"]
    );
}

// d10, d13: a quantity keeps its unit in an f-string, in a list and in
// a tuple; a stripped one (`to`) is a plain number.
// r[verify mimas.display.one]
#[test]
fn units_survive_fstrings_lists_and_tuples() {
    let src = "let a = 2.0px\nlet t = 2s\n\
               print(a)\nprint([a, a])\nprint((a, a))\nprint(f\"w={a} t={t}\")\nprint(a.to(1px))\n";
    assert_eq!(printed(src), ["2 px", "[2, 2] px", "(2 px, 2 px)", "w=2 px t=2 s", "2.0"]);
}

// g09, x14: a tuple prints in parentheses, through a generic and from a
// dict walk; a one-item tuple keeps its comma.
// r[verify mimas.display.one]
#[test]
fn tuples_print_in_parentheses() {
    let src = "fn swap<A, B>(t: (A, B)) -> (B, A) { (t.1, t.0) }\n\
               print(swap((1, \"a\")))\n\
               for k in {\"x\": 1} { print(k) }\n\
               print([(1, \"x\")])\nprint((5,))\n";
    assert_eq!(printed(src), ["(\"a\", 1)", "(\"x\", 1)", "[(1, \"x\")]", "(5,)"]);
}

// Positional structs, enum variants and atoms read as they are written.
// r[verify mimas.display.one]
#[test]
fn positional_forms_read_as_written() {
    let src = "enum Shape { Circle { r: float }, Square(int), Dot }\nstruct V(int, int)\n\
               print(Shape::Circle { r: 1.0 })\nprint(Shape::Square(4))\nprint(Shape::Dot)\n\
               print(V(1, 2))\nprint(#Pair(1, \"x\"))\n";
    assert_eq!(
        printed(src),
        ["Shape::Circle { r: 1.0 }", "Shape::Square(4)", "Shape::Dot", "V(1, 2)", "#Pair(1, \"x\")"]
    );
}

// u05: a failing check names the struct's fields; a tuple check its
// parentheses.
// r[verify mimas.display.one]
#[test]
fn check_messages_use_the_same_renderer() {
    let mut vm = vm::Vm::compile(
        "struct P { x: int }\ncheck P { x: 1 } == P { x: 2 }\ncheck (1, 2) == (1, 3)\n",
        library::std,
    )
    .unwrap();
    let results = vm.run_tests();
    let errs: Vec<String> = results.iter().filter_map(|r| r.error.clone()).collect();
    assert!(errs[0].contains("P { x: 1 } == P { x: 2 }"), "{errs:?}");
    assert!(errs[1].contains("(1, 2) == (1, 3)"), "{errs:?}");
}

// A field's declared unit shows in the unit the page wrote.
// r[verify mimas.display.one]
#[test]
fn a_field_shows_the_page_unit() {
    let src = "struct S { ws: [px], k: kW }\nprint(S { ws: [3.0px], k: 25kW })\n";
    assert_eq!(printed(src), ["S { ws: [3] px, k: 25 kW }"]);
}

// A tuple field shows each slot's unit, named or positional.
// r[verify mimas.display.one]
#[test]
fn a_tuple_field_shows_each_slots_unit() {
    let src = "struct T { p: (px, px), q: (px, s, str), n: int }\n\
               print(T { p: (1.0px, 2.0px), q: (1.0px, 2.0s, \"a\"), n: 3 })\n\
               struct U(px, (px, px))\nprint(U(1.0px, (2.0px, 3.0px)))\n";
    assert_eq!(
        printed(src),
        ["T { p: (1 px, 2 px), q: (1 px, 2 s, \"a\"), n: 3 }", "U(1 px, (2 px, 3 px))"]
    );
}

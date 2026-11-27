//! Enum variant fields take defaults, as struct fields do (L3).

use crate::test_runner::try_execute;

const SRC: &str = "enum Img {
    Dot { r: int },
    Group { kids: [Img], alpha: float = 1.0, w: px = 0.0px },
}
fn alpha_of(i: Img) -> float { match i { Img::Group { alpha, .. } => alpha, Img::Dot { .. } => 0.0 } }
fn w_of(i: Img) -> px { match i { Img::Group { w, .. } => w, Img::Dot { .. } => 0.0px } }";

test_run!(
    // r[verify mimas.enum.field-default]
    a_variant_literal_may_leave_out_a_defaulted_field,
    SRC,
    "alpha_of(Img::Group { kids: [] })" => "1.0",
    "alpha_of(Img::Group { kids: [], alpha: 0.5 })" => "0.5",
    // a nested variant fills its own defaults
    "alpha_of(Img::Group { kids: [Img::Group { kids: [Img::Dot { r: 1 }] }] })" => "1.0",
    "{ let g = Img::Group { kids: [Img::Group { kids: [], alpha: 0.25 }] }; match g { Img::Group { kids, .. } => alpha_of(kids[0]), Img::Dot { .. } => 0.0 } }" => "0.25",
);

test_run!(
    // r[verify mimas.enum.field-default]
    // a defaulted field keeps its unit through the dims pass
    a_defaulted_field_keeps_its_unit,
    SRC,
    "(w_of(Img::Group { kids: [] }) + 2.0px).to(1px)" => "2.0",
);

#[test]
// r[verify mimas.enum.field-default]
fn a_defaulted_field_displays_and_checks_units() {
    use std::{cell::RefCell, rc::Rc};
    let src = format!("{SRC}\nprint(Img::Group {{ kids: [] }})\nprint(Img::Group {{ kids: [], w: 0.0px }})\n");
    let mut vm = vm::Vm::compile(&src, library::std).expect("compiles");
    let lines = Rc::new(RefCell::new(Vec::new()));
    let sink = lines.clone();
    vm.fixture::<vm::fixtures::Out>().set(move |l| sink.borrow_mut().push(l.to_string()));
    vm.run().expect("runs");
    let out = lines.borrow().clone();
    // the defaulted value prints as the same value written out does
    assert!(out[0].contains("alpha: 1.0"), "{out:?}");
    assert_eq!(out[0], out[1], "a defaulted field must display as a written one");
    // the default's unit is checked: px plus s is refused
    let err = format!(
        "{:?}",
        try_execute(&format!("{SRC}\nlet TEST_VALUE = w_of(Img::Group {{ kids: [] }}) + 1.0s;")).unwrap_err()
    );
    assert!(!err.is_empty());
}

test_fail!(
    // r[verify mimas.enum.field-default]
    // a field without a default must still be written
    a_variant_field_without_a_default_is_required,
    "enum E { V { a: int, b: float = 1.0 } }\nlet TEST_VALUE = E::V { b: 2.0 };"
);

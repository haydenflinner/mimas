//! The builtin `sheet` unit on host natives: a card's coordinates on its
//! page's sheet. A native declares them with the `Sheet*` wrappers or
//! `#[mimas_dim("sheet↓")]` at registration, and the checker keeps them
//! apart from every pixel space.

use vm::{Captured, Ctx, MimasStruct, SheetDown, SheetRight, Vm};

#[derive(MimasStruct, Clone, PartialEq, Debug)]
struct Spot {
    #[mimas_dim("sheet→")]
    x: f64,
    #[mimas_dim("sheet↓")]
    y: f64,
}

fn spot(_ctx: Ctx<'_>) -> Spot {
    Spot { x: 10.0, y: 40.0 }
}

fn place(_ctx: Ctx<'_>, x: SheetRight, y: SheetDown) -> f64 {
    x.0 * 1000.0 + y.0
}

fn install(api: &mut vm::api::Api<'_, '_>) {
    api.add_adt::<Spot>();
    api.add(spot);
    api.add(place);
}

fn run(src: &str) -> Captured {
    let mut vm = Vm::execute(src, install).expect("compiles and runs");
    vm.resolve_name("TEST_VALUE").expect("TEST_VALUE bound")
}

fn refused(src: &str, needle: &str) {
    let err = Vm::execute(src, install).map(|_| ()).expect_err(src);
    let text = format!("{err:?}");
    assert!(text.contains(needle), "`{src}`: expected `{needle}` in:\n{text}");
}

// r[verify mimas.unit.sheet]
// r[verify mimas.unit.direction.endless]
#[test]
fn a_literal_adopts_the_natives_y_down_slot() {
    // `20.0sheet` is written in the slot's own terms: 20 down, not -20
    assert_eq!(run("let TEST_VALUE = place(3.0sheet, 20.0sheet);"), Captured::Float(3020.0));
    // a field's direction carries through arithmetic and back in
    assert_eq!(
        run("let s = spot();\nlet TEST_VALUE = place(s.x, s.y + 5.0sheet);"),
        Captured::Float(10045.0)
    );
}

// r[verify mimas.unit.sheet]
#[test]
fn a_pixel_never_reaches_a_sheet_slot() {
    refused("let a = place(3.0px, 20.0sheet);", "expected");
    refused("let a = place(3.0sheet, 20.0spx↓);", "expected");
    refused("let a = place(3.0sheet, 20.0wpx);", "expected");
    refused("let s = spot();\nlet a = s.y + 2.0px;", "cannot add");
}

// r[verify mimas.unit.direction.refuse]
#[test]
fn an_up_y_refuses_the_down_slot() {
    refused("let a = place(3.0sheet, 20.0sheet↑);", "point opposite ways");
    refused("let a = place(3.0sheet, 20.0sheet↑);", "`(v)↓`");
    // the written conversion passes and negates
    assert_eq!(
        run("let TEST_VALUE = place(0.0sheet, (20.0sheet↑)↓);"),
        Captured::Float(-20.0)
    );
}

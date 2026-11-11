//! A host paints a value by calling its `draw` from outside: on a binding, or on each
//! element of a list binding in turn.
use vm::{Inspect, Vm};

#[test]
fn the_elements_of_a_list_draw_in_turn() {
    let mut vm = Vm::compile(
        "pact Draw { fn draw(self) -> int; }
         struct Face { n: int }
         impl Draw for Face { fn draw(self) -> int { self.n * 10 } }
         struct Blank { n: int }
         let faces = [Face { n = 1 }, Face { n = 3 }];
         let blanks = [Blank { n = 2 }];
         let one = Face { n = 4 };
         let nums = [1, 2];",
        |_| {},
    )
    .unwrap();
    vm.run().unwrap();
    let depth = vm.depth();
    // each element, by its index
    let drawn = vm.resolve_elems_for_scene("faces", "draw").unwrap();
    let got: Vec<(usize, i64)> = drawn
        .iter()
        .map(|(i, v)| match v {
            Inspect::Int(n) => (*i, *n),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(got, [(0, 10), (1, 30)]);
    assert_eq!(vm.depth(), depth);
    // a list whose elements have no `draw` is an empty answer; a non-list is none
    assert!(
        vm.resolve_elems_for_scene("blanks", "draw")
            .unwrap()
            .is_empty()
    );
    assert!(
        vm.resolve_elems_for_scene("nums", "draw")
            .unwrap()
            .is_empty()
    );
    assert!(vm.resolve_elems_for_scene("one", "draw").is_none());
    assert!(vm.resolve_elems_for_scene("nope", "draw").is_none());
}

//! Protocol pacts (L4): a user type iterates in `for`, writes itself in an
//! f-string and through `print`, and compares with `<`.

const BAG: &str = "struct Bag { xs: [int] }
impl Bag {
    fn iter(self) -> () -> int? {
        let at = [0];
        let xs = self.xs;
        || { if at[0] < xs.len() { let v = xs[at[0]]; at[0] += 1; v } else { null } }
    }
}
struct P { x: int, y: int }
impl Show for P { fn show(self) -> str { f\"<{self.x},{self.y}>\" } }
struct V { n: int }
impl Ord for V { fn cmp(self, o: V) -> int { self.n - o.n } }
let bag = Bag { xs = [3, 1, 2] };";

test_run!(
    // r[verify mimas.protocol.iter]
    a_user_type_iterates_in_for,
    BAG,
    "for x in bag collect x * 10" => "[30, 10, 20]",
    "{ let s = 0; for x in bag { s += x; } s }" => "6",
);

test_run!(
    // r[verify mimas.protocol.show]
    a_user_type_writes_itself,
    BAG,
    "f\"p = {P { x = 1, y = 2 }}\"" => "\"p = <1,2>\"",
    "f\"{V { n = 1 }.n}\"" => "\"1\"",
);

test_run!(
    // r[verify mimas.protocol.ord]
    a_user_type_compares_through_cmp,
    BAG,
    "V { n = 1 } < V { n = 2 }" => "true",
    "V { n = 3 } <= V { n = 2 }" => "false",
    "V { n = 3 } > V { n = 2 }" => "true",
    "V { n = 2 } >= V { n = 2 }" => "true",
    // the built-in scalars implement `Ord` too
    "3.cmp(5)" => "-1",
    "\"b\".cmp(\"a\")" => "1",
    "2.5.cmp(2.5)" => "0",
);

#[test]
// r[verify mimas.protocol.show]
fn print_calls_show() {
    use std::{cell::RefCell, rc::Rc};
    let src = format!("{BAG}\nprint(P {{ x = 1, y = 2 }})\nprint(V {{ n = 3 }})\n");
    let mut vm = vm::Vm::compile(&src, library::std).expect("compiles");
    let lines = Rc::new(RefCell::new(Vec::new()));
    let sink = lines.clone();
    vm.fixture::<vm::fixtures::Out>().set(move |l| sink.borrow_mut().push(l.to_string()));
    vm.run().expect("runs");
    // `P` implements `Show`; `V` does not and keeps the plain display
    assert_eq!(lines.borrow().clone(), ["<1,2>", "V { n: 3 }"]);
}

test_run!(
    // r[verify mimas.tuple.ord]
    // C11: tuples order lexicographically, slot by slot
    tuples_compare,
    "struct V { n: int }\nimpl Ord for V { fn cmp(self, o: V) -> int { self.n - o.n } }",
    "(1, 2) < (1, 3)" => "true",
    "(2, 0) < (1, 9)" => "false",
    "(1, \"b\") >= (1, \"a\")" => "true",
    "(1, 2) <= (1, 2)" => "true",
    "((1, 2), 3) < ((1, 3), 0)" => "true",
    "(V { n = 1 }, 0) < (V { n = 2 }, 0)" => "true",
    "[(2, 1), (1, 5), (1, 2)].sorted_by(|p: (int, int)| p.0 * 10 + p.1)" => "[[1, 2], [1, 5], [2, 1]]",
);

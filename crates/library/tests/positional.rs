// A struct without naming each field (`mimas.struct.positional`): its name
// called like a function, `V(0, -1)`, or a tuple where the struct is
// expected, `[(0, -1), (1, 1)]`. Fields fill in declaration order; a named
// argument (`V(y = 5, x = 2)`) fills its field by name.
#[macro_use]
mod test_runner;

const DECLS: &str = "struct V { x: float, y: float }
     struct Piece { cells: [V] }
     fn f(v: V) -> float { v.x + v.y }";

// r[verify mimas.struct.positional]
test_run!(
    a_struct_name_calls_like_a_function,
    DECLS,
    "V(0, -1).y" => "-1.0",
    "V(y = 5, x = 2).x" => "2.0",
    "Piece([V(0, -1), V(1, 1)]).cells[1].x" => "1.0",
);

// r[verify mimas.struct.positional]
test_run!(
    a_tuple_fills_a_struct_slot,
    DECLS,
    "{ let p: V = (5, 6); p.x }" => "5.0",
    "f((1, 2))" => "3.0",
    "Piece { cells: [(0, -1), (0, 0), (1, 1)] }.cells[2].x" => "1.0",
    // with nothing expecting a struct, a tuple stays a tuple
    "{ let t = (1, 2); t.0 }" => "1",
);

// r[verify mimas.struct.positional] r[verify mimas.record.const]
test_run!(
    positional_structs_fold_in_consts,
    "struct V { x: float, y: float }
     struct Piece { cells: [V] }
     const L = Piece { cells: [(0, -1), (0, 0), (0, 1), (1, 1)] }
     const M = Piece([V(2, 3)])
     const O: V = (7, 8);",
    "L.cells[3].x" => "1.0",
    "M.cells[0].y" => "3.0",
    "O.y" => "8.0",
);

// r[verify mimas.struct.positional]
test_fail!(
    positional_mistakes_are_compile_errors,
    "struct V { x: float, y: float } let v = V(1, 2, 3);",
    "struct V { x: float, y: float } let v = V(1);",
    "struct V { x: float, y: float } let v = V(z = 1, x = 2);",
    "struct V { x: float, y: float } let v: V = (1, 2, 3);",
);

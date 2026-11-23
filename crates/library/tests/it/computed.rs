// A `const` may call any pure Mimas fn (`mimas.const.computed`). A value that
// folding can't run (a loop, a list push, a string method, a notation's
// reader) is computed once, before the program runs, by the real VM. It must
// be pure: a call with effects is a compile error.

const READER: &str = "struct V { x: float, y: float }
     struct Piece { cells: [V] }
     // a notation: `piece\\\"#.\\\\n##\\\"` reads a picture of the piece
     notation piece(s: str) -> Piece {
         let out: [V] = [];
         let rows = s.trim().split(\"\\n\");
         for r in 0..rows.len() {
             let line = rows[r].trim().chars();
             for c in 0..line.len() {
                 if line[c] == \"#\" { out.push(V(c.to_float(), r.to_float())); }
             }
         }
         Piece(out)
     }
     fn total(xs: [int]) -> int { let n = 0; for x in xs { n += x; } n }
     const L = piece\"
         #.
         #.
         ##\"
     const T = total([1, 2, 3])
     const BIG = T * 10
     impl Piece { fn samples() -> [Piece] { [L] } }";

// r[verify mimas.const.computed]
test_run!(
    a_const_calls_a_pure_fn,
    READER,
    "L.cells.len()" => "4",
    "L.cells[3].x" => "1.0",
    "T" => "6",
    // a const built from a computed one is computed after it
    "BIG" => "60",
    // a fn reads it like any const
    "Piece::samples()[0].cells[2].y" => "2.0",
);

// r[verify mimas.const.computed]
test_fail!(
    a_computed_const_must_be_pure,
    // io
    "fn f() -> int { print(\"hi\"); 1 } const X = f()",
    // randomness
    "fn f() -> int { int::random(5) } const X = f()",
    // and it is still a const
    "fn f() -> int { 1 + 1 } fn g() -> int { let n = 0; for i in 0..2 { n += i; } n } const X = g() X = 3;",
);

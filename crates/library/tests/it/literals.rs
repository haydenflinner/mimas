// A whole-number literal that fills a `float` slot is that float
// (`mimas.num.literal`): `let a: float = 3`, `{ x: 0, y: -1 }` for a struct of
// floats, `[1, 2]` for a `[float]`. Only literals: an `int` variable stays an
// `int`, and passing one where a `float` goes is still a type error.

// r[verify mimas.num.literal]
test_run!(
    whole_numbers_fill_float_slots,
    "struct V { x: float, y: float }
     struct Piece { cells: [V] }
     const P = Piece { cells: [{ x: 0 y: -1 } { x: 1 y: (2) }] }
     fn half(v: float) -> float { v / 2.0 }
     fn one() -> float { 1 }
     fn two() -> float { return 2; }
     let a: float = 3;
     let xs: [float] = [1, 2, -3];
     let o: float? = 4;",
    "a" => "3.0",
    "xs" => "[1.0, 2.0, -3.0]",
    "P.cells[0].y" => "-1.0",
    "P.cells[1].y" => "2.0",
    "half(5)" => "2.5",
    "one()" => "1.0",
    "two()" => "2.0",
    "o!" => "4.0",
    "V { x: 7, y: 8 }.x + 0.5" => "7.5",
    // an int slot keeps an int
    "{ let n: int = 3; n }" => "3",
);

// r[verify mimas.num.literal]
test_fail!(
    an_int_variable_is_not_widened,
    "let n = 3; let f: float = n;",
    "fn g(x: float) -> float { x } let n = 2; g(n);",
);

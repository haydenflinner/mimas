#[macro_use]
mod test_runner;

// `Foo { x }` is field punning — `Foo { x = x }` — and mixes with explicit fields
test_run!(
    struct_field_punning,
    "struct W { x: int, y: int }
     let x = 1;
     let w = W { x, y = 2 };
     struct P { a: int }
     let a = 9;
     let p = P { a };",
    "w.x" => "1",
    "w.y" => "2",
    "p.a" => "9",
    // struct patterns pun too: `P { a }` binds `a`
    "match p { P { a } => a, _ => 0 }" => "9",
);

test_fail!(
    struct_pun_only_takes_idents,
    "struct W { x: int } fn f() -> int { 1 } let w = W { f() };",
    "struct W { x: int } let a = 1; let w = W { a.x };",
);

// A const array may hold struct consts (game-kit's `DIRS4`): it has no
// scalar constant form, so each use site allocates it fresh -- writing
// into one use never reaches the next.
test_run!(
    const_array_of_struct_consts,
    "struct Dir { dx: float, dy: float }
     const RIGHT = Dir { dx = 1.0, dy = 0.0 };
     const DOWN = Dir { dx = 0.0, dy = 1.0 };
     const DIRS = [RIGHT DOWN];
     const NESTED = [[RIGHT], [DOWN RIGHT]];
     fn poke() -> float { let ds = DIRS; ds[0] = DOWN; ds[0].dx }",
    "DIRS.len()" => "2",
    "DIRS[1].dy" => "1",
    "NESTED[1][1].dx" => "1",
    "poke()" => "0",
    "DIRS[0].dx" => "1",
    "{ let s = 0.0; for d in DIRS { s = s + d.dx + d.dy } s }" => "2",
);

// An operator through a pact: `pact Dichotomy { fn inv(self) -> Self; }`
// declares `!`'s method, so `!x` on a `Dichotomy`-typed value calls
// whichever implementer's `inv` the value turns out to be, at runtime --
// the same dispatch `x.inv()` through the pact gets.
test_run!(
    unary_operator_through_a_pact,
    "pact Dichotomy { fn inv(self) -> Self; }
     struct Dial { lo: float, hi: float }
     impl Dichotomy for Dial { fn inv(self) -> Dial { Dial { lo = self.hi, hi = self.lo } } }
     enum Sky { Sun, Moon }
     impl Dichotomy for Sky {
         fn inv(self) -> Sky { match self { Sky::Sun => Sky::Moon, Sky::Moon => Sky::Sun } }
     }
     fn involutive(x: Dichotomy) -> bool { !(!x) == x }
     fn flipped(x: Dichotomy) -> Dichotomy { !x }",
    "involutive(Dial { lo = 5.0, hi = 95.0 })" => "true",
    "involutive(Sky::Sun)" => "true",
    "flipped(Sky::Sun) == Sky::Moon" => "true",
    "flipped(Dial { lo = 5.0, hi = 95.0 }) == Dial { lo = 95.0, hi = 5.0 }" => "true",
    "!true" => "false",
);

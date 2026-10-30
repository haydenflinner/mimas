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

// An operator through a pact: the built-in `pact Opp { fn inv(self)
// -> Self; }` declares `!`'s method, so `!x` on a `Opp`-typed value
// calls whichever implementer's `inv` the value turns out to be, at
// runtime -- the same dispatch `x.inv()` through the pact gets.
test_run!(
    unary_operator_through_a_pact,
    "struct Dial { lo: float, hi: float }
     impl Opp for Dial { fn inv(self) -> Dial { Dial { lo = self.hi, hi = self.lo } } }
     enum Sky { Sun, Moon }
     impl Opp for Sky {
         fn inv(self) -> Sky { match self { Sky::Sun => Sky::Moon, Sky::Moon => Sky::Sun } }
     }
     fn involutive(x: Opp) -> bool { !(!x) == x }
     fn flipped(x: Opp) -> Opp { !x }",
    "involutive(Dial { lo = 5.0, hi = 95.0 })" => "true",
    "involutive(Sky::Sun)" => "true",
    "flipped(Sky::Sun) == Sky::Moon" => "true",
    "flipped(Dial { lo = 5.0, hi = 95.0 }) == Dial { lo = 95.0, hi = 5.0 }" => "true",
    "!true" => "false",
);

// `opp Sky { Sun, Moon }` declares a dichotomy: the enum of its two ends
// plus a synthesized `impl Opp` whose `inv` swaps them. `opp` stays
// an ordinary name anywhere a type name and `{` don't follow it, and a
// unit variant prints as its bare name.
test_run!(
    opp_declares_a_dichotomy,
    "opp Sky { Sun, Moon }
     opp Hand { Left, Right }
     fn involutive(x: Opp) -> bool { !(!x) == x }
     fn other(x: Opp) -> Opp { !x }
     let opp = 3;",
    "!Sky::Sun == Sky::Moon" => "true",
    "not Hand::Right == Hand::Left" => "true",
    "Sky::Moon.inv() == Sky::Sun" => "true",
    "involutive(Sky::Sun) && involutive(Hand::Right)" => "true",
    "other(Hand::Left) == Hand::Right" => "true",
    "match !Sky::Moon { Sky::Sun => 1, Sky::Moon => 2 }" => "1",
    "f\"{!Sky::Sun}\"" => "\"Sky::Moon\"",
    "opp + 1" => "4",
    // the built-in pairs every program has
    "!Size::Big == Size::Small && !Sign::Neg == Sign::Pos" => "true",
    "!Fill::Full == Fill::Empty" => "true",
    "!Order::Asc == Order::Desc" => "true",
    "involutive(Daytime::Night) && involutive(Flow::Push) && involutive(Shade::Dark)" => "true",
);

// Std's operations on the built-in opps, and the mirror law each obeys:
// flipping the opp is reversing the result. `Increasing`/`Decreasing` and
// `Smaller`/`Bigger` are alternative spellings of the same ends.
test_run!(
    builtin_opp_operations_mirror,
    "fn rev<T>(xs: [T]) -> [T] {
         let out: [T] = [];
         for k in xs.len() { out.push(xs[xs.len() - 1 - k]) }
         out
     }
     let ks = [3.0, 9.0, 1.0, 9.0, 4.0];
     let xs = [1, 2];",
    "Order::Asc.rank(ks) == [2, 0, 4, 1, 3]" => "true",
    "(!Order::Asc).rank(ks) == rev(Order::Asc.rank(ks))" => "true",
    "{ let names = [\"c\", \"i\", \"a\", \"j\", \"d\"]; names.reorder(Order::Desc.rank(ks)); names }" => "[\"j\", \"i\", \"d\", \"c\", \"a\"]",
    "Side::End.pad(xs, 4, 0) == [1, 2, 0, 0] && Side::Start.pad(xs, 4, 0) == [0, 0, 1, 2]" => "true",
    "(!Side::End).pad(xs, 4, 0) == rev(Side::End.pad(rev(xs), 4, 0))" => "true",
    "Side::End.pad(xs, 1, 0) == xs" => "true",
    "Side::Start.justify(\"7\", 3) + \"|\" + Side::End.justify(\"7\", 3, \"0\")" => "\"  7|700\"",
    "Order::Increasing == Order::Asc && Order::Decreasing == Order::Desc" => "true",
    "!Order::Increasing == Order::Decreasing" => "true",
    "Compare::Bigger == Compare::Greater && !Compare::Smaller == Compare::Bigger" => "true",
);

// A field-less variant is a constant: `const ME = Team::Blue` (built at
// each use, like any struct const), and a const array may hold variants.
// A tuple variant's constructor is a fn, not a value -- still rejected.
test_run!(
    unit_variant_consts,
    "opp Team { Blue, Red }
     enum Shape { Dot, Box(int) }
     const ME = Team::Blue;
     const BOTH = [Team::Blue, Team::Red];
     const D = Shape::Dot;
     fn mine(t: Team) -> bool { t == ME }",
    "mine(Team::Blue) && !mine(!ME)" => "true",
    "BOTH[1] == Team::Red && BOTH.len() == 2" => "true",
    "D == Shape::Dot" => "true",
);

// `bool` is the first opp: the built-in source's `impl Opp for bool` lets a
// bool go wherever an `Opp` is expected -- through the same runtime dispatch
// as an enum -- while `!b` stays the built-in op.
test_run!(
    bool_is_an_opp,
    "opp Sky { Sun, Moon }
     fn involutive(x: Opp) -> bool { !(!x) == x }
     fn flips(x: Opp) -> bool { !x != x }",
    "involutive(true) && involutive(false)" => "true",
    "involutive(Sky::Sun) && involutive(true)" => "true",
    "flips(false) && flips(Sky::Moon)" => "true",
    "true.inv() == false" => "true",
    "!true" => "false",
    "!(1 < 2)" => "false",
);

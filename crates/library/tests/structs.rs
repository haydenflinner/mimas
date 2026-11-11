#[macro_use]
mod test_runner;

// `Foo { x }` is field punning — `Foo { x = x }` — and mixes with explicit fields
test_run!(
    // r[verify mimas.struct.shorthand]
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
    // r[verify mimas.const.variant]
    const_array_of_struct_consts,
    "struct Dir { dx: float, dy: float }
     const RIGHT = Dir { dx = 1.0, dy = 0.0 };
     const DOWN = Dir { dx = 0.0, dy = 1.0 };
     const DIRS = [RIGHT DOWN];
     const NESTED = [[RIGHT], [DOWN RIGHT]];
     fn poke() -> float { let ds = DIRS; ds[0] = DOWN; ds[0].dx }",
    "DIRS.len()" => "2",
    "DIRS[1].dy" => "1.0",
    "NESTED[1][1].dx" => "1.0",
    "poke()" => "0.0",
    "DIRS[0].dx" => "1.0",
    "{ let s = 0.0; for d in DIRS { s = s + d.dx + d.dy } s }" => "2.0",
);

// A `#[ctor]` assoc fn packs its args into the receiver's named fields and
// nothing else, so a const may hold the call: `Vec2::new(1.0, 0.0)` is the
// `Vec2 { x = 1.0, y = 0.0 }` struct literal spelled as a call. Bare and
// inside a const array (each use site allocates fresh, like a struct const).
test_run!(
    // r[verify mimas.const.ctor]
    const_holds_a_ctor_call,
    "use std::math::{Vec2, Vec3};
     const ORIGIN = Vec2::new(0.0, 0.0);
     const P = Vec2::new(1.0 + 1.0, 3.0);
     const POINTS = [Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)];
     const NESTED = [[Vec3::new(1.0, 2.0, 3.0)], [Vec3::new(4.0, 5.0, 6.0)]];
     fn poke() -> float { let ps = POINTS; ps[0] = Vec2::new(9.0, 9.0); ps[0].x }",
    "ORIGIN.x" => "0.0",
    "ORIGIN.y" => "0.0",
    "P.x" => "2.0",
    "POINTS.len()" => "2",
    "POINTS[1].y" => "1.0",
    "NESTED[0][0].z" => "3.0",
    "poke()" => "9.0",
    "POINTS[0].x" => "1.0",
);

// Only a `#[ctor]` call may sit in a const: an ordinary assoc fn, a
// non-ctor arg, or a named arg all stay compile errors.
test_fail!(
    non_ctor_calls_stay_non_const,
    "use std::math::Vec2; const V = Vec2::from_angle(0.5rad);",
    "use std::math::Vec2; fn y() -> float { 2.0 } const V = Vec2::new(1.0, y());",
    "use std::math::Vec2; const V = Vec2::new(x = 1.0, y = 2.0);",
);

// An operator through a pact: the built-in `pact Opp { fn inv(self)
// -> Self; }` declares `!`'s method, so `!x` on a `Opp`-typed value
// calls whichever implementer's `inv` the value turns out to be, at
// runtime -- the same dispatch `x.inv()` through the pact gets.
test_run!(
    // r[verify mimas.pact.operator]
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
    // r[verify mimas.opp.decl]
    // r[verify mimas.opp.builtin]
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
    // r[verify mimas.opp.builtin]
    // the built-in pairs every program has
    "!BigSmall::Big == BigSmall::Small && !PosNeg::Neg == PosNeg::Pos" => "true",
    "!DayNight::Day == DayNight::Night && !LightDark::Light == LightDark::Dark" => "true",
    "!AscDesc::Asc == AscDesc::Desc" => "true",
    "involutive(DayNight::Night) && involutive(StartEnd::Start) && involutive(LightDark::Dark)" => "true",
);

// Std's operations on the built-in opps, and the mirror law each obeys:
// flipping the opp is reversing the result.
test_run!(
    // r[verify mimas.std.order-rank]
    // r[verify mimas.std.side-pad]
    builtin_opp_operations_mirror,
    "fn rev<T>(xs: [T]) -> [T] {
         let out: [T] = [];
         for k in xs.len() { out.push(xs[xs.len() - 1 - k]) }
         out
     }
     let ks = [3.0, 9.0, 1.0, 9.0, 4.0];
     let xs = [1, 2];",
    "AscDesc::Asc.rank(ks) == [2, 0, 4, 1, 3]" => "true",
    "(!AscDesc::Asc).rank(ks) == rev(AscDesc::Asc.rank(ks))" => "true",
    "{ let names = [\"c\", \"i\", \"a\", \"j\", \"d\"]; names.reorder(AscDesc::Desc.rank(ks)); names }" => "[\"j\", \"i\", \"d\", \"c\", \"a\"]",
    "StartEnd::End.pad(xs, 4, 0) == [1, 2, 0, 0] && StartEnd::Start.pad(xs, 4, 0) == [0, 0, 1, 2]" => "true",
    "(!StartEnd::End).pad(xs, 4, 0) == rev(StartEnd::End.pad(rev(xs), 4, 0))" => "true",
    "StartEnd::End.pad(xs, 1, 0) == xs" => "true",
    "StartEnd::Start.justify(\"7\", 3) + \"|\" + StartEnd::End.justify(\"7\", 3, \"0\")" => "\"  7|700\"",
);

// A field-less variant is a constant: `const ME = Team::Blue` (built at
// each use, like any struct const), and a const array may hold variants.
// A tuple variant's constructor is a fn, not a value -- still rejected.
test_run!(
    // r[verify mimas.const.variant]
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
    // r[verify mimas.opp.builtin]
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

// `space` — axes taken together: the struct of its fields plus a flip of
// every field (`impl Opp`), so a space is an axis itself and nests. Ordered
// quantities are axes (their flip is negation); a field without a flip is a
// targeted error. A number serves static flips only: it doesn't travel as an
// `Opp`-typed value (that dispatch tests the value's kind at runtime).
test_run!(
    // r[verify mimas.space.decl]
    // r[verify mimas.opp.number]
    space_flips_every_axis,
    "opp Hand { Left, Right }
     opp Grip { Lib, Auth }
     space Politics { econ: Hand, power: Grip }
     space Frame { x: float, y: int, on: bool, view: Politics, w: px }
     fn involutive(o: Opp) -> bool { !(!o) == o }
     let p = Politics { econ = Hand::Left, power = Grip::Auth };
     let f = Frame { x = 1.5, y = 2, on = true, view = p, w = 10.0px };",
    "!p == Politics { econ = Hand::Right, power = Grip::Lib }" => "true",
    "involutive(p)" => "true",
    "(!f).x == -1.5 && (!f).y == -2 && !(!f).on && (!f).view == !p" => "true",
    "(!f).w == -10.0px" => "true",
    "!(!f) == f" => "true",
);

// r[verify mimas.space.axes]
// r[verify mimas.opp.number]
#[test]
fn space_fields_must_be_axes() {
    let reject = |src: &str, needle: &str| {
        let err = test_runner::try_execute(src).expect_err(src);
        let text = format!("{err:?}");
        assert!(
            text.contains(needle),
            "`{src}`: expected `{needle}` in:\n{text}"
        );
    };
    reject("space Bad { name: str }", "`name: str` isn't an axis");
    reject(
        "struct Data { n: int }\nspace Bad { d: Data }",
        "`d: Data` isn't an axis",
    );
    reject("space Empty {}", "a `space` needs at least one axis");
}

// r[verify mimas.opp.number]
// r[verify mimas.pact.primitive]
// a number goes where an `Opp` is expected too: the chain tests the value's
// kind at run time and its flip is negation
test_run!(
    numbers_pass_as_opp_values,
    "fn flip(o: Opp) -> Opp { !o }
     fn involutive(o: Opp) -> bool { !(!o) == o }",
    "flip(3.0) == -3.0" => "true",
    "flip(2) == -2" => "true",
    "involutive(1.5) && involutive(-4)" => "true",
);

// r[verify mimas.struct.update]
// `S { f = v, ..base }`: the listed fields change, the rest are read from
// `base` once; the base must be the same struct, and an enum variant's
// fields can't be filled this way.
test_run!(
    struct_update,
    "struct P { x: int, y: int, z: int }
     fn up(p: P) -> P { P { y = p.y + 1, ..p } }
     fn calls() -> int { 0 }
     let n = 0;
     fn base() -> P { n += 1; P { x = 1, y = 2, z = 3 } }",
    "up(P { x = 1, y = 2, z = 3 }).y" => "3",
    "up(P { x = 1, y = 2, z = 3 }).z" => "3",
    "{ let q = P { x = 9, ..base() }; q.x + q.y + q.z }" => "14",
    "{ let _q = P { x = 9, ..base() }; n }" => "1",
);

#[test]
fn struct_update_is_checked() {
    let rejected = |src: &str, needle: &str| {
        let err = test_runner::try_execute(src).expect_err("should be rejected");
        let text = format!("{err:?}");
        assert!(
            text.contains(needle),
            "`{src}`: expected `{needle}` in:\n{text}"
        );
    };
    rejected(
        "struct P { x: int, y: int }\nstruct Q { x: int, y: int }\nlet q = Q { x = 1, y = 2 };\nlet p = P { x = 3, ..q };",
        "mismatch",
    );
    rejected(
        "enum E { A { x: int, y: int }, B { x: int, y: int } }\nlet b = E::B { x = 1, y = 2 };\nlet a = E::A { x = 3, ..b };",
        "needs a struct",
    );
}

// -- bounded type params with the std in scope -------------------------------
// `<T: Show>` reaches the pact's methods through a closure over `[T]`: the array's
// `map` is a std method, so the solver/vm harnesses (no std) can't host this one.
test_run!(
    bounded_param_maps_over_array,
    "pact Show { fn show(self) -> str; }
     struct Dot { x: int }
     impl Show for Dot { fn show(self) -> str { f\"d{self.x}\" } }
     fn labels<T: Show>(xs: [T]) -> [str] { xs.map(|x| x.show()) }
     fn same<T>(xs: [T]) -> [T] { xs.map(|x| x) }",
    "labels([Dot { x = 1 }, Dot { x = 2 }])" => "[\"d1\", \"d2\"]",
    "same([1, 2])" => "[1, 2]",
);

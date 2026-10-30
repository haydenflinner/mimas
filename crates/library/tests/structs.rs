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

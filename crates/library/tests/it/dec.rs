// `dec`: an exact decimal in 64 bits (docs/decimal.md). r[verify mimas.dec]

test_run!(
    dec_arithmetic_is_exact,
    r#""0.1".to_dec()! + "0.2".to_dec()! == "0.3".to_dec()!"# => "true",
    r#"("1.10".to_dec()! + "2.20".to_dec()!).to_str()"# => r#""3.30""#,
    r#"("2.675".to_dec()!).round(2).to_str()"# => r#""2.68""#,
    r#"("1".to_dec()! / 3).to_str()"# => r#""0.3333333333333333""#,
    r#""0.3".to_dec()! < "0.1".to_dec()! + "0.2".to_dec()!"# => "false",
    r#"("1.5".to_dec()! * 2).to_str()"# => r#""3.0""#,
    r#"("1.5".to_dec()! + 0.25).to_str()"# => r#""1.75""#,
    r#"(-"2.5".to_dec()!).to_str()"# => r#""-2.5""#,
    r#"0.1.to_dec().to_str()"# => r#""0.1""#,
    r#"3.to_dec() == 3"# => "true",
);

// a numeric literal in a `dec` slot, or beside a `dec`, is exact
// r[verify mimas.dec.literal]
test_run!(
    dec_literals_take_their_slot,
    "let p: dec = 0.30;\nfn total(a: dec, b: dec) -> dec { a + b }\nstruct Item { cost: dec }",
    "p.to_str()" => r#""0.30""#,
    "(p + 0.25).to_str()" => r#""0.55""#,
    "p == 0.3" => "true",
    "total(0.1, 0.2) == 0.3" => "true",
    "(p * 3).to_str()" => r#""0.90""#,
    "(-p).to_str()" => r#""-0.30""#,
    "Item { cost: 3.5 }.cost + 0.1 == 3.6" => "true",
    "p < 0.31" => "true",
);

// `==` between a `dec` and a `float` is by value, as floats (mimas.dec):
// it used to answer `false` even for `2.5 == 2.5`
test_run!(
    dec_equals_a_float_by_value,
    "let v: dec = 2.5;\nlet f: float = 2.5;\nlet n: dec = -2.68;",
    "v == f" => "true",
    "f == v" => "true",
    "n == -2.68" => "true",
    "n != -2.68" => "false",
    "v == 2.6" => "false",
);

test_run!(
    arithmetic_of_written_numbers_fills_a_dec_slot,
    "let subtotal: dec = 2.99 + 4.50 + 0.75;\nlet half: dec = (1 + 0.5) / 2;",
    "subtotal.to_str()" => r#""8.24""#,
    "(subtotal * 3).to_str()" => r#""24.72""#,
    "half == 0.75" => "true",
);

// commas group thousands in a number written out; underscores too
// r[verify mimas.num.groups]
test_run!(
    numbers_group_with_commas_and_underscores,
    "1,000,000" => "1000000",
    "1,234.50" => "1234.5",
    "1_000_000" => "1000000",
    // inside brackets a comma always separates (L2): `[1,000, 2]` has
    // three items, and `f(1,234)` passes two values
    "[1,000, 2].len()" => "3",
    "[1, 2].len()" => "2",
    "{ let n = 1,000,000; n }" => "1000000",
    "2,500lb == 2500lb" => "true",
);

test_run!(
    a_grouped_number_fills_a_dec_slot,
    "let p: dec = 1,234.50;",
    "p.to_str()" => r#""1234.50""#,
    "p + 0.25 == 1,234.75" => "true",
);

test_run!(
    approx_on_dec_is_eq,
    "let a: dec = 0.1;\nlet b: dec = 0.2;",
    "a + b ≈ 0.3.to_dec()" => "true",
);

#[macro_use]
mod test_runner;

test_run!(
    print_does_not_error,
    r#"{ print(1); print("hi"); 0 }"# => "0",
    r#"{ print([1, 2, 3]); 0 }"# => "0",
);

// a quantity prints its unit with the base-unit value (`25kW` -> `25000 W`);
// a bare number stays bare
#[test]
fn print_names_the_unit() {
    let vm = vm::Vm::execute(
        "print(25kW); print(9ft / 24px); print(42); dbg(3s); warn(5kg);",
        library::std,
    )
    .expect("runs");
    let lines = vm
        .fixture::<vm::fixtures::Prints>()
        .take()
        .into_iter()
        .map(|(_, _, text)| text)
        .collect::<Vec<_>>();
    assert_eq!(
        lines,
        [
            "25000.0 W",
            "0.11430000000000001 m/px",
            "42",
            "dbg value: 3.0 s",
            "5.0 kg",
        ]
    );
}

// a bare expression at the top level echoes its value at its line once a
// host turns echoes on, and so does a top-level `let NAME`; a print (unit),
// a check line, a destructuring let and a let inside a fn stay silent
// r[verify mimas.echo] r[verify mimas.stmt.trailing] r[verify mimas.echo.let]
#[test]
fn top_level_expressions_echo() {
    let src = "3 + 5\nlet x = 2;\nx * 3\nprint(\"hi\");\nfn f() -> int { let y = 9; y - 2 }\nf()\nlet (a, b) = (1, 2);\n\"go\" + \"!\"";
    let mut vm = vm::Vm::compile(src, library::std).expect("compiles");
    vm.fixture::<vm::fixtures::Echoes>().on.set(true);
    vm.run().expect("runs");
    let lines = vm
        .fixture::<vm::fixtures::Echoes>()
        .take()
        .into_iter()
        .map(|(loc, text, value)| {
            (
                &src[loc.span.start()..loc.span.start() + 5],
                text,
                matches!(value, vm::Inspect::Int(_) | vm::Inspect::Str(_)),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        lines,
        [
            ("3 + 5", "8".to_string(), true),
            ("let x", "2".to_string(), true),
            ("x * 3", "6".to_string(), true),
            ("f()\nl", "7".to_string(), true),
            ("\"go\" ", "\"go!\"".to_string(), true),
        ]
    );
    // off by default: a plain run records nothing
    let mut vm = vm::Vm::compile(src, library::std).expect("compiles");
    vm.run().expect("runs");
    assert!(vm.fixture::<vm::fixtures::Echoes>().take().is_empty());
}

// an echo carries its unit, a `let`'s and a bare expression's alike — the
// value a host shows beside the line reads `28.0 usd`, not `28.0`
// r[verify mimas.echo.let]
#[test]
fn echoes_carry_units() {
    let src = "let pv = 25kW;\nlet bill = pv * 8h * 0.14usd/kWh;\n3kg + 2kg\nlet n = 4;\nlet m = { let inner = 7; inner + 1 };\nfor i in 0..2 { let j = i; }";
    let mut vm = vm::Vm::compile(src, library::std).expect("compiles");
    vm.fixture::<vm::fixtures::Echoes>().on.set(true);
    vm.run().expect("runs");
    let texts: Vec<String> = vm
        .fixture::<vm::fixtures::Echoes>()
        .take()
        .into_iter()
        .map(|(_, t, _)| t)
        .collect();
    // a `let` inside a block or a loop is not a top-level binding: no echo;
    // a quantity shows in the unit the program wrote for its dimension
    assert_eq!(texts, ["25 kW", "28 usd", "5 kg", "4", "8"]);
}

fn echo_texts(src: &str) -> Vec<String> {
    let mut vm = vm::Vm::compile(src, library::std).expect("compiles");
    vm.fixture::<vm::fixtures::Echoes>().on.set(true);
    vm.run().expect("runs");
    vm.fixture::<vm::fixtures::Echoes>()
        .take()
        .into_iter()
        .map(|(_, t, _)| t)
        .collect()
}

// an echo shows a quantity in the unit the program wrote for its dimension,
// not the coherent unit: pounds stay pounds, a list shows its unit once, a
// ratio of two written units composes, and base-unit round-off is trimmed
// r[verify mimas.echo.unit]
#[test]
fn echoes_show_units_as_written() {
    let src = "let qty = [12lb, 8lb, 3lb];
let t = 12lb + 3lb;
let price = [1.10usd, 1.40usd] ;
let per = 3usd / 2lb;
let m = 2kg;
let run = 5km + 300m;";
    assert_eq!(
        echo_texts(src),
        ["[12, 8, 3] lb", "15 lb", "[1.1, 1.4] usd", "1.5 usd/lb", "4.4092452437 lb", "5.3 km"]
    );
}

// free aggregates over lists keep the unit; `count` counts trues or rows;
// the `_by` forms group by key text in first-seen order
// r[verify mimas.agg.free] r[verify mimas.agg.by]
#[test]
fn free_aggregates() {
    let src = "let qty = [12lb, 8lb, 3lb];
let kind = [\"grain\", \"bean\", \"grain\"];
sum(qty)
avg(qty)!
count(kind)
count([true, false, true])
sum([1, 2, 3])
sum_by(qty, kind)
count_by(qty, kind)
avg_by(qty, kind)";
    let got = echo_texts(src);
    assert_eq!(
        &got[2..],
        [
            "23 lb",
            "7.66666666667 lb",
            "3",
            "2",
            "6",
            "{grain: 15, bean: 8} lb",
            "{grain: 2, bean: 1}",
            "{grain: 7.5, bean: 8} lb",
        ]
    );
}

// panic has type `!`, so a panicking match/if arm coerces to the other arm's type
test_run!(
    panic_arm_coerces_via_never,
    r#"let x: int = if true { 5 } else { panic("nope") };"#,
    "x" => "5",
);

test_fail!(
    library_names_are_not_values,
    "std;",
    "array;",
    "dict;",
    "let a = std::fs;",
    r#"f"{std}";"#,
);

// the shipped prelude — solve's builtin.mim plus library's <std::opp> and
// every std_lib .mim — lints warning-free (unused bindings, dead stores,
// must_use discards): the grades pass is part of the contract pages rely on
// r[verify mimas.std.lint-clean]
#[test]
fn builtins_compile_warning_free() {
    let vm = vm::Vm::execute("", library::std).expect("prelude compiles");
    let warnings: Vec<String> = vm.warnings().iter().map(|w| format!("{w:?}")).collect();
    assert!(warnings.is_empty(), "warnings:\n{}", warnings.join("\n"));
}

// r[verify mimas.prelude.shadow]
// A user type may take a built-in opp's name: the built-in's own impls
// bound before the user file hoisted, and user code sees the user's type.
// (The built-ins themselves keep descriptive names -- `StartEnd`,
// `AscDesc` -- so the everyday `Side`/`Order` are free without shadowing.)
test_run!(
    user_types_shadow_builtin_opps,
    "enum PosNeg { Up, Down }
     struct LightDark { n: int }
     enum Side { Left, Right }
     fn pick(s: PosNeg) -> int { match s { PosNeg::Up => 1, PosNeg::Down => 2 } }",
    "pick(PosNeg::Down)" => "2",
    "LightDark { n = 3 }.n" => "3",
    "Side::Left == Side::Left" => "true",
    "!DayNight::Day == DayNight::Night" => "true",
    "StartEnd::End.pad([1], 2, 0) == [1, 0]" => "true",
);

// r[verify mimas.opp.glyph]
// r[verify mimas.ident.symbols]
// An opp's ends may carry glyphs (`glyph` on `Opp` defaults to `""`), the
// prelude ships them, and an end may be a glyph itself.
test_run!(
    opp_glyphs_and_symbol_names,
    "opp Sky { ☀, ☾ }
     opp Mood { Up \"😀\", Down \"😞\" }
     fn label(x: Opp) -> str { x.glyph() }",
    "!Sky::☀ == Sky::☾" => "true",
    "Sky::☀.glyph()" => "\"\"",
    "label(Mood::Down)" => "\"😞\"",
    "label(!Mood::Down)" => "\"😀\"",
    "DayNight::Day.glyph() + DayNight::Night.glyph()" => "\"🌞🌙\"",
    "(!AscDesc::Asc).glyph()" => "\"↘\"",
    "StartEnd::Start.glyph() + BigSmall::Small.glyph() + PosNeg::Pos.glyph() + LightDark::Dark.glyph()" => "\"⇤●➕☾\"",
    "true.glyph() + false.glyph() + (!true).glyph()" => "\"✓✗✗\"",
);

// r[verify mimas.enum.glyph]
// plain enum variants carry glyphs the same way; the synthesized method is inherent
test_run!(
    enum_glyphs,
    "enum Suit { Spades \"♠\", Hearts \"♥\", Diamonds \"♦\", Clubs \"♣\" }
     fn row(ss: [Suit]) -> str { let out = \"\"; for s in ss { out += s.glyph(); } out }",
    "Suit::Hearts.glyph()" => "\"♥\"",
    "row([Suit::Spades, Suit::Clubs])" => "\"♠♣\"",
);

// r[verify mimas.notation.user]
// A page's own tagged literal: `tag"…"` lowers to `__lit_tag("…")`, which
// the page may define directly or through the `notation` spelling.
test_run!(
    user_notations,
    "fn __lit_twice(s: str) -> str { s + s }
     notation rpn(s: str) -> int {
         let st: [int] = [];
         for w in s.split(\" \") {
             if w == \"+\" { let b = st.pop() ?? 0; let a = st.pop() ?? 0; st.push(a + b); }
             else if w == \"*\" { let b = st.pop() ?? 0; let a = st.pop() ?? 0; st.push(a * b); }
             else { st.push(w.to_int() ?? 0); }
         }
         st.pop() ?? 0
     }",
    "twice\"ab\"" => "\"abab\"",
    "rpn\"3 4 + 2 *\"" => "14",
    "rpn\"1 2 3 + +\" + 1" => "7",
);

// the shape a table card's question lines lower to: per-row blocks over
// index lists keep the columns' units, so `sum(qty * price)` reads in usd
// r[verify mimas.echo.unit]
#[test]
fn row_blocks_keep_units() {
    let src = "let __q = [12lb, 8lb, 3lb];\nlet __p = [1.1usd/lb, 1.4usd/lb, 0.9usd/lb];\nlet v = for __i in [0, 1, 2] collect { let qty = __q[__i]; let price = __p[__i]; qty * price };\nsum(v)\nlet r = [0, 1, 2].filter(|__i: int| { let qty = __q[__i]; qty < 5lb });\nfor __i in r collect __q[__i]";
    assert_eq!(
        echo_texts(src),
        ["[12, 8, 3] lb", "[1.1, 1.4, 0.9] usd/lb", "[13.2, 11.2, 2.7] usd", "27.1 usd", "[2]", "[3] lb"]
    );
}

// a unit times itself echoes as its square
// r[verify mimas.echo.unit]
#[test]
fn a_unit_times_itself_echoes_as_its_square() {
    assert_eq!(echo_texts("3px * 4px\n2m * 5m"), ["12 px²", "10 m²"]);
}

// a declared unit counts as written before the value's own literals: a
// `[lb]` list holding a `2kg` cell echoes in pounds
// r[verify mimas.echo.unit]
#[test]
fn declared_units_win() {
    assert_eq!(
        echo_texts("let w: [lb] = [2kg, 1lb];\nsum(w)\nlet p: [usd/lb] = [2usd/kg];"),
        ["[4.4092452437, 1] lb", "5.4092452437 lb", "[0.90718474] usd/lb"]
    );
}

// the shipped `Id<T>` is a nominal entity key: `Id<Thing>` and
// `Id<Hand>` are distinct types even though both wrap an `int`, and an
// `Id`-taking member subscript is the idiomatic store lookup.
// r[verify mimas.id.newtype]
test_run!(
    id_newtype_indexes_its_store,
    "struct Thing { id: int, speed: float }
     struct Entities<T> { slots: [T] }
     impl Entities {
         subscript(g: Id<T>) -> T {
             get { self.slots[g.raw()] }
         }
     }
     let e: Entities<Thing> = Entities { slots = [Thing { id = 0, speed = 1.0 }, Thing { id = 1, speed = 2.0 }] }
     let g: Id<Thing> = Id::new(1)
     let empty: Id<Thing>? = null
     let TEST_VALUE = (e[g].speed, g.raw(), empty);",
    "TEST_VALUE" => "[2.0, 1, null]",
);

// r[verify mimas.id.newtype]
test_fail!(
    a_wrong_tagged_id_cannot_index,
    "struct Thing { id: int }
     struct Entities<T> { slots: [T] }
     impl Entities {
         subscript(g: Id<T>) -> T {
             get { self.slots[g.raw()] }
         }
     }
     let e: Entities<Thing> = Entities { slots = [Thing { id = 0 }] }
     let wrong: Id<int> = Id { id = 0 }
     e[wrong]",
    "expected Id<Thing> but found Id<int>",
);

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
    // `dbg(3s)` at the top level also echoes its value — not a print line
    let lines = vm
        .fixture::<vm::fixtures::Prints>()
        .take()
        .into_iter()
        .filter(|(_, kind, _)| !matches!(kind, vm::fixtures::PrintKind::Echo))
        .map(|(_, _, text)| text)
        .collect::<Vec<_>>();
    assert_eq!(
        lines,
        [
            "25000 W",
            "0.11430000000000001 m/px",
            "42",
            "dbg value: 3 s",
            "5 kg",
        ]
    );
}

// an explicit `echo(v)` records v at its call site and returns it
#[test]
fn explicit_echo_records() {
    let vm = vm::Vm::execute("let y = echo(3 + 5);", library::std).expect("runs");
    let lines = vm
        .fixture::<vm::fixtures::Prints>()
        .take()
        .into_iter()
        .map(|(_, kind, text)| (matches!(kind, vm::fixtures::PrintKind::Echo), text))
        .collect::<Vec<_>>();
    assert_eq!(lines, [(true, "8".to_string())]);
}

// a bare expression at the top level echoes its value at its line; a let,
// a print (unit) and a check line stay silent; the script's result is unchanged
// r[verify mimas.echo] r[verify mimas.stmt.trailing]
#[test]
fn top_level_expressions_echo() {
    let vm = vm::Vm::execute(
        "3 + 5\nlet x = 2;\nx * 3\nprint(\"hi\");\nfn f() -> int { 7 }\nf()\n\"go\" + \"!\"",
        library::std,
    )
    .expect("runs");
    let lines = vm
        .fixture::<vm::fixtures::Prints>()
        .take()
        .into_iter()
        .map(|(loc, kind, text)| (loc.span.start(), matches!(kind, vm::fixtures::PrintKind::Echo), text))
        .collect::<Vec<_>>();
    assert_eq!(
        lines,
        [
            (0, true, "8".to_string()),
            (17, true, "6".to_string()),
            (23, false, "\"hi\"".to_string()),
            (56, true, "7".to_string()),
            (60, true, "\"go!\"".to_string()),
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

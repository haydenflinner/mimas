//! `AbelianGroup`, streams over any group, and tallies (`library/src/group.mim`).
//! The pact's laws run on each std impl through `group_laws` below, which
//! calls the operators through the pact bound, so every runtime dispatch arm
//! (a primitive, a `DataFrame`, a struct) is exercised.

#[macro_use]
mod test_runner;

const LAWS: &str = r#"use std::polars::*;
    /// The pact's laws over `xs`, through the bound.
    fn group_laws<T: AbelianGroup>(xs: [T]) -> bool {
        for a in xs {
            if a + a.zero() != a { return false; }
            if a + -a != a.zero() { return false; }
            if a.times(3) != a + a + a { return false; }
            if a.times(-2) != -(a + a) { return false; }
            for b in xs {
                if a + b != b + a { return false; }
                if a - b != a + -b { return false; }
                for c in xs {
                    if (a + b) + c != a + (b + c) { return false; }
                }
            }
        }
        true
    }
    let frames = [
        query z { table { item, w
            "apple", 2
            "bread", -1
        } },
        query z { table { item, w
            "fig", 3
            "apple", -2
        } },
    ];
    let tallies = [tally(["a", "a", "b"]), tally_w([("b", -1), ("c", 4)]), tally(["c"])];"#;

// r[verify mimas.group.pact]
test_run!(
    group_laws_hold_on_every_std_impl,
    LAWS,
    "group_laws([0, 1, -2, 7])" => "true",
    // sums of these floats are exact
    "group_laws([0.0, 1.0, -2.5, 0.5, 3.0])" => "true",
    // a quantity is a float with a dimension
    "group_laws([0.0px, 1.5px, -2.0px])" => "true",
    "group_laws([#1.10, #2.25, #0.05])" => "true",
    // `dec`'s own `+` stays built in, inside the impl and out
    "#1.10 + #2.25 == #3.35" => "true",
    "group_laws(frames)" => "true",
    "group_laws(frames.map(|z: DataFrame| z.weighted()))" => "true",
    "group_laws(tallies)" => "true",
    // the default `times` doubles: 13 copies in five additions
    "5.times(13) == 65 && tally([\"x\"]).times(13).weight(\"x\") == 13" => "true",
);

// A page's own impl gets the pact's laws as checks, like any pact's.
// r[verify mimas.group.pact]
#[test]
fn a_page_impl_gets_the_group_laws() {
    let src = "
        struct Money { c: int }
        impl Money { fn samples() -> [Money] { [Money { c: 0 }, Money { c: 5 }, Money { c: -3 }] } }
        impl AbelianGroup for Money {
            fn zero(self) -> Money { Money { c: 0 } }
            fn add(self, o: Money) -> Money { Money { c: self.c + o.c } }
            fn sub(self, o: Money) -> Money { Money { c: self.c - o.c } }
            fn neg(self) -> Money { Money { c: -self.c } }
        }
        struct Bad { c: int }
        impl Bad { fn samples() -> [Bad] { [Bad { c: 1 }, Bad { c: 2 }] } }
        impl AbelianGroup for Bad {
            fn zero(self) -> Bad { Bad { c: 0 } }
            fn add(self, o: Bad) -> Bad { Bad { c: self.c - o.c } }
            fn sub(self, o: Bad) -> Bad { Bad { c: self.c - o.c } }
            fn neg(self) -> Bad { Bad { c: -self.c } }
        }";
    let mut vm = vm::Vm::compile(src, library::std).unwrap_or_else(|e| panic!("{e:?}"));
    vm.run().unwrap();
    let results = vm.run_tests();
    let named = |ty: &str| -> Vec<bool> {
        results
            .iter()
            .filter(|t| t.name.contains(&format!("for {ty}:")))
            .map(|t| t.passed())
            .collect()
    };
    let money = named("Money");
    assert_eq!(money.len(), 6, "one check per law: {:?}", results.iter().map(|t| &t.name).collect::<Vec<_>>());
    assert!(money.iter().all(|p| *p), "Money keeps every law");
    // `-` for `+` is not commutative: the laws catch it
    assert!(named("Bad").iter().any(|p| !*p), "Bad breaks a law");
}

const STREAMS: &str = r#"
    let vs = stream([1.5px, 2.0px, -0.5px]);
    let ns = stream([1, 2, 3, 4]);
    fn dbl(x: int) -> int { 2 * x }"#;

// r[verify mimas.group.stream]
test_run!(
    streams_of_numbers_and_quantities,
    STREAMS,
    "(∫ vs).list()" => "[1.5, 3.5, 3.0]",
    "(Δ⁻¹ vs).list()" => "[0.0, 1.5, 3.5, 3.0]",
    "(Δ ns).list()" => "[1, 1, 1]",
    "(∇ ns).list()" => "[1, 1, 1, 1]",
    "(z⁻¹ ns).list()" => "[0, 1, 2, 3]",
    "∇ (∫ vs) == vs" => "true",
    "Δ (Δ⁻¹ ns) == ns" => "true",
    "Δ⁻¹ (Δ ns) == ns - ns[0]" => "true",
    "∇ ns == ns - z⁻¹ ns" => "true",
    "(Δ⁻¹ ∘ Δ)(ns) == ns - ns[0]" => "true",
    "(2 * ns).list()" => "[2, 4, 6, 8]",
    "(0.5 * vs).list()" => "[0.75, 1.0, -0.25]",
    "(ns + 10).list()" => "[11, 12, 13, 14]",
    "dbl^Δ(ns) == ns.map(dbl)" => "true",
    // a quantity keeps its dimension through the stream
    "{ let x: px = (∫ vs).last(); x + 1.0px }" => "4.0",
);

// r[verify mimas.group.running-total]
test_run!(
    a_running_total_is_the_last_step_of_the_integral,
    STREAMS,
    // `+=` keeps only the last value, O(1) a tick; `∫` keeps every prefix
    "{ let total = 0; for c in ns.list() { total += c; } total == (∫ ns).last() && total == ∑ ns }" => "true",
    "{ let at = 0.0px; for v in vs.list() { at += v * 1.0; } at == (∫ vs).last() }" => "true",
    "{ let held = tally([\"w\"]).zero();
       let evs = [tally([\"w\"]), tally([\"s\"]), -tally([\"w\"])];
       for e in evs { held += e; }
       held == (∫ stream(evs)).last() && held == tally([\"s\"]) }" => "true",
);

// r[verify mimas.group.stream]
#[test]
fn table_only_stream_ops_fault_with_the_step() {
    let err = match test_runner::try_execute("let s = stream([1, 2]); let _u = s ∪ s;") {
        Ok(_) => panic!("expected ∪ on an int stream to fault"),
        Err(e) => format!("{e}"),
    };
    assert!(err.contains("tables with weights"), "{err}");
    assert!(err.contains("an int"), "{err}");
}

const TALLIES: &str = r#"use std::polars::*;
    let held = tally(["w", "arrowup"]);
    let steer = tally_w([("w", 1), ("s", -1)]);
    let a = tally(["x", "x", "y"]);
    let b = tally_w([("y", 2), ("z", -1)]);
    struct Cell { r: int, c: int }"#;

// r[verify mimas.group.tally]
test_run!(
    tallies_are_light_zsets,
    TALLIES,
    "a.weight(\"x\")" => "2",
    "a.weight(\"q\")" => "0",
    "(a + b).pairs()" => "[[\"x\", 2], [\"y\", 3], [\"z\", -1]]",
    "(a - a).len()" => "0",
    "-b == tally_w([(\"y\", -2), (\"z\", 1)])" => "true",
    "3 · a == a + a + a" => "true",
    "b⁺ == tally_w([(\"y\", 2)])" => "true",
    "b⁻ == tally([\"z\"])" => "true",
    "b == b⁺ - b⁻" => "true",
    "‖b‖" => "3",
    "b.count()" => "1",
    "[\"y\" ∈ b, \"z\" ∈ b, \"q\" ∉ b]" => "[true, false, true]",
    // order does not matter to `==`
    "tally([\"p\", \"q\"]) == tally([\"q\", \"p\"])" => "true",
    "(a ∪ tally([\"y\", \"y\", \"w\"])).pairs()" => "[[\"x\", 2], [\"y\", 2], [\"w\", 1]]",
    "(a ∩ tally([\"y\", \"y\", \"w\"])).pairs()" => "[[\"y\", 1]]",
    "a ⊎ b == a + b" => "true",
    // a join on the whole item multiplies weights: held keys meet their bindings
    "(held ⋈ steer).count()" => "1",
    "a.as_set() == tally([\"x\", \"y\"])" => "true",
    // any printable item: tuples and records
    "(1, 2) ∈ tally([(1, 2), (3, 4)])" => "true",
    "Cell { r: 1, c: 0 } ∈ tally([Cell { r: 1, c: 0 }])" => "true",
    // `put` adds in place for a hot loop
    "{ let t = tally([\"k\"]); t.put(\"k\", -1); t.len() }" => "0",
    // a tally becomes a Z-set table only when asked
    "tally_table(a, \"item\") == query z { table { item, w
        \"x\", 2
        \"y\", 1
    } }" => "true",
);

// r[verify mimas.group.tally]
#[test]
fn tally_bag_ops_refuse_removals() {
    let err = match test_runner::try_execute(
        "let b = tally_w([(\"z\", -1)]); let _u = b ∪ b;",
    ) {
        Ok(_) => panic!("expected ∪ on a removal to fault"),
        Err(e) => format!("{e}"),
    };
    assert!(err.contains("negative weight"), "{err}");
}

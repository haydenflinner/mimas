//! `AbelianGroup`, streams over any group, and tallies (`library/src/group.mim`).
//! The pact's laws run on each std impl through `group_laws` below, which
//! calls the operators through the pact bound, so every runtime dispatch arm
//! (a primitive, a `DataFrame`, a struct) is exercised.

use crate::test_runner;

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
    // any item a dict can key by: tuples and structs
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

// the item is the key: items that are `==` are one item, with no printing
// r[verify mimas.group.tally.by-value]
test_run!(
    tally_keys_by_the_item,
    "struct Cell { r: int, c: int }",
    "tally([(1, 2), (1, 2), (2, 1)]).weight((1, 2))" => "2",
    "tally([(1, 2), (1, 2), (2, 1)]).len()" => "2",
    "tally([Cell { r: 1, c: 0 }, Cell { r: 1, c: 0 }]).weight(Cell { r: 1, c: 0 })" => "2",
    "tally([3, 1, 3, 2]).items()" => "[3, 1, 2]",
    "(tally([(0, 0), (0, 1)]) - tally([(0, 0)])).items()" => "[[0, 1]]",
    // a moved struct is not the item it was when it went in
    "{ let c = Cell { r: 1, c: 0 }; let t = tally([c]); c.r = 5; t.contains(Cell { r: 1, c: 0 }) && !t.contains(c) }" => "true",
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

// ---- posts and panels ------------------------------------------------------

const POSTS: &str = r#"
    let p = post(3.0px);
    let q = post(5.5px);
    let ps = posts([0.0, 1.0, 3.0, 6.0]);
    let v = stream([1.0, 2.0, 3.0]);
    struct Post2 { at: int }"#;

// r[verify mimas.group.affine]
// r[verify mimas.group.affine.post]
// r[verify mimas.group.affine.sub-posts]
// r[verify mimas.group.affine.add-panel]
test_run!(
    a_post_moves_by_panels,
    POSTS,
    "q - p" => "2.5",
    "(p + 2.0px).at" => "5.0",
    "(2.0px + p).at" => "5.0",
    "(p - 1.0px).at" => "2.0",
    "(p + (q - p)) == q" => "true",
    "{ let at = p; at += 1.5px; at -= 0.5px; at.at }" => "4.0",
    // any group's posts: an index and a vector-like tally
    "(post(3) + 2) - post(3)" => "2",
    "(post(tally([\"a\"])) + tally([\"b\"])).at == tally([\"a\", \"b\"])" => "true",
);

// r[verify mimas.group.affine.posts]
// r[verify mimas.group.affine.forward-difference]
// r[verify mimas.group.affine.before]
// r[verify mimas.group.affine.after]
test_run!(
    a_stream_of_posts_has_one_panel_fewer,
    POSTS,
    "ps.len()" => "4",
    "ps[2].at" => "3.0",
    "ps.at" => "[0.0, 1.0, 3.0, 6.0]",
    "(Δ ps).list()" => "[1.0, 2.0, 3.0]",
    "(∇ ps).list()" => "[1.0, 2.0, 3.0]",
    "ps.before().at" => "[0.0, 1.0, 3.0]",
    "ps.after().at" => "[1.0, 3.0, 6.0]",
    "(ps - ps[0]).list()" => "[0.0, 1.0, 3.0, 6.0]",
    "(ps.before() + Δ ps) == ps.after()" => "true",
    "(ps.after() - ∇ ps) == ps.before()" => "true",
    "(ps + 1.0).at" => "[1.0, 2.0, 4.0, 7.0]",
    "ps.last().at" => "6.0",
    // a group stream gets the same helpers
    "v.before().list()" => "[1.0, 2.0]",
    "v.after().list()" => "[2.0, 3.0]",
);

// r[verify mimas.group.affine.from]
test_run!(
    from_names_the_start,
    POSTS,
    "(∫ v from post(10.0)).at" => "[11.0, 13.0, 16.0]",
    "(Δ⁻¹ v from post(10.0)).at" => "[10.0, 11.0, 13.0, 16.0]",
    "(∫ v from 10.0).list()" => "[11.0, 13.0, 16.0]",
    "(Δ⁻¹ v from 10.0).list()" => "[10.0, 11.0, 13.0, 16.0]",
    "(∇ ps from post(-1.0)).list()" => "[1.0, 1.0, 2.0, 3.0]",
    "(∇ v from 1.0).list()" => "[0.0, 1.0, 1.0]",
    "(z⁻¹ ps from post(-1.0)).at" => "[-1.0, 0.0, 1.0, 3.0]",
    "(z⁻¹ v from 9.0).list()" => "[9.0, 1.0, 2.0]",
    // the default start is the zero
    "(∫ v from 0.0) == ∫ v && (Δ⁻¹ v from 0.0) == Δ⁻¹ v" => "true",
    // the fundamental theorem with its constant
    "Δ⁻¹ (Δ ps) from ps[0] == ps" => "true",
    "∫ (Δ ps) from ps[0] == ps.after()" => "true",
    "∫ (∇ ps from ps[0]) from ps[0] == ps" => "true",
    // `from` binds after the glyph's operand, and a name `from` still works
    "{ let from = 2; from + 1 }" => "3",
    "(∫ v.map(|x: float| 2.0 * x) from 1.0).list()" => "[3.0, 7.0, 13.0]",
);

// r[verify mimas.group.affine.zip]
test_run!(
    zip_pairs_aligned_streams,
    POSTS,
    "ps.before().zip(Δ ps).len()" => "3",
    "ps.after().zip(∇ ps)[0].0.at" => "1.0",
    "v.zip(v).len()" => "3",
    "(Δ ps).zip(ps.before())[2].1.at" => "3.0",
    // lists keep their quiet cut to the shorter
    "[1, 2, 3].zip([4, 5]).len()" => "2",
);

fn compile_err(src: &str) -> String {
    match vm::Vm::compile(src, library::std) {
        Ok(_) => panic!("expected a compile error:\n{src}"),
        Err(e) => format!("{e:?}"),
    }
}

fn post_err(expr: &str) -> String {
    let e = compile_err(&format!("{POSTS}\n let _x = {expr};"));
    e.replace('│', " ").split_whitespace().collect::<Vec<_>>().join(" ")
}

// r[verify mimas.group.affine.add-posts]
// r[verify mimas.group.affine.scale]
// r[verify mimas.group.affine.errors]
#[test]
fn post_arithmetic_that_has_no_meaning_is_a_check_error() {
    let e = post_err("p + q");
    assert!(e.contains("adding two posts"), "{e}");
    assert!(e.contains("subtract them to get a panel"), "{e}");
    let e = post_err("{ let r = p; r += q; r }");
    assert!(e.contains("adding two posts"), "{e}");
    let e = post_err("ps + ps");
    assert!(e.contains("adding two posts"), "{e}");
    let e = post_err("3.0 * p");
    assert!(e.contains("scaling a post: only a panel scales"), "{e}");
    let e = post_err("p * 2.0");
    assert!(e.contains("scaling a post"), "{e}");
    let e = post_err("p / 2.0");
    assert!(e.contains("scaling a post"), "{e}");
    let e = post_err("2 * ps");
    assert!(e.contains("scaling a stream of posts"), "{e}");
    let e = post_err("-p");
    assert!(e.contains("negating a post"), "{e}");
    let e = post_err("1.0px - p");
    assert!(e.contains("subtracting a post from a panel"), "{e}");
    // a post is no group, so a `Stream` of posts is refused with the reason
    let e = post_err("stream([p, q])");
    assert!(e.contains("is a post, and posts do not add"), "{e}");
    assert!(e.contains("posts(xs)"), "{e}");
    // a page's own type named like a post is another type: the plain error
    let e = post_err("Post2 { at: 1 } + Post2 { at: 2 }");
    assert!(!e.contains("adding two posts"), "{e}");
}

// r[verify mimas.group.affine.from]
#[test]
fn a_stream_of_posts_needs_its_start() {
    let e = post_err("∫ ps");
    assert!(e.contains("`∫` adds up every step, and posts do not add"), "{e}");
    assert!(e.contains("∫ (Δ ps) from ps[0]"), "{e}");
    let e = post_err("∫ ps from post(0.0)");
    assert!(e.contains("posts do not add"), "{e}");
    let e = post_err("Δ⁻¹ ps");
    assert!(e.contains("`Δ⁻¹` adds up every step"), "{e}");
    let e = post_err("z⁻¹ ps");
    assert!(e.contains("needs the post before the first"), "{e}");
    assert!(e.contains("z⁻¹ ps from p"), "{e}");
    let e = post_err("ps.sum()");
    assert!(e.contains("summing a stream of posts"), "{e}");
    // panels where posts go: the start is what is missing
    let e = compile_err(&format!("{POSTS}\n let _x: Posts<float> = ∫ v;"));
    assert!(e.contains("name the starting post"), "{e}");
    assert!(e.contains("∫ c from p"), "{e}");
    let e = compile_err(&format!("{POSTS}\n let _x: Post<float> = 3.0;"));
    assert!(e.contains("a panel where a post goes"), "{e}");
    // `Δ` looks before no step, so it takes no start
    let e = compile_err(&format!("{POSTS}\n let _x = Δ ps from post(0.0);"));
    assert!(e.contains("takes no `from`"), "{e}");
}

// r[verify mimas.group.affine.zip]
#[test]
fn zipping_misaligned_streams_is_refused() {
    // the checker sees both sides come from `ps`
    let e = post_err("ps.zip(Δ ps)");
    assert!(e.contains("zipping two streams of different lengths"), "{e}");
    assert!(e.contains("this side has n and the other n - 1"), "{e}");
    assert!(e.contains("ps.before().zip(Δ ps)"), "{e}");
    let e = post_err("ps.zip(∇ ps)");
    assert!(e.contains("n - 1"), "{e}");
    let e = post_err("v.zip(Δ⁻¹ v)");
    assert!(e.contains("n + 1"), "{e}");
    // lengths known only at run time fault with the same advice
    let err = match test_runner::try_execute(
        "fn pair(a: Posts<float>, b: Stream<float>) -> int { a.zip(b).len() }
         let _n = pair(posts([0.0, 1.0, 3.0]), stream([1.0, 2.0]));",
    ) {
        Ok(_) => panic!("expected a misaligned zip to fault"),
        Err(e) => format!("{e}"),
    };
    assert!(err.contains("these streams have 3 and 2 steps"), "{err}");
    assert!(err.contains("ps.after().zip(∇ ps)"), "{err}");
}

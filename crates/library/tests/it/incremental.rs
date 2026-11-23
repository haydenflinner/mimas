//! Incremental operators (`library/src/incremental.mim`). Each operator runs
//! over a stream of changes and is checked two ways: against `incremental(q)`
//! from group.mim (`∇ ∘ q ∘ ∫`, the definition of `q^Δ`), and against the
//! batch query on the integrated states, differentiated. The change streams
//! come from a seeded generator, with removals, re-adds and weights above 1,
//! so items cross zero both ways.

use crate::test_runner;

const GEN: &str = r#"
    /// `n` changes over a small domain of `(key, value)` items, weights in
    /// -2..=2, from the seed `seed`.
    fn changes(seed: int, n: int) -> Stream<Tally<(str, int)>> {
        random::seed(seed);
        let keys = ["a", "b", "c"];
        let out: [Tally<(str, int)>] = [];
        for _t in n {
            let c: Tally<(str, int)> = tally([]);
            for _i in int::random(4) {
                let w = int::random(5) - 2;
                if w != 0 { c.put((keys[int::random(3)], int::random(3)), w); }
            }
            out.push(c);
        }
        stream(out)
    }
    /// Right-hand changes for joins: `(key, label)` items.
    fn labels(seed: int, n: int) -> Stream<Tally<(str, str)>> {
        random::seed(seed + 1000);
        let keys = ["a", "b", "c"];
        let names = ["x", "y"];
        let out: [Tally<(str, str)>] = [];
        for _t in n {
            let c: Tally<(str, str)> = tally([]);
            for _i in int::random(3) {
                let w = int::random(5) - 2;
                if w != 0 { c.put((keys[int::random(3)], names[int::random(2)]), w); }
            }
            out.push(c);
        }
        stream(out)
    }
    fn seeds() -> [int] { [1, 2, 3, 4, 5, 6, 7, 8] }
"#;

/// `body` (statements ending in a `bool`) run after the generators.
fn holds(body: &str) -> String {
    test_runner::render(GEN, &format!("{{ {body} }}"))
}

// r[verify mimas.incremental.integrate]
// r[verify mimas.incremental.differentiate]
// r[verify mimas.incremental.delay]
test_run!(
    integrate_differentiate_delay_step_by_step,
    "{ let s = stream([1, 2, -3, 5, 0, 4]);
       (integrator(0).run(s) == ∫ s) &&
         (differentiator(0).run(s) == ∇ s) &&
         (delayer(0).run(s) == z⁻¹ s) &&
         (differentiator(0).run(integrator(0).run(s)) == s) }" => "true",
    // the same laws on tallies, with an item that comes and goes
    "{ let s = stream([tally([\"k\"]), tally_w([(\"k\", -1), (\"j\", 2)]), tally([\"k\"])]);
       let z: Tally<str> = tally([]);
       (integrator(z).run(s) == ∫ s) && (differentiator(z).run(∫ s) == s) && (delayer(z).run(s) == z⁻¹ s) }" => "true",
    // quantities keep their dimension through the state
    "integrator(0.0px).run(stream([1.0px, 2.0px])).last() == 3.0px" => "true",
);

// r[verify mimas.incremental.linear]
// r[verify mimas.incremental.map]
// r[verify mimas.incremental.filter]
// r[verify mimas.incremental.flat_map]
#[test]
fn linear_operators_equal_their_incremental_form() {
    let body = r#"
        let ok = true;
        for seed in seeds() {
            let s = changes(seed, 12);
            let m = |t: Tally<(str, int)>| zmap(t, |p: (str, int)| p.0);
            let f = |t: Tally<(str, int)>| zfilter(t, |p: (str, int)| p.1 > 0);
            let fm = |t: Tally<(str, int)>| zflat_map(t, |p: (str, int)| [p.0, p.0 + "!"]);
            // map and flat_map change the item type, which `incremental`
            // (`(T) -> T`) cannot take: compare with ∇ of the batch query on
            // the states, which is what `incremental` computes.
            let states = (∫ s).list();
            if zmap_op(|p: (str, int)| p.0).run(s) != ∇ stream(states.map(m)) { ok = false; }
            if zfilter_op(|p: (str, int)| p.1 > 0).run(s) != incremental(f)(s) { ok = false; }
            if zflat_map_op(|p: (str, int)| [p.0, p.0 + "!"]).run(s) != ∇ stream(states.map(fm)) { ok = false; }
            // `linear(q)` on any group: a linear q is its own incremental form
            if linear(f).run(s) != incremental(f)(s) { ok = false; }
        }
        ok"#;
    assert_eq!(holds(body), "true");
}

// r[verify mimas.incremental.distinct]
#[test]
fn distinct_equals_its_incremental_form() {
    let body = r#"
        let ok = true;
        for seed in seeds() {
            let s = changes(seed, 16);
            let q = |t: Tally<(str, int)>| zdistinct(t);
            if zdistinct_op().run(s) != incremental(q)(s) { ok = false; }
        }
        ok"#;
    assert_eq!(holds(body), "true");
}

// r[verify mimas.incremental.group_sum]
#[test]
fn group_sum_equals_batch_on_the_states() {
    let body = r#"
        let ok = true;
        for seed in seeds() {
            let s = changes(seed, 16);
            let op = zgroup_sum_op(|p: (str, int)| p.0, |p: (str, int)| p.1);
            let got = op.run(s);
            let states = (∫ s).list();
            let want: [Tally<(str, int)>] = [];
            for st in states { want.push(zgroup_sum(st, |p: (str, int)| p.0, |p: (str, int)| p.1)); }
            if got != ∇ stream(want) { ok = false; }
        }
        ok"#;
    assert_eq!(holds(body), "true");
}

// r[verify mimas.incremental.join]
#[test]
fn join_equals_batch_on_the_states() {
    let body = r#"
        let ok = true;
        for seed in seeds() {
            let sa = changes(seed, 12);
            let sb = labels(seed, 12);
            let op = zjoin_op(|a: (str, int)| a.0, |b: (str, str)| b.0,
                              |a: (str, int), b: (str, str)| (a.0, a.1, b.1));
            let got = op.run(sa, sb);
            let ia = (∫ sa).list();
            let ib = (∫ sb).list();
            let want: [Tally<(str, int, str)>] = [];
            for t in 0..ia.len() {
                want.push(zjoin(ia[t], ib[t], |a: (str, int)| a.0, |b: (str, str)| b.0,
                                |a: (str, int), b: (str, str)| (a.0, a.1, b.1)));
            }
            if got != ∇ stream(want) { ok = false; }
        }
        ok"#;
    assert_eq!(holds(body), "true");
}

// A chain of operators is one circuit: changes in, changes out, and the
// composition of incremental operators is the incremental form of the
// composed query.
// r[verify mimas.incremental]
#[test]
fn a_circuit_of_operators_equals_its_batch_query() {
    let body = r#"
        let ok = true;
        for seed in seeds() {
            let sa = changes(seed, 12);
            let sb = labels(seed, 12);
            // positive values, joined with labels, then the count of rows per label
            let keep = zfilter_op(|a: (str, int)| a.1 > 0);
            let join = zjoin_op(|a: (str, int)| a.0, |b: (str, str)| b.0,
                                |a: (str, int), b: (str, str)| (b.1, a.1));
            let per = zgroup_sum_op(|r: (str, int)| r.0, |r: (str, int)| 1);
            let got: [Tally<(str, int)>] = [];
            for t in 0..sa.len() { got.push(per.step(join.step(keep.step(sa[t]), sb[t]))); }
            let ia = (∫ sa).list();
            let ib = (∫ sb).list();
            let want: [Tally<(str, int)>] = [];
            for t in 0..ia.len() {
                let kept = zfilter(ia[t], |a: (str, int)| a.1 > 0);
                let rows = zjoin(kept, ib[t], |a: (str, int)| a.0, |b: (str, str)| b.0,
                                 |a: (str, int), b: (str, str)| (b.1, a.1));
                want.push(zgroup_sum(rows, |r: (str, int)| r.0, |r: (str, int)| 1));
            }
            if stream(got) != ∇ stream(want) { ok = false; }
        }
        ok"#;
    assert_eq!(holds(body), "true");
}

// A group whose weights add to 0 keeps its sum while the sum is not 0:
// `+(a, 1)`, then `-(a, 2)` leaves weight 0 and sum -1 (no row), and
// `+(a, 2)` brings the row back as `(a, 1)`, not `(a, 2)`.
// r[verify mimas.incremental.group_sum]
test_run!(
    group_sum_keeps_a_weightless_groups_sum,
    "{ let g = zgroup_sum_op(|p: (str, int)| p.0, |p: (str, int)| p.1);
       let a = g.step(tally([(\"a\", 1)]));
       let b = g.step(tally_w([((\"a\", 2), -1)]));
       let c = g.step(tally([(\"a\", 2)]));
       (a.weight((\"a\", 1)) == 1) && (b.weight((\"a\", 1)) == -1) && (b.len() == 1) &&
         (c.weight((\"a\", 1)) == 1) && (c.len() == 1) }" => "true",
);

// An incremental build (`examples/incremental-build.mim`): items `(name,
// body, iface)` and edges `(from, to)` in, the change of the artifacts out.
// `c` uses `b`, `b` uses `a`.
const BUILD: &str = include_str!("../../../../examples/incremental-build.mim");

// r[verify mimas.incremental]
#[test]
fn incremental_build_rebuilds_what_an_edit_reaches() {
    let got = test_runner::render(BUILD, "{ let none: Tally<(str, int, int)> = tally([]);
       let no_edges: Tally<(str, str)> = tally([]);
       let b = build();
       let first = to_build(b.step(tally([(\"a\", 1, 10), (\"b\", 2, 20), (\"c\", 3, 30)]),
                                   tally([(\"b\", \"a\"), (\"c\", \"b\")])));
       // a body edit of `a` (same interface): only `a`
       let body = to_build(b.step(edit((\"a\", 1, 10), (\"a\", 4, 10)), no_edges));
       // an interface edit of `a`: `a` and its direct user `b`, not `c`
       let iface = to_build(b.step(edit((\"a\", 4, 10), (\"a\", 5, 11)), no_edges));
       // a new item that uses `a`: only the new item
       let added = to_build(b.step(tally([(\"d\", 6, 60)]), tally([(\"d\", \"a\")])));
       // `c` stops using `b`: only `c`
       let unlinked = to_build(b.step(none, tally_w([((\"c\", \"b\"), -1)])));
       // nothing changes: nothing to build
       let quiet = to_build(b.step(none, no_edges));
       [first, body, iface, added, unlinked, quiet] }");
    assert_eq!(got, "[[\"a\", \"b\", \"c\"], [\"a\"], [\"a\", \"b\"], [\"d\"], [\"c\"], []]");
}

// Transitive closure, `R = distinct(E + R ⋈ E)`, as a nested circuit: one
// body copy per inner iteration, each fed the outer-time changes.
const CLOSURE: &str = r#"
    fn tc_body() -> (Tally<(int, int)>, Tally<(int, int)>) -> Tally<(int, int)> {
        let j = zjoin_op(|r: (int, int)| r.1, |e: (int, int)| e.0,
                         |r: (int, int), e: (int, int)| (r.0, e.1));
        let d = zdistinct_op();
        |de: Tally<(int, int)>, dr: Tally<(int, int)>| d.step(de + j.step(dr, de))
    }
    fn tc_batch(e: Tally<(int, int)>) -> Tally<(int, int)> {
        zfixpoint(e, |e: Tally<(int, int)>, r: Tally<(int, int)>|
            zdistinct(e + zjoin(r, e, |r: (int, int)| r.1, |x: (int, int)| x.0,
                                |r: (int, int), x: (int, int)| (r.0, x.1))))
    }
    /// `n` changes of a set of edges over `k` nodes: each change toggles up
    /// to three edges, adding one that is absent and taking away one that
    /// is there, so edges come, go, come back and form cycles. Closure is
    /// defined on a set, so the integrated weights stay 0 or 1.
    fn edges(seed: int, n: int, k: int) -> Stream<Tally<(int, int)>> {
        random::seed(seed);
        let held: Tally<(int, int)> = tally([]);
        let out: [Tally<(int, int)>] = [];
        for _t in n {
            let c: Tally<(int, int)> = tally([]);
            for _i in int::random(4) {
                let e = (int::random(k), int::random(k));
                if c.weight(e) != 0 { continue; }
                let w = if held.weight(e) > 0 { -1 } else { 1 };
                c.put(e, w);
            }
            held.put_all(c);
            out.push(c);
        }
        stream(out)
    }
"#;

// r[verify mimas.incremental.fixpoint]
#[test]
fn transitive_closure_equals_the_batch_fixpoint() {
    let body = r#"
        let ok = true;
        for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
            let s = edges(seed, 14, 5);
            let got = zfixpoint_op(|| tc_body()).run(s);
            let want: [Tally<(int, int)>] = [];
            for st in (∫ s).list() { want.push(tc_batch(st)); }
            if got != ∇ stream(want) { ok = false; }
        }
        ok"#;
    let got = test_runner::render(CLOSURE, &format!("{{ {body} }}"));
    assert_eq!(got, "true");
}

// A chain that grows one edge a step makes every step need one inner
// iteration more than any before: each new body copy must start with the
// history the deepest copy saw, or the far ends of the chain go missing.
// r[verify mimas.incremental.fixpoint]
#[test]
fn a_growing_chain_needs_deeper_copies_primed_with_history() {
    let body = r#"
        let tc = zfixpoint_op(|| tc_body());
        let ok = true;
        let all: Tally<(int, int)> = tally([]);
        for n in 1..8 {
            let c = tally([(n, n + 1)]);
            all.put_all(c);
            let got = tc.step(c);
            // the new edge reaches back to every earlier node: n new pairs
            if got.len() != n { ok = false; }
            if got.weight((1, n + 1)) != 1 { ok = false; }
        }
        // and taking the first edge away removes every pair that started at 1
        let gone = tc.step(tally_w([((1, 2), -1)]));
        if gone.len() != 7 || gone.weight((1, 8)) != -1 { ok = false; }
        ok"#;
    assert_eq!(test_runner::render(CLOSURE, &format!("{{ {body} }}")), "true");
}

// The build without interfaces (`DeepBuild`): an artifact depends on every
// body it reaches, through the recursive closure of the edges.
// r[verify mimas.incremental.fixpoint]
#[test]
fn deep_build_rebuilds_everything_that_reaches_an_edit() {
    let got = test_runner::render(BUILD, "{ let none: Tally<(str, int)> = tally([]);
       let no_edges: Tally<(str, str)> = tally([]);
       let b = deep_build();
       let first = deep_to_build(b.step(tally([(\"a\", 1), (\"b\", 2), (\"c\", 3)]),
                                        tally([(\"b\", \"a\"), (\"c\", \"b\")])));
       // a body edit of `a`: `a`, and `b` and `c`, which reach it
       let leaf = deep_to_build(b.step(tally_w([((\"a\", 1), -1), ((\"a\", 4), 1)]), no_edges));
       // a body edit of `c`: only `c`, nothing reaches it
       let root = deep_to_build(b.step(tally_w([((\"c\", 3), -1), ((\"c\", 5), 1)]), no_edges));
       // `b` stops using `a`: `b`, and `c` through `b`
       let cut = deep_to_build(b.step(none, tally_w([((\"b\", \"a\"), -1)])));
       // now an edit of `a` reaches nobody else
       let alone = deep_to_build(b.step(tally_w([((\"a\", 4), -1), ((\"a\", 6), 1)]), no_edges));
       [first, leaf, root, cut, alone] }");
    assert_eq!(got, "[[\"a\", \"b\", \"c\"], [\"a\", \"b\", \"c\"], [\"c\"], [\"b\", \"c\"], [\"a\"]]");
}

// `t.put_all(o)` adds `o`'s weights into `t` in place: `t + o`, kept in `t`,
// an item whose weight comes to 0 dropped.
// r[verify mimas.group.tally]
test_run!(
    tally_put_all_adds_in_place,
    "{ let t = tally([\"a\", \"b\"]);
       t.put_all(tally_w([(\"a\", 2), (\"b\", -1), (\"c\", 1)]));
       (t == tally_w([(\"a\", 3), (\"c\", 1)])) && (t.len() == 2) }" => "true",
);

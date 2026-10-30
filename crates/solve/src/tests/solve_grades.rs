//! Grades-pass coverage: the usage-count lints (unused binding, unused param, dead store)
//! and the `Fx` inference that backs safe page eval.
//!
//! Effect fakes register through `Solver::declare_native_fn` -- the same entry point real
//! libraries drive via `install_library`, so `#[effects]` declarations and test stubs land
//! identically in `NativeBinding::sig`.

use super::solve_test_utils::*;
use crate::components::{DecId, Ty};
use api::NativeId;
use shared::Fx;

/// Register a zero-arg native named `name`, with `fx` as its `#[effects]` declaration
/// (`None` = unannotated, which inference must read as `Fx::unknown()`).
fn native(name: &str, fx: Option<Fx>) {
    native_opts(name, 0, &[], false, fx);
}

/// `arity` untyped params, `consumes[i]` marks slot `i` `#[consumes]`, `must_use`
/// flags the return. What `#[consumes]`/`#[must_use]` land as post-install.
fn native_opts(name: &str, arity: usize, consumes: &[usize], must_use: bool, fx: Option<Fx>) {
    TEST_SESSION.with(|s| {
        let mut slots = vec![false; arity];
        for &i in consumes {
            slots[i] = true;
        }
        s.borrow_mut().0.declare_native_fn(
            name.to_string(),
            vec![None; arity],
            vec![None; arity],
            Some(Ty::Unit),
            None,
            NativeId::from(999),
            None,
            None,
            String::new(),
            fx,
            slots,
            must_use,
        );
    });
}

fn run(src: &str) {
    TEST_SESSION.with(|s| s.borrow_mut().run(src, "test").unwrap());
}

/// The solve's lint output, as the diagnostic's own message line.
fn warning_texts() -> Vec<String> {
    TEST_SESSION.with(|s| {
        s.borrow()
            .0
            .warnings
            .iter()
            .map(|w| w.to_string())
            .collect()
    })
}

fn warns(src: &str) -> Vec<String> {
    run(src);
    warning_texts()
}

/// The `DecId` of the dec named `name` (any kind). `fn` decs are what `fn_effects` keys on.
fn dec_named(name: &str) -> DecId {
    TEST_SESSION.with(|s| {
        s.borrow()
            .0
            .decs
            .iter()
            .find(|(_, d)| d.name == name)
            .map_or_else(|| panic!("no dec named {name}"), |(id, _)| id)
    })
}

fn fn_fx(name: &str) -> Fx {
    let dec = dec_named(name);
    TEST_SESSION.with(|s| {
        s.borrow()
            .0
            .fn_effects
            .get(&dec)
            .copied()
            .unwrap_or_else(|| panic!("no inferred effects for {name}"))
    })
}

fn script_fx() -> Fx {
    TEST_SESSION.with(|s| {
        s.borrow()
            .0
            .script_effects
            .get("test")
            .copied()
            .unwrap_or_else(|| panic!("no script effects recorded"))
    })
}

fn dec_reads(name: &str) -> crate::Use {
    let dec = dec_named(name);
    TEST_SESSION.with(|s| s.borrow().0.dec_uses.get(&dec).map(|u| u.reads).unwrap())
}

// ---- usage lints

#[test]
fn unused_let_binding_warns() {
    let _t = TestResetter;
    let w = warns("fn f() -> int { let x = 1; 2 }");
    assert_eq!(w, ["unused variable `x`"]);
}

#[test]
fn underscore_binding_is_quiet() {
    let _t = TestResetter;
    let w = warns("fn f() -> int { let _x = 1; 2 }");
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn read_binding_is_quiet() {
    let _t = TestResetter;
    let w = warns("fn f() -> int { let x = 1; x + 1 }");
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn unused_param_warns() {
    let _t = TestResetter;
    let w = warns("fn f(x, y) { y }");
    assert_eq!(w, ["unused parameter `x`"]);
}

#[test]
fn unused_closure_param_warns() {
    let _t = TestResetter;
    let w = warns("fn f() -> int { let g = |x| 1; g(2) }");
    assert_eq!(w, ["unused parameter `x`"]);
}

#[test]
fn self_param_reads_count() {
    let _t = TestResetter;
    // `self` is the method's declared parameter -- reading it must count.
    // The identity resolve used to return no dec, so every impl method
    // warned `unused parameter self` no matter what the body did.
    let w = warns("struct W { x: float } impl W { fn f(self) -> float { self.x } }");
    assert!(w.is_empty(), "{w:?}");
    let w = warns(
        "pact P { fn f(self) -> float; } struct W { x: float } impl P for W { fn f(self) -> float { self.x } }",
    );
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn unused_self_param_still_warns() {
    let _t = TestResetter;
    let w = warns("struct W { x: float } impl W { fn f(self) -> float { 1.0 } }");
    assert_eq!(w, ["unused parameter `self`"]);
}

#[test]
fn dead_store_warns_on_overwrite() {
    let _t = TestResetter;
    let w = warns("fn f() -> int { let x = 1; x = 2; x }");
    assert_eq!(w, ["this store to `x` is never read"]);
}

#[test]
fn read_write_is_not_dead() {
    let _t = TestResetter;
    // `x = x + 1` reads the old value -- nothing is dead
    let w = warns("fn f() -> int { let x = 1; x = x + 1; x }");
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn init_dead_when_both_arms_overwrite() {
    let _t = TestResetter;
    // `x = 1` is dead on both paths -- each arm overwrites before any read. The arms'
    // writes survive the merge to the `x` read, so only the `let` reports.
    let w = warns("fn f(c: bool) -> int { let x = 1; if c { x = 2; } else { x = 3; }; x }");
    assert_eq!(w, ["this store to `x` is never read"]);
}

#[test]
fn branch_writes_survive_merge() {
    let _t = TestResetter;
    // on the else path `x`'s value is the `let` -- the union merge keeps all three writes
    // alive until the read, so nothing is dead
    let w = warns("fn f(c: bool) -> int { let x = 1; if c { x = 2; }; x }");
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn store_dead_on_every_path_warns() {
    let _t = TestResetter;
    // the `let`, both arm writes, all dead once `x = 4` lands -- reported once each
    let w = warns("fn f(c: bool) -> int { let x = 1; if c { x = 2; } else { x = 3; }; x = 4; x }");
    assert_eq!(w.len(), 3, "{w:?}");
}

#[test]
fn write_before_return_is_dead() {
    let _t = TestResetter;
    let w = warns("fn f() -> int { let x = 1; x = 2; return x }");
    assert_eq!(w, ["this store to `x` is never read"]);
}

#[test]
fn unused_loop_var_warns() {
    let _t = TestResetter;
    let w = warns("fn f() { for x in [1, 2] { 0; } }");
    assert_eq!(w, ["unused variable `x`"]);
}

#[test]
fn unused_binding_unread_at_scope_end() {
    let _t = TestResetter;
    // x is never read: one `unused` warning, not a dead-store pile
    let w = warns("fn f() -> int { let x = 1; x = 2; 0 }");
    assert_eq!(w, ["unused variable `x`"]);
}

#[test]
fn site_counts_are_recorded() {
    let _t = TestResetter;
    run("fn f() -> int { let x = 1; let y = x; let z = x + y; z }");
    assert_eq!(dec_reads("x"), crate::Use::Many);
    assert_eq!(dec_reads("y"), crate::Use::Once);
    assert_eq!(dec_reads("z"), crate::Use::Once);
}

// ---- effect inference

#[test]
fn pure_fn_has_no_effects() {
    let _t = TestResetter;
    run("fn f() -> int { 1 }");
    assert_eq!(fn_fx("f"), Fx::empty());
}

#[test]
fn native_declared_effects_reach_caller() {
    let _t = TestResetter;
    native("net_get", Some(Fx::NET));
    run("fn f() { net_get() }");
    assert_eq!(fn_fx("f"), Fx::NET);
}

#[test]
fn effects_flow_through_user_calls() {
    let _t = TestResetter;
    native("net_get", Some(Fx::NET));
    native("print_it", Some(Fx::IO));
    run("fn inner() { net_get() } fn mid() { inner(); print_it(); } fn f() { mid() }");
    assert_eq!(fn_fx("f"), Fx::NET | Fx::IO);
}

#[test]
fn unannotated_native_is_unaudited() {
    let _t = TestResetter;
    native("mystery", None);
    run("fn f() { mystery() }");
    assert!(fn_fx("f").contains(Fx::UNAUDITED));
    assert!(fn_fx("f").contains(Fx::ANY));
}

#[test]
fn explicit_pure_annotation_stays_pure() {
    let _t = TestResetter;
    native("tally", Some(Fx::empty()));
    run("fn f() { tally() }");
    assert_eq!(fn_fx("f"), Fx::empty());
}

#[test]
fn recursion_settles() {
    let _t = TestResetter;
    native("net_get", Some(Fx::NET));
    run("fn f(n) { if n > 0 { f(n - 1) } else { net_get() } }");
    assert_eq!(fn_fx("f"), Fx::NET);
}

#[test]
fn script_level_effects_are_recorded() {
    let _t = TestResetter;
    native("net_get", Some(Fx::NET));
    run("let _x = net_get();");
    assert_eq!(script_fx(), Fx::NET);
}

#[test]
fn script_picks_up_fn_effects() {
    let _t = TestResetter;
    native("net_get", Some(Fx::NET));
    run("fn f() { net_get() } let _x = f();");
    assert_eq!(script_fx(), Fx::NET);
}

#[test]
fn closure_call_site_is_unknown() {
    let _t = TestResetter;
    // `g` is a fn-typed binding -- the call's effects can't be graded
    run("fn f(g) { g() }");
    assert!(fn_fx("f").contains(Fx::UNAUDITED));
}

#[test]
fn warnings_are_not_errors() {
    let _t = TestResetter;
    // lints never fail a solve -- the code still runs
    TEST_SESSION.with(|s| {
        s.borrow_mut()
            .run("fn f() -> int { let x = 1; x = 2; x }", "test")
            .unwrap();
    });
}

// ---- consuming + must_use (Stage 1)

/// The solve error a source produces, if any.
fn err(src: &str) -> Option<String> {
    TEST_SESSION.with(|s| s.borrow_mut().run(src, "test").err().map(|e| e.to_string()))
}

#[test]
fn use_after_consume_errors() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    let e = err("fn f() -> int { let ch = 1; close(ch); ch }").unwrap();
    assert!(e.contains("`ch` was consumed by `close`"), "{e}");
}

#[test]
fn consume_counts_as_a_use() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    // passing `ch` to `close` reads it -- no unused-binding lint
    let w = warns("fn f() { let ch = 1; close(ch); }");
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn restoring_consumed_binding_reborns_it() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    run("fn f() -> int { let ch = 1; close(ch); ch = 2; ch }");
}

#[test]
fn consume_on_one_branch_poisons_the_join() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    let e = err("fn f(c: bool) -> int { let ch = 1; if c { close(ch); }; ch }").unwrap();
    assert!(e.contains("consumed"), "{e}");
}

#[test]
fn consume_branch_with_rebirth_is_fine() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    // consumed only inside the arm, re-stored before the join -- `ch` is live again
    run("fn f(c: bool) -> int { let ch = 1; if c { close(ch); ch = 2; }; ch }");
}

#[test]
fn consuming_a_temporary_is_free() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    native_opts("mk", 0, &[], false, Some(Fx::empty()));
    run("fn f() { close(mk()); }");
}

/// Struct decl used by the field-path tests. A free fn builds values so each
/// test source can stay on one line after `PDECL`.
const PDECL: &str =
    "struct P { id: int, other: int, list: [int] } fn pctor() -> P { P { id = 1, other = 2, list = [0] } } ";

#[test]
fn consuming_an_indexed_element_is_refused() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    // `a[i]` can't name which element died -- bind it (`let x = a[i]`) first
    let e = err("fn f() { let arr = [1, 2]; close(arr[0]); }").unwrap();
    assert!(e.contains("can't be consumed"), "{e}");
    let e = err(&format!("{PDECL} fn f() {{ let p = pctor(); close(p.list[0]); }}"))
        .unwrap();
    assert!(e.contains("can't be consumed"), "{e}");
}

#[test]
fn consuming_a_field_kills_just_that_field() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    // `r.id` is dead but its sibling `r.other` stays live
    run(&format!(
        "{PDECL} fn f() -> int {{ let r = pctor(); close(r.id); r.other }}"
    ));
}

#[test]
fn reading_a_consumed_field_errors() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    let e = err(&format!(
        "{PDECL} fn f() -> int {{ let r = pctor(); close(r.id); r.id }}"
    ))
    .unwrap();
    assert!(e.contains("`r.id` was consumed by `close`"), "{e}");
}

#[test]
fn reading_the_parent_after_field_consume_errors() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    native_opts("touch", 1, &[], false, Some(Fx::empty()));
    // whole `r` reads every member -- dead `r.id` included
    let e =
        err(&format!("{PDECL} fn f() {{ let r = pctor(); close(r.id); touch(r); }}"))
            .unwrap();
    assert!(e.contains("consumed by `close`"), "{e}");
}

#[test]
fn restoring_a_consumed_field_reborns_it() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    run(&format!(
        "{PDECL} fn f() -> int {{ let r = pctor(); close(r.id); r.id = 9; r.id }}"
    ));
}

#[test]
fn field_consume_on_one_branch_poisons_the_join() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    let e = err(&format!(
        "{PDECL} fn f(c: bool) -> int {{ let r = pctor(); if c {{ close(r.id); }}; r.id }}"
    ))
    .unwrap();
    assert!(e.contains("consumed"), "{e}");
    // ...but the untouched sibling stays live through the same join
    run(&format!(
        "{PDECL} fn f(c: bool) -> int {{ let r = pctor(); if c {{ close(r.id); }}; r.other }}"
    ));
}

#[test]
fn field_consume_branch_rebirth_is_fine() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    run(&format!(
        "{PDECL} fn f(c: bool) -> int {{ let r = pctor(); if c {{ close(r.id); r.id = 3; }}; r.id }}"
    ));
}

#[test]
fn consuming_a_loopvar_field_is_fine() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    // each iteration binds `r` fresh -- its fields can die there (the asteroids
    // `squish::drop(r.id)` shape)
    run(&format!("{PDECL} fn f() {{ for r in [pctor()] {{ close(r.id); }} }}"));
}

#[test]
fn consuming_an_outer_dec_field_inside_a_loop_is_refused() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    let e = err(&format!(
        "{PDECL} fn f() {{ let r = pctor(); for x in [1, 2] {{ close(r.id); }} }}"
    ))
    .unwrap();
    assert!(e.contains("can't be consumed"), "{e}");
}

#[test]
fn double_consuming_a_field_errors() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    let e = err(&format!(
        "{PDECL} fn f() {{ let r = pctor(); close(r.id); close(r.id); }}"
    ))
    .unwrap();
    assert!(e.contains("consumed"), "{e}");
}

#[test]
fn consuming_a_global_is_refused() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    let e = err("let ch = 1; close(ch);").unwrap();
    assert!(e.contains("can't be consumed"), "{e}");
}

#[test]
fn consuming_outer_dec_inside_a_loop_is_refused() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    let e = err("fn f() { let ch = 1; for x in [1, 2] { close(ch); } }").unwrap();
    assert!(e.contains("can't be consumed"), "{e}");
}

#[test]
fn consuming_a_loopvar_is_fine() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    // each iteration binds `x` fresh -- it can die there
    run("fn f() { for x in [1, 2] { close(x); } }");
}

#[test]
fn consuming_a_capture_is_refused() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    let e = err("fn f() { let ch = 1; let g = || { close(ch); 0 }; g(); }").unwrap();
    assert!(e.contains("can't be consumed"), "{e}");
}

#[test]
fn consuming_a_dec_inside_its_own_loop_scope_is_fine() {
    let _t = TestResetter;
    native_opts("close", 1, &[0], false, Some(Fx::empty()));
    run("fn f() { for x in [1, 2] { let ch = x; close(ch); } }");
}

#[test]
fn must_use_result_discard_warns() {
    let _t = TestResetter;
    native_opts("reply", 0, &[], true, Some(Fx::empty()));
    let w = warns("fn f() { reply(); }");
    assert_eq!(
        w,
        ["the result of `reply` is `must_use` and can't be discarded"]
    );
}

#[test]
fn must_use_result_bound_is_quiet() {
    let _t = TestResetter;
    native_opts("reply", 0, &[], true, Some(Fx::empty()));
    let w = warns("fn f() { let _x = reply(); }");
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn non_must_use_discard_is_quiet() {
    let _t = TestResetter;
    native("reply", Some(Fx::empty()));
    let w = warns("fn f() { reply(); }");
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn while_counter_increment_is_live() {
    let _t = TestResetter;
    // `i += 1` at the loop bottom feeds `i < n` on the next pass -- the
    // back-edge keeps the store live
    let w = warns("fn f(n: int) -> int { let i = 0; while i < n { i += 1; } i }");
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn for_carried_prev_is_live() {
    let _t = TestResetter;
    let w = warns(
        "fn f(a: [int]) -> int {
            let prev = 0;
            for v in a { if prev == v { prev = v + 1; } }
            prev
        }",
    );
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn nested_loop_counter_is_live() {
    let _t = TestResetter;
    let w = warns(
        "fn f(n: int) -> int {
            let i = 0;
            while i < n {
                let j = 0;
                while j < n { j += 1; }
                i += 1;
            }
            i
        }",
    );
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn truly_dead_store_inside_loop_still_warns() {
    let _t = TestResetter;
    // `x` is never read anywhere -- the write is dead for real
    let w = warns("fn f(n: int) { let x = 0; while x < n { x += 1; } x = 0; }");
    // x is read (in `x < n`) so unused-binding doesn't own it; the `x = 0`
    // overwrite past the loop is dead
    assert!(
        w.iter().any(|m| m.contains("never read")),
        "{w:?}"
    );
}

#[test]
fn continue_path_store_is_live_when_read_next_pass() {
    let _t = TestResetter;
    let w = warns(
        "fn f(a: [int]) -> int {
            let acc = 0;
            for v in a {
                acc += v;
                if v < 0 { acc += 1; continue; }
            }
            acc
        }",
    );
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn break_path_store_reaches_exit_read() {
    let _t = TestResetter;
    // `acc += 100` on the break path is read by the `acc` after the loop --
    // exit flows carry evidence just like fall-through ones
    let w = warns(
        "fn f(a: [int]) -> int {
            let acc = 0;
            for v in a {
                acc += v;
                if v < 0 { acc += 100; break; }
            }
            acc
        }",
    );
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn let_unread_on_one_continue_path_stays_live() {
    let _t = TestResetter;
    // `c` is unread on the `continue` path but read on the others -- scope
    // exit must consult the break/continue flows before calling a store dead
    let w = warns(
        "fn f(spells: [int], cards: [int]) -> int {
            let acc = 0;
            for s in spells {
                let c = cards[s];
                if s > 0 { acc += s; continue; }
                acc += c;
            }
            acc
        }",
    );
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn merge_inherited_pending_store_stays_live() {
    let _t = TestResetter;
    // `over = 2` kills the `over = 1` span on this path, but that span is
    // still pending on the `by <= 10.0` sibling -- a merge must not rebrand
    // inherited pending writes as segment-local and flag them eagerly
    let w = warns(
        "fn f(lives: int, score: int, by: float) -> int {
            let over = 0;
            if score >= 400 { over = 1; }
            if by > 10.0 {
                if lives > 1 { return 7; }
                over = 2;
            }
            over
        }",
    );
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn else_if_arm_write_then_late_write_stays_live() {
    let _t = TestResetter;
    // the asteroids shape: read in an else-if arm, writes in two branches,
    // read at the tail
    let w = warns(
        "fn f(ship: int, reborn0: float, dt: float) -> float {
            let reborn = reborn0;
            if ship > 0 { } else if reborn > 0.0 { reborn -= dt; }
            if ship < 0 { reborn = 1.4; }
            reborn
        }",
    );
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn loop_carried_store_read_after_loop() {
    let _t = TestResetter;
    // ui-kit's `hit`: written inside the loop on every path that reaches it,
    // read after the loop
    let w = warns(
        "fn f(ws: [int]) -> int {
            let hit = -1;
            for s in ws {
                if s < 0 { continue; }
                hit = s;
            }
            hit
        }",
    );
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn break_path_write_inside_loop_is_live() {
    let _t = TestResetter;
    // meridian's `leaf`/`lv`: written right before `break`, read after the
    // loop -- the exit flow proves the store live
    let w = warns(
        "fn f(a: [int]) -> int {
            let leaf = 0;
            let lv = 0.5;
            loop {
                let tn = a[0];
                if tn > 0 { leaf = tn; lv = 1.0; break; }
                if tn < 0 { break; }
            }
            if lv > 0.0 { leaf } else { -leaf }
        }",
    );
    assert!(w.is_empty(), "{w:?}");
}

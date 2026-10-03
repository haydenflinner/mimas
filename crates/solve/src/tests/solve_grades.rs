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
    TEST_SESSION.with(|s| {
        s.borrow_mut().0.declare_native_fn(
            name.to_string(),
            vec![],
            vec![],
            Some(Ty::Unit),
            None,
            NativeId::from(999),
            None,
            None,
            fx,
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

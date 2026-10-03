//! `#[effects]` declarations on real library natives, end to end: the `#[native]` macro
//! submits a `NativeEffects` record to inventory, `install` joins it onto the native's
//! `sig.effects`, and the grades pass's inference lifts it into
//! `Resolutions::{fn_effects, script_effects}` -- the sets a host gates peer code on.

use shared::Fx;
use solve::{Resolutions, load_files};

fn solve(src: &str) -> Resolutions {
    let mut vm = vm::Vm::new();
    let library = vm.install_library(library::std);
    let loaded = load_files([("test", src)], &library);
    assert!(loaded.errors.is_empty(), "{:?}", loaded.errors);
    Resolutions::from(loaded.solver)
}

#[test]
fn print_is_io() {
    assert_eq!(solve("print(1);").script_effects["test"], Fx::IO);
}

#[test]
fn random_draws_are_rng() {
    let r = solve("let _x = int::random(3);");
    assert_eq!(r.script_effects["test"], Fx::RNG);
}

#[test]
fn shuffle_and_seed_are_rng() {
    let r = solve("let xs = [1, 2]; xs.shuffle(); random::seed(4);");
    assert_eq!(r.script_effects["test"], Fx::RNG);
}

#[test]
fn effects_accumulate_through_user_calls() {
    let r = solve("fn deal() -> int { int::random(52) } fn f() { print(deal()); } let _x = f();");
    assert_eq!(r.script_effects["test"], Fx::IO | Fx::RNG);
}

#[test]
fn unannotated_native_fails_closed() {
    // `arr.len` declares no `#[effects]` -- its callers can't be graded
    let r = solve("let _x = [1].len();");
    assert!(r.script_effects["test"].contains(Fx::UNAUDITED));
}

#[test]
fn declared_pure_native_stays_pure() {
    // `sys::file` is `#[effects()]` -- an explicit empty declaration, not unaudited
    let r = solve("use std::sys; let _x = sys::file();");
    assert_eq!(r.script_effects["test"], Fx::empty());
}

#[test]
fn fs_reads_are_io() {
    let r = solve("use std::fs; let _x = fs::read(\"x\");");
    assert_eq!(r.script_effects["test"], Fx::IO);
}

#[test]
fn pure_script_has_no_effects() {
    assert_eq!(solve("let _x = 1 + 2;").script_effects["test"], Fx::empty());
}

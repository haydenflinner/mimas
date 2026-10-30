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

// ---- the gate: `Vm::compile_files_gated` refuses a file whose inferred
// effects exceed the grant, before codegen; `Vm::audit_files` reports the
// same footprint without building a program.

#[test]
fn gate_admits_fitting_code() {
    let files = [("main", "print(1);")];
    assert!(vm::Vm::compile_files_gated(&files, library::std, Some(Fx::IO)).is_ok());
}

#[test]
fn gate_refuses_io_under_pure_grant() {
    let files = [("main", "print(1);")];
    let err = vm::Vm::compile_files_gated(&files, library::std, Some(Fx::empty()))
        .err()
        .expect("io page must not build under an empty grant");
    let msg = err.to_string();
    assert!(msg.contains("main"), "{msg}");
    assert!(msg.contains("io"), "{msg}");
}

#[test]
fn gate_is_per_file() {
    // the peer helper wants rng; own page stays pure. A `doc`-only grant
    // rejects on `helper`'s name even though the offending call is there.
    let files = [
        ("own", "let _x = helper();"),
        ("helper", "fn helper() -> int { int::random(3) }"),
    ];
    let err = vm::Vm::compile_files_gated(&files, library::std, Some(Fx::DOC))
        .err()
        .expect("helper's rng must not fit a doc grant");
    let msg = err.to_string();
    assert!(
        msg.contains("own") || msg.contains("helper"),
        "{msg}"
    );
}

#[test]
fn unannotated_native_needs_unaudited_grant() {
    let files = [("main", "let _x = [1].len();")];
    // `ANY` does not carry the flag: ungraded code is refused even under a
    // maximally permissive *audited* grant -- opting in is explicit.
    assert!(
        vm::Vm::compile_files_gated(&files, library::std, Some(Fx::ANY & !Fx::UNAUDITED)).is_err()
    );
    assert!(vm::Vm::compile_files_gated(&files, library::std, Some(Fx::ANY | Fx::UNAUDITED)).is_ok());
}

#[test]
fn audit_reports_each_files_fx() {
    let files = [
        ("own", "let _x = helper();"),
        ("helper", "fn helper() -> int { int::random(3) }"),
    ];
    let audit = vm::Vm::audit_files(&files, library::std).unwrap();
    assert_eq!(audit.script_effects["helper"], Fx::empty());
    assert!(audit.script_effects["own"].contains(Fx::RNG));
}

#[test]
fn audit_surfaces_lint_warnings() {
    // script-top-level bindings are globals (always ω, never linted) -- the
    // unused-binding lint only fires inside a fn body
    let files = [("main", "fn f() -> int { let unused = 4; 1 } let _x = f();")];
    let audit = vm::Vm::audit_files(&files, library::std).unwrap();
    assert!(
        audit
            .warnings
            .iter()
            .any(|w| w.to_string().contains("unused")),
        "{:?}",
        audit.warnings
    );
}

//! The host can cap how many ops an entry runs: a loop with no end is an error, not a hang.
use vm::Vm;

#[test]
fn endless_loop_hits_the_op_budget() {
    let mut vm = Vm::compile("let i = 0;\nwhile true { i += 1; }", |_| {}).unwrap();
    vm.set_op_budget(50_000);
    let err = vm.run().unwrap_err().to_string();
    assert!(err.contains("ran too long"), "{err}");
}

#[test]
fn a_finite_program_fits_the_budget() {
    let mut vm = Vm::compile("let i = 0;\nwhile i < 100 { i += 1; }", |_| {}).unwrap();
    vm.set_op_budget(1_000_000);
    vm.run().unwrap();
}

#[test]
fn budget_covers_calls_into_functions() {
    let mut vm = Vm::compile("fn spin() { while true { } }\nfn f() -> int { spin(); 1 }", |_| {}).unwrap();
    vm.set_op_budget(20_000);
    let err = vm.call_fn_result("f").unwrap_err().to_string();
    assert!(err.contains("ran too long"), "{err}");
}

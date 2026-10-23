//! The memory budget with the std library installed (`push` is a library intrinsic). The
//! budget's own semantics are tested in `mimas-vm`'s `tests/budget.rs`.
use vm::Vm;

const MB: usize = 1 << 20;

fn compile(src: &str) -> Vm {
    Vm::compile(src, |api| library::std(api)).unwrap()
}

#[test]
fn repeat_checks_its_size_before_allocating() {
    let mut vm = compile("let s = \"ab\".repeat(1099511627776);");
    vm.set_mem_budget(64 * MB);
    let err = vm.run().unwrap_err().to_string();
    assert!(err.contains("bigger than 16 MB"), "{err}");
}

#[test]
fn new_filled_checks_its_size_before_allocating() {
    let mut vm = compile("let a = array::new_filled(0, 1099511627776);");
    vm.set_mem_budget(64 * MB);
    let err = vm.run().unwrap_err().to_string();
    assert!(err.contains("bigger than 16 MB"), "{err}");
}

#[test]
fn retained_arrays_hit_the_memory_budget() {
    let mut vm = compile(
        "let keep = [];\nlet i = 0;\nwhile true { keep.push([i, i, i, i, i, i, i, i]); i += 1; }",
    );
    vm.set_op_budget(100_000_000);
    vm.set_mem_budget(8 * MB);
    let err = vm.run().unwrap_err().to_string();
    assert!(err.contains("used more than 8 MB of memory"), "{err}");
}


#[test]
fn settle_measures_what_is_live() {
    let mut vm = compile(
        "let keep = [];\nlet i = 0;\nwhile i < 250000 { keep.push(i); i += 1; }",
    );
    vm.run().unwrap();
    // 250k ints in a typed store: ~2 MB of buffer (capacity rounds up)
    let live = vm.mem_settle();
    assert!(live >= 2 * MB - MB / 8 && live < 8 * MB, "{live}");
}


#[test]
fn a_program_over_budget_stays_over() {
    let mut vm = compile(
        "let keep = [];\nlet i = 0;\nwhile true { keep.push([i, i, i, i]); i += 1; }\nfn f() -> int { 1 }",
    );
    vm.set_op_budget(100_000_000);
    vm.set_mem_budget(4 * MB);
    assert!(vm.run().is_err());
    // the heap still holds the array: the next entry fails at once
    vm.set_op_budget(1_000);
    let err = vm.call_fn_result("f").unwrap_err().to_string();
    assert!(err.contains("MB of memory"), "{err}");
}


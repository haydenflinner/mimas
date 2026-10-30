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
    let mut vm = Vm::compile(
        "fn spin() { while true { } }\nfn f() -> int { spin(); 1 }",
        |_| {},
    )
    .unwrap();
    vm.set_op_budget(20_000);
    let err = vm.call_fn_result("f").unwrap_err().to_string();
    assert!(err.contains("ran too long"), "{err}");
}

// ----- memory budget (see `vm::mem`) -----

const MB: usize = 1 << 20;

#[test]
fn doubling_a_string_hits_the_memory_budget() {
    // `s = s + s` thirty times is ~1 GB in a handful of ops — the op budget can't see it
    let mut vm = Vm::compile(
        "let s = \"xxxxxxxx\";\nlet i = 0;\nwhile i < 30 { s = s + s; i += 1; }",
        |_| {},
    )
    .unwrap();
    vm.set_op_budget(1_000_000);
    vm.set_mem_budget(64 * MB);
    let err = vm.run().unwrap_err().to_string();
    assert!(err.contains("MB"), "{err}");
    assert!(!err.contains("ran too long"), "{err}");
}

#[test]
fn garbage_is_not_counted_once_collected() {
    // ~25 MB allocated in all, but almost none of it live at once: a batched run collects
    // between batches and measures, so it stays under a 4 MB budget
    let mut vm = Vm::compile(
        "let i = 0;\nwhile i < 200000 { let t = [i, i, i, i, i, i, i, i, i, i, i, i]; i += 1; }",
        |_| {},
    )
    .unwrap();
    vm.set_op_budget(100_000_000);
    vm.set_mem_budget(4 * MB);
    vm.run().unwrap();
    assert!(vm.mem_estimate() < 4 * MB, "{}", vm.mem_estimate());
}

#[test]
fn garbage_inside_a_host_call_is_collected_too() {
    // `call_fn_result` runs in batches with collections between them, like `run`
    let mut vm = Vm::compile(
        "fn churn() -> int {\n  let i = 0;\n  while i < 200000 { let t = [i, i, i, i, i, i, i, i, i, i, i, i]; i += 1; }\n  i\n}",
        |_| {},
    )
    .unwrap();
    vm.run().unwrap();
    vm.set_op_budget(100_000_000);
    vm.set_mem_budget(4 * MB);
    vm.call_fn_result("churn").unwrap();
}

#[test]
fn dead_strings_leave_the_intern_table() {
    // every intermediate string used to stay in memory through the intern table's own `Arc`
    let mut vm = Vm::compile(
        "let s = \"\";\nlet i = 0;\nwhile i < 2000 { s = s + \"abcdefgh\"; i += 1; }",
        |_| {},
    )
    .unwrap();
    vm.run().unwrap();
    // the 2000 intermediates total ~16 MB; only the last (16 KB) is live
    let live = vm.mem_settle();
    assert!(live < MB, "{live}");
}

// ----- call depth -----

#[test]
fn deep_recursion_is_its_own_error() {
    let mut vm = Vm::compile(
        "fn down(n: int) -> int { down(n + 1) + 1 }\nfn go() -> int { down(0) }",
        |_| {},
    )
    .unwrap();
    vm.run().unwrap();
    vm.set_op_budget(100_000_000);
    vm.set_max_depth(500);
    let err = vm.call_fn_result("go").unwrap_err().to_string();
    assert!(err.contains("recursion is too deep"), "{err}");
    assert!(!err.contains("loop that never ends"), "{err}");
}

#[test]
fn default_depth_limit_stops_endless_recursion_before_the_op_budget() {
    let mut vm = Vm::compile("fn f() -> int { f() + 1 }\nlet x = f();", |_| {}).unwrap();
    vm.set_op_budget(u64::MAX);
    let err = vm.run().unwrap_err().to_string();
    assert!(err.contains("more than 10000 calls"), "{err}");
}

#[test]
fn recursion_inside_the_limit_still_works() {
    let mut vm = Vm::compile(
        "fn sum(n: int) -> int { if n == 0 { 0 } else { n + sum(n - 1) } }\nlet x = sum(5000);",
        |_| {},
    )
    .unwrap();
    vm.set_op_budget(10_000_000);
    vm.run().unwrap();
    assert_eq!(vm.depth(), 1);
}

#[test]
fn a_faulting_probe_leaves_the_depth_unchanged() {
    // hosts probe a live session with injected calls (`resolve_for_scene`); a fault in the probe
    // must unwind its frames inside the Vm, so the host needs no heap snapshot to repair the thread
    let mut vm = Vm::compile(
        "pact Draw { fn draw(self) -> int; }
         struct Spin { n: int }
         impl Draw for Spin { fn draw(self) -> int { while true { } self.n } }
         struct Deep { n: int }
         impl Draw for Deep { fn draw(self) -> int { Deep { n = self.n + 1 }.draw() } }
         let s = Spin { n = 1 };
         let d = Deep { n = 0 };",
        |_| {},
    )
    .unwrap();
    vm.run().unwrap();
    let depth = vm.depth();
    vm.set_op_budget(10_000);
    let _ = vm.resolve_for_scene("s", "draw");
    assert_eq!(vm.depth(), depth);
    vm.set_op_budget(10_000_000);
    vm.set_max_depth(100);
    let _ = vm.resolve_for_scene("d", "draw");
    assert_eq!(vm.depth(), depth);
    // and the session still answers
    assert!(vm.resolve_name("s").is_some());
}

#[macro_use]
mod test_runner;

// `std::debug::caller_line` maps a live frame back to source: the CallNative pc writeback plus
// the chunk loc tables turn `frames[k].ip` into a line number without the script passing one.
#[test]
fn caller_line_reports_the_call_site() {
    // `emit()` is invoked on line 7, the direct `caller_line` on line 9.
    let src = "use std::debug;
fn emit() -> int {
    debug::caller_line(1)!
}
fn run() -> int {
    emit()
}
let VIA_EMIT = run();
let DIRECT = debug::caller_line(0)!;
";
    let mut vm = vm::Vm::execute(src, library::std).expect("compiled");
    assert_eq!(
        vm.resolve_name_to_string("VIA_EMIT").unwrap().unwrap(),
        "6"
    );
    assert_eq!(
        vm.resolve_name_to_string("DIRECT").unwrap().unwrap(),
        "9"
    );
}

#[test]
fn srcfile_returns_the_caller_file() {
    let src = "use std::debug;
let SRC = debug::srcfile(0)!;
";
    let mut vm = vm::Vm::execute(src, library::std).expect("compiled");
    let text = vm.resolve_name_to_string("SRC").unwrap().unwrap();
    assert!(text.starts_with("use std::debug;"), "{text}");
    assert!(text.contains("let SRC"), "{text}");
}

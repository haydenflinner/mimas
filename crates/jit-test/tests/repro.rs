use jit_test::natives;
use vm::{Captured, Vm};

fn run(
    source: &str,
    bodies: Option<Vec<Option<vm::bc::BodyFn>>>,
) -> (Option<Captured>, Vec<Captured>, Option<String>) {
    let (program, sources) =
        Vm::compile_parts(&[("main", source)], natives::install).expect("fixture compiles");
    let mut vm = Vm::new();
    vm.load_prebuilt(program, sources, natives::install);
    if let Some(bodies) = bodies {
        vm.install_bc(bodies);
    }
    let err = vm.run().err().map(|e| e.to_string());
    let test_value = vm.resolve_name("TEST_VALUE");
    let kept = vm.fixture::<natives::Kept>().0.borrow().clone();
    (test_value, kept, err)
}

fn jit_bodies(source: &str) -> jit::Jit {
    let (program, _sources) =
        Vm::compile_parts(&[("main", source)], natives::install).expect("fixture compiles");
    jit::compile(&program).expect("jit compile")
}

#[test]
fn repro_call() {
    for src in [
        // minimal: call + absolve
        r#"fn risky(ok: bool) -> int! { if ok { 42 } else { raise "nope" } }
           let ok = risky(true) absolve |_| 0;
           let TEST_VALUE = f"{ok}";"#,
        // same but is_raised result unused via absolve on raised path
        r#"fn risky(ok: bool) -> int! { if ok { 42 } else { raise "nope" } }
           let failed = risky(false) absolve |_| 99;
           let TEST_VALUE = f"{failed}";"#,
        // call + if, no absolve
        r#"fn seven() -> int { 7 }
           let x = seven(); let h = 0;
           if x == 7 { h += 1; }
           let TEST_VALUE = f"{h}";"#,
        // call then bool-producing op then if
        r#"fn seven() -> int { 7 }
           let x = seven(); let h = 0;
           if x < 10 { h += 1; }
           let TEST_VALUE = f"{h}";"#,
    ] {
        let j = jit_bodies(src);
        assert_eq!(run(src, None), run(src, Some(j.bodies())), "src: {src}");
    }
}

#[test]
fn repro_in() {
    for src in [
        // the bare `in` checks
        r#"let arr = [10, 20, 30]; arr[0] = 5; let h = 0;
           if 20 in arr { h += 1; }
           if 99 in arr { h += 100; }
           let TEST_VALUE = f"{h}";"#,
        // + dict membership
        r#"let arr = [10, 20, 30]; arr[0] = 5; let d = ~{ k = 9 }; d["j"] = 4;
           let h = 0;
           if 20 in arr { h += 1; }
           if 99 in arr { h += 100; }
           if "k" in d { h += 10; }
           let TEST_VALUE = f"{h}";"#,
        // + prior int accumulation on h
        r#"let arr = [10, 20, 30]; arr[0] = 5; let h = 0;
           if 20 in arr { h += 1; }
           if 99 in arr { h += 100; }
           h += 10 + 20 + 5 + 6;
           let TEST_VALUE = f"{h}";"#,
        // absolve/raise context before the checks
        r#"fn risky(ok: bool) -> int! { if ok { 42 } else { raise "nope" } }
           let ok = risky(true) absolve |_| 0;
           let failed = risky(false) absolve |_| 99;
           let arr = [10, 20, 30]; arr[0] = 5;
           let h = 0;
           if 20 in arr { h += 1; }
           if 99 in arr { h += 100; }
           let TEST_VALUE = f"{h}:{ok}:{failed}";"#,
    ] {
        let j = jit_bodies(src);
        assert_eq!(run(src, None), run(src, Some(j.bodies())), "src: {src}");
    }
}

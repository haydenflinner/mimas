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
        r#"let arr = [10, 20, 30]; arr[0] = 5; let d = { k : 9 }; d["j"] = 4;
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

#[test]

#[test]
fn repro_subscripts() {
    let decl = "struct C { data: [int], gone: [int], log: int }";
    for src in [
        // get/set arms + reads, writes, compounds (get+set path)
        &format!("{decl}
            impl C {{ subscript(i: int) -> int {{ get {{ self.data[i] }} set {{ self.data[i] = new_value; }} }} }}
            let c = C {{ data = [1, 2], gone = [0, 0], log = 0 }};
            let v = c[0]; c[1] = 9; c[0] += 4;
            let TEST_VALUE = f\"{{v}}|{{c.data}}\";"),
        // multi-index accessor
        &format!("{decl}
            impl C {{ subscript(i: int, j: int) -> int {{ get {{ self.data[i * 2 + j] }} set {{ self.data[i * 2 + j] = new_value; }} }} }}
            let c = C {{ data = [1, 2, 3, 4], gone = [0, 0, 0, 0], log = 0 }};
            c[1, 0] = 9;
            let TEST_VALUE = f\"{{c.data}}\";"),
        // remove arm dispatch
        &format!("{decl}
            impl C {{ subscript(i: int) -> int {{ get {{ self.data[i] }} set {{ self.data[i] = new_value; }} remove {{ self.gone[i] = 1; }} }} }}
            let c = C {{ data = [1, 2], gone = [0, 0], log = 0 }};
            c[0].remove();
            let TEST_VALUE = f\"{{c.gone}}\";"),
        // inout-only accessor: read/write through the synthesized ops
        &format!("{decl}
            impl C {{ subscript(i: int) -> int {{ inout {{ yield self.data[i]; }} }} }}
            let c = C {{ data = [1, 2], gone = [0, 0], log = 0 }};
            let v = c[0]; c[1] = 9;
            let TEST_VALUE = f\"{{v}}|{{c.data}}\";"),
        // scoped borrow + write-through
        &format!("{decl}
            impl C {{ subscript(i: int) -> int {{ inout {{ yield self.data[i]; }} }} }}
            let c = C {{ data = [1, 2], gone = [0, 0], log = 0 }};
            {{ let p = &c[1]; p = 9; }};
            let TEST_VALUE = f\"{{c.data}}\";"),
        // borrow + place removal (the __RemovablePlace path -- emits
        // trace_edge ops that used to crash the JIT emitter)
        &format!("{decl}
            impl C {{ subscript(i: int) -> int {{ inout {{ yield self.data[i]; }} remove {{ self.gone[i] = 1; }} }} }}
            let c = C {{ data = [1, 2], gone = [0, 0], log = 0 }};
            {{ let p = &c[1]; p.remove(); }};
            let TEST_VALUE = f\"{{c.gone}}\";"),
        // all four arms: compound dispatch prefers inout's modify
        &format!("{decl}
            impl C {{ subscript(i: int) -> int {{ get {{ self.data[i] }} set {{ self.data[i] = new_value; }} inout {{ yield self.data[i]; }} remove {{ self.gone[i] = 1; }} }} }}
            let c = C {{ data = [1, 2], gone = [0, 0], log = 0 }};
            c[0] += 9;
            let TEST_VALUE = f\"{{c.data}}\";"),
        // builtin list slot borrow
        r#"let xs = [1, 2]; { let p = &xs[0]; p = 9; };
           let TEST_VALUE = f"{xs}";"#,
        // nested list slots
        r#"let m = [[1, 2], [3]]; m[1][0] += 10;
           let TEST_VALUE = f"{m}";"#,
        // a generator suspending while a borrow is live
        &format!("{decl}
            impl C {{ subscript(i: int) -> int {{ inout {{ yield self.data[i]; }} }} }}
            let c = C {{ data = [1, 2], gone = [0, 0], log = 0 }};
            subscript work() -> int {{ {{ let p = &c[0]; yield 5; p += 1; }} yield 6; }}
            let it = work();
            let a = it(); let b = it();
            let TEST_VALUE = f\"{{a}}|{{b}}|{{c.data}}\";"),
    ] {
        let j = jit_bodies(src);
        assert_eq!(run(src, None), run(src, Some(j.bodies())), "src: {src}");
    }
}

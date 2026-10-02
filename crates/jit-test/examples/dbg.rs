fn run(source: &str, jit_bodies: bool) -> String {
    let (program, sources) =
        vm::Vm::compile_parts(&[("main", source)], jit_test::natives::install).expect("compile");
    let j = jit::compile(&program).expect("jit compile");
    let mut vm = vm::Vm::new();
    vm.load_prebuilt(program, sources, jit_test::natives::install);
    if jit_bodies {
        vm.install_bc(j.bodies());
    }
    let err = vm.run().err().map(|e| e.to_string());
    format!(
        "TEST_VALUE={:?} kept={:?} err={err:?}",
        vm.resolve_name("TEST_VALUE"),
        vm.fixture::<jit_test::natives::Kept>().0.borrow()
    )
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or("all".into());
    let cases: &[(&str, &str)] = &[
        ("arith", "let TEST_VALUE = 1 + 2 * 3;"),
        (
            "locals",
            "let a = 5; let b = a + 10; let TEST_VALUE = b * 2;",
        ),
        (
            "ifelse",
            "let a = 3; let TEST_VALUE = if a <= 1 { 10 } else { 20 };",
        ),
        (
            "call_direct",
            "fn f(x: int) -> int { x + 1 } let TEST_VALUE = f(4);",
        ),
        (
            "recursion",
            "fn fib(n: int) -> int { if n <= 1 { n } else { fib(n - 1) + fib(n - 2) } } let TEST_VALUE = fib(6);",
        ),
        (
            "forloop",
            "let t = 0; for x in [1,2,3] { t += x; } let TEST_VALUE = t;",
        ),
        (
            "struct",
            "struct P { x: int, y: int } let p = P { x = 3, y = 4 }; let TEST_VALUE = p.x * p.y;",
        ),
        (
            "closure",
            "let add = |a: int, b: int| -> int { a + b }; let TEST_VALUE = add(1,2);",
        ),
        ("native", "let TEST_VALUE = host_add(40, 2);"),
        ("keep", "keep(5); let TEST_VALUE = 9;"),
        (
            "str",
            "let s = \"hi\"; let TEST_VALUE = if s == \"hi\" { 7 } else { 0 };",
        ),
        (
            "dict",
            "let d = dict{}; d[\"k\"] = 5; let TEST_VALUE = d[\"k\"];",
        ),
        (
            "floats",
            "let f = 1.5; let g = f * 2.0; let TEST_VALUE = sqrt(g) ;",
        ),
    ];
    for (name, src) in cases {
        if which != "all" && which != *name {
            continue;
        }
        println!("== {name} interp: {}", run(src, false));
        println!("== {name} jit:");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        println!("{}", run(src, true));
    }
}

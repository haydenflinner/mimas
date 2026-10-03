use std::time::Instant;
fn main() {
    let name = std::env::args().nth(1).unwrap_or("mandelbrot".into());
    let n: u32 = std::env::var("MIMAS_PROBE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let source =
        std::fs::read_to_string(format!("benchmarks/{name}/{name}.mim")).unwrap();
    let (program, sources) =
        vm::Vm::compile_parts(&[("main", &source)], library::std).expect("compile");
    let j = jit::compile(&program).expect("jit");
    let mut best = std::time::Duration::MAX;
    for _ in 0..n {
        let mut vm = vm::Vm::new();
        vm.load_prebuilt(program.clone(), sources.clone(), library::std);
        vm.install_bc(j.bodies());
        let t = Instant::now();
        match vm.run() {
            Ok(_) => best = best.min(t.elapsed()),
            Err(e) => eprintln!("ERR {e}"),
        }
    }
    eprintln!("done in {best:?}");
}

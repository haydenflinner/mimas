use std::time::Instant;
fn main() {
    let name = std::env::args().nth(1).unwrap_or("mandelbrot".into());
    let source =
        std::fs::read_to_string(format!("benchmarks/{name}/{name}.mim")).unwrap();
    let (program, sources) =
        vm::Vm::compile_parts(&[("main", &source)], library::std).expect("compile");
    let j = jit::compile(&program).expect("jit");
    let mut vm = vm::Vm::new();
    vm.load_prebuilt(program, sources, library::std);
    vm.install_bc(j.bodies());
    let t = Instant::now();
    match vm.run() {
        Ok(_) => eprintln!("done in {:?}", t.elapsed()),
        Err(e) => eprintln!("ERR {e}"),
    }
}

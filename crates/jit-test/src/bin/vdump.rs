fn main() {
    let name = std::env::args().nth(1).unwrap();
    let source = std::fs::read_to_string(format!("benchmarks/{name}/{name}.mim")).unwrap();
    let (program, _s) = vm::Vm::compile_parts(&[("main", &source)], library::std).expect("compile");
    match jit::compile(&program) {
        Ok(_) => println!("ok"),
        Err(e) => println!("ERR {e:#}"),
    }
}

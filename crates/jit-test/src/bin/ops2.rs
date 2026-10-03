fn main() {
    let path = std::env::args().nth(1).unwrap();
    let source = std::fs::read_to_string(&path).unwrap();
    let (program, _) =
        vm::Vm::compile_parts(&[("main", &source)], jit_test::natives::install).expect("compile");
    for (i, _c) in program.chunks.iter().enumerate() {
        println!("== body {i}");
        for (off, op) in program.ops(compile::BodyId::from(i as u32)) {
            println!("  {off}: {op:?}");
        }
    }
}

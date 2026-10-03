fn main() {
    let path = std::env::args().nth(1).unwrap();
    let source = std::fs::read_to_string(&path).unwrap();
    let (program, _) =
        mimas::Vm::compile_parts(&[("main", &source)], mimas::library::std).expect("compile");
    for (i, _c) in program.chunks.iter().enumerate() {
        println!("== body {i}");
        for (off, op) in program.ops(mimas::vm::bc::BodyId::from(i as u32)) {
            println!("  {off}: {op:?}");
        }
    }
}

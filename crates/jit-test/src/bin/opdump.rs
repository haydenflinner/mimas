fn main() {
    let path = std::env::args().nth(1).unwrap();
    let source = std::fs::read_to_string(&path).unwrap();
    let (program, _s) = vm::Vm::compile_parts(&[("main", &source)], library::std).expect("compile");
    for b in 0..program.chunks.len() {
        let ops = program.ops(compile::BodyId::from(b as u32));
        println!("=== body {b}: {} ops", ops.len());
        for (i, (off, op)) in ops.iter().enumerate() {
            let s = format!("{:?}", op);
            println!("  {i} @{off} {s:.90}");
        }
    }
}

fn main() {
    let source = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();
    let (program, _s) = vm::Vm::compile_parts(&[("main", source.as_str())], mimas::library::std).unwrap();
    for b in 0..program.chunks.len() {
        let bid = compile::BodyId::from(b as u32);
        println!("== body {b} regs={} params={:?}", program.chunks[bid].regs, program.chunks[bid].params);
        for (i, (off, op)) in program.ops(bid).iter().enumerate() {
            println!("{i} @{off} {op:?}");
        }
    }
}

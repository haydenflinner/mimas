// Dump the decoded op stream per chunk for a .mim file — for JIT hot-op analysis.
fn main() {
    let path = std::env::args().nth(1).expect("usage: ops <file.mim>");
    let source = std::fs::read_to_string(&path).expect("read");
    let (program, _s) =
        vm::Vm::compile_parts(&[("main", source.as_str())], library::std).expect("compile");
    for (id, chunk) in program.chunks.iter() {
        println!(
            "== body {} offset={} args={} regs={} params={:?} captures={:?}",
            id.index(),
            chunk.offset,
            chunk.args,
            chunk.regs,
            chunk.params.iter().map(|r| r.index()).collect::<Vec<_>>(),
            chunk.captures.iter().map(|r| r.index()).collect::<Vec<_>>()
        );
        for (off, op) in program.ops(id) {
            println!("  {off:>4}: {op:?}");
        }
    }
}

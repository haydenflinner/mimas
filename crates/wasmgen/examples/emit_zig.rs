// Compile a .mim file to Zig source for `zig build-lib -target wasm32-freestanding`.
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: emit_zig <file.mim> [out.zig]");
    let out = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "out.zig".to_string());
    let source = std::fs::read_to_string(&path).expect("read");
    let (program, _s) =
        vm::Vm::compile_parts(&[("main", source.as_str())], mimas::library::std).expect("compile");
    let z = mimas_wasmgen::zig::emit_zig(&program).expect("emit_zig");
    std::fs::write(&out, &z.source).expect("write zig");
    let bodies = z
        .bodies
        .iter()
        .map(|b| format!("{{\"body\":{},\"name\":\"{}\"}}", b.body, b.name))
        .collect::<Vec<_>>()
        .join(",");
    let skipped = z
        .skipped
        .iter()
        .map(|s| {
            format!(
                "{{\"body\":{},\"reason\":\"{}\"}}",
                s.body,
                s.reason.replace('"', "'")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    println!("{{\"bodies\":[{}],\"skipped\":[{}]}}", bodies, skipped);
}

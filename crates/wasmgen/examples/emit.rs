// Compile a .mim file to a .wasm module and print a JSON manifest of
// exports/skips for the Node driver.
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: emit <file.mim> [out.wasm]");
    let out = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "out.wasm".to_string());
    let source = std::fs::read_to_string(&path).expect("read");
    let (program, _s) =
        vm::Vm::compile_parts(&[("main", source.as_str())], mimas::library::std).expect("compile");
    let w = mimas_wasmgen::emit(&program).expect("emit");
    std::fs::write(&out, &w.bytes).expect("write wasm");
    let bodies = w
        .bodies
        .iter()
        .map(|b| {
            format!(
                "{{\"body\":{},\"func\":{},\"name\":\"{}\"}}",
                b.body, b.func, b.name
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let skipped = w
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
    println!(
        "{{\"bytes\":{},\"bodies\":[{}],\"skipped\":[{}]}}",
        w.bytes.len(),
        bodies,
        skipped
    );
}

// Compile a .mim file to a .wasm module and print a JSON manifest of
// exports/skips for the Node driver.
//
// usage: emit <file.mim> [out.wasm] [--resume] [--fuel] [--pause] [--cov]
//   --resume  pc-dispatch lane (irreducible CFGs, mid-function pause/resume,
//             memory-resident registers for checkpointing)
//   --fuel    decrement imported env.__fuel (i64) per op/region
//   --pause   check imported env.__pause (i32) at back-edges / region heads
//   --cov     write per-op coverage bytes into exported memory
fn main() {
    let mut args = std::env::args().skip(1);
    let mut opts = mimas_wasmgen::Opts::default();
    let mut resume = false;
    let mut pos = Vec::new();
    for a in args.by_ref() {
        match a.as_str() {
            "--resume" => resume = true,
            "--fuel" => opts.fuel = true,
            "--pause" => opts.pause = true,
            "--cov" => opts.coverage = true,
            _ => pos.push(a),
        }
    }
    let path = pos.first().expect("usage: emit <file.mim> [out.wasm] [flags]");
    let out = pos.get(1).cloned().unwrap_or_else(|| "out.wasm".to_string());
    let source = std::fs::read_to_string(path).expect("read");
    let t0 = std::time::Instant::now();
    let (program, _s) =
        vm::Vm::compile_parts(&[("main", source.as_str())], mimas::library::std).expect("compile");
    let t1 = std::time::Instant::now();
    let w = if resume {
        mimas_wasmgen::resume::emit_resumable(&program, &opts, None, None, 0, 0).expect("emit resumable")
    } else {
        mimas_wasmgen::emit_opts(&program, &opts).expect("emit")
    };
    eprintln!(
        "compile_parts: {:?}  emit: {:?}",
        t1 - t0,
        t1.elapsed()
    );
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

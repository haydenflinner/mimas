//! `cargo run -p mimas-cpp --example dump -- file.cpp` — print the lowered .mim.
fn main() {
    let path = std::env::args().nth(1).expect("usage: dump <file.cpp>");
    let src = std::fs::read_to_string(&path).unwrap();
    let out = match cpp::transpile(&src) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    for d in &out.diagnostics {
        eprintln!("{d}");
    }
    println!("{}", out.source);
}

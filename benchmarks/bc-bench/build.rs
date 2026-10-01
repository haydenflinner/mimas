use std::{env, fs, path::PathBuf};

/// Compile each `benchmarks/<name>/<name>.mim` with the real mimas front end
/// and emit its bcgen-specialized bodies into OUT_DIR, where `main.rs`
/// `include!`s them as `bc::<name>` modules.
fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let benches = ["fib_iter", "fib_rec", "mandelbrot", "prime_numbers", "physics"];
    for name in benches {
        let path = format!("../{name}/{name}.mim");
        println!("cargo:rerun-if-changed={path}");
        let source = fs::read_to_string(&path).expect("read benchmark source");
        let (program, _sources) =
            mimas::Vm::compile_parts(&[("main", source.as_str())], mimas::library::std)
                .unwrap_or_else(|e| panic!("{name} failed to compile: {e:?}"));
        fs::write(out.join(format!("{name}.rs")), bcgen::emit(&program, "mimas::vm"))
            .expect("write emitted module");
    }
}

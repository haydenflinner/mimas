mod natives {
    include!("natives.rs");
}

use std::{env, fs, path::PathBuf};

/// For each `fixtures/*.mimas`: compile to a `Program` and emit a specialized
/// bodies module into `OUT_DIR` (included by `src/lib.rs` as `generated::<stem>`).
/// Compiling here rather than checking generated source in keeps the fixture
/// the single source of truth.
fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=fixtures");
    println!("cargo:rerun-if-changed=natives.rs");
    let mut paths: Vec<_> = fs::read_dir("fixtures")
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "mimas"))
        .collect();
    paths.sort();
    for path in paths {
        let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
        let source = fs::read_to_string(&path).unwrap();
        let (program, _sources) =
            vm::Vm::compile_parts(&[("main", source.as_str())], natives::install)
                .unwrap_or_else(|e| panic!("fixture {} failed to compile: {e:?}", path.display()));
        fs::write(out.join(format!("{stem}.rs")), bcgen::emit(&program, "vm")).unwrap();
    }
}

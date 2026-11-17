//! Structural smoke test for `emit_module` — the emitted text's shape only;
//! the end-to-end parse/lower/run check lives in the `pliron-mimas` driver
//! (tools/pliron-mimas in the outer repo), which owns the pliron deps.

use std::collections::HashMap;

use vm::Vm;
use vm::bc::jit as j;

fn addrs() -> HashMap<&'static str, usize> {
    let mut m = HashMap::new();
    for name in mimas_plgen::spec_names() {
        // bind by table order — every name in plgen's SPECS names a
        // `vm::bc::jit` fn; the driver's map is the binding of record, this
        // just needs *a* valid address per name.
        m.insert(name, j::step_at as *const u8 as usize);
    }
    m
}

#[test]
fn emits_func_per_chunk_and_body_table() {
    let (program, _sources) = Vm::compile_parts(
        &[(
            "main",
            "fn add(a: int, b: int) -> int { a + b }\nlet TEST_VALUE = add(1, 2) + 40;",
        )],
        |api| library::codegen(api),
    )
    .expect("compiles");
    let text = mimas_plgen::emit_module(&program, &vm::bc::jit::layout(), &addrs());
    assert!(text.starts_with("builtin.module @mimas_pliron {"));
    // one llvm.func per chunk plus the exported globals
    for (i, _) in program.chunks.iter() {
        assert!(
            text.contains(&format!("llvm.func @{}", mimas_plgen::body_name(i.index()))),
            "missing func for chunk {}",
            i.index()
        );
    }
    assert!(text.contains(mimas_plgen::BODIES_GLOBAL));
    assert!(text.contains(mimas_plgen::NBODIES_GLOBAL));
    // dispatch is the llvm.switch over op byte offsets
    assert!(text.contains("llvm.switch"));
    // braces balance
    assert_eq!(text.matches('{').count(), text.matches('}').count());
}

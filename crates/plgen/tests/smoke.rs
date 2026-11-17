//! Structural smoke test for `emit_module` — the emitted text's shape only;
//! the end-to-end parse/lower/run check lives in the `pliron-mimas` driver
//! (tools/pliron-mimas in the outer repo), which owns the pliron deps.

use vm::Vm;

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
    let text = mimas_plgen::emit_module(&program, &vm::bc::jit::layout());
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
    // extern decls for the helper surface — the companion-runtime symbols
    for name in mimas_plgen::spec_names() {
        assert!(text.contains(&format!("llvm.func @{name}:")), "missing {name}");
    }
    // dispatch is the llvm.switch over op byte offsets
    assert!(text.contains("llvm.switch"));
    // direct symbol calls, no baked address immediates
    assert!(text.contains("llvm.call @mj_step_at"));
    // braces balance
    assert_eq!(text.matches('{').count(), text.matches('}').count());
}

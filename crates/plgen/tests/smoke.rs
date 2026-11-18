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
    let (hashes, text) =
        mimas_plgen::emit_module(&program, &vm::bc::jit::layout(), false, 0);
    assert!(text.starts_with("builtin.module @mimas_pliron {"));
    // one llvm.func per distinct chunk content plus the exported globals
    for (i, _) in program.chunks.iter() {
        assert!(
            text.contains(&format!(
                "llvm.func @{}",
                mimas_plgen::body_sym(hashes[i.index()])
            )),
            "missing func for chunk {}",
            i.index()
        );
    }
    assert!(text.contains(mimas_plgen::BODIES_GLOBAL));
    assert!(text.contains(mimas_plgen::NBODIES_GLOBAL));
    // extern decls for the helper surface — the companion-runtime symbols,
    // each carrying llvm_func_attrs in the outlined_attributes trailer
    for name in mimas_plgen::spec_names() {
        assert!(text.contains(&format!("llvm.func @{name}:")), "missing {name}");
    }
    assert!(
        text.contains("outlined_attributes:\n!0 = [llvm_func_attrs = llvm.attributes"),
        "missing llvm_func_attrs trailer"
    );
    // dispatch is the llvm.switch over op byte offsets
    assert!(text.contains("llvm.switch"));
    // direct symbol calls, no baked address immediates
    assert!(text.contains("llvm.call @mj_step_at"));
    // braces balance
    assert_eq!(text.matches('{').count(), text.matches('}').count());
}

/// Thin emit leaves the ops out of the body text, so without the
/// chunk-offset salt every thin body is byte-identical and content-hash
/// dedup folds them into one canonical body — a `Call` to `g` then runs
/// `f`'s ops (the `cold`/`absolve` miscompile). Each thin body must keep
/// its own hash; the ops a thin body runs live only in its chunk, so
/// text identity can never prove two thin bodies interchangeable.
#[test]
fn thin_bodies_do_not_dedup() {
    let (program, _sources) = Vm::compile_parts(
        &[(
            "main",
            "fn f() -> int { 1 }\nfn g() -> int { 2 }\nlet V = f() + g();",
        )],
        |api| library::codegen(api),
    )
    .expect("compiles");
    assert!(program.chunks.len() >= 3, "expected main + f + g");
    let (hashes, texts) = mimas_plgen::emit_bodies_t(
        &program,
        &vm::bc::jit::layout(),
        false,
        &|_| mimas_plgen::Tier::Thin,
        0,
    );
    let unique: std::collections::HashSet<_> = hashes.iter().collect();
    assert_eq!(unique.len(), hashes.len(), "thin bodies must not dedup");
    // thin bodies carry no op dispatch at all
    for t in &texts {
        assert!(!t.contains("llvm.switch"), "thin body must not dispatch");
    }
}

/// A thick body over `seg_ops` splits into a `void` outer dispatcher and
/// `i8`-returning `mimas_body_x{h}s{k}` segment fns; an edge leaving a
/// slice goes through an `xoff` trampoline (`code.ip = target; ret 0`),
/// which the outer turns into the owning segment's call. Loop backedges
/// out of a slice are what force the trampolines.
#[test]
fn segmented_body_emits_segments_and_trampolines() {
    let (program, _sources) = Vm::compile_parts(
        &[(
            "main",
            "fn sum(n: int) -> int { let s = 0; for i in n { s += i }; s }\nlet TEST_VALUE = sum(10);",
        )],
        |api| library::codegen(api),
    )
    .expect("compiles");
    // find the biggest chunk and segment it finely so it must split
    let big = (0..program.chunks.len())
        .max_by_key(|&i| program.ops(compile::BodyId::from(i as u32)).len())
        .unwrap();
    let nops = program.ops(compile::BodyId::from(big as u32)).len();
    assert!(nops > 3, "need a multi-op body, got {nops}");
    let (hashes, texts) = mimas_plgen::emit_bodies_t(
        &program,
        &vm::bc::jit::layout(),
        false,
        &|_| mimas_plgen::Tier::Thick,
        3,
    );
    let h = format!("{:016x}", hashes[big]);
    let blob = &texts[big];
    // outer dispatcher keeps the body's own symbol, void (env) signature
    assert!(
        blob.contains(&format!(
            "llvm.func @mimas_body_x{h}: llvm.func <llvm.void"
        )),
        "missing outer dispatcher"
    );
    // at least two i8-returning segment fns under the body's hash
    assert!(
        blob.contains(&format!("llvm.func @mimas_body_x{h}s0:"))
            && blob.contains(&format!("llvm.func @mimas_body_x{h}s1:")),
        "missing segment fns"
    );
    assert!(
        blob.contains("llvm.func <builtin.integer i8 (llvm.ptr (0)) variadic = false>"),
        "segment fns must return i8"
    );
    // a cross-slice edge emits an xoff trampoline in this blob
    assert!(blob.contains("^xoff"), "missing cross-segment trampoline");
}

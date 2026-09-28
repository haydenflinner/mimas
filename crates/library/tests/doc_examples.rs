//! Every ```mimas fenced example in the natives' doc comments runs through
//! the VM. A doc example that fails to compile is a lie the reference prints
//! (`docs/std-api.txt`, `pair docs`), so the examples are the test corpus.

/// Blocks calling these have real-world effects the test process can't
/// survive (`sys::exit` exits it) or can't satisfy (`sys::stdin` blocks).
const SKIP_IF: &[&str] = &[
    "sys::exit",
    "sys::stdin",
    "std::fs",
    "std::process::Command",
    "::sleep",
];

/// Pulls the ` ```mimas ` fenced blocks out of a doc string.
fn mimas_blocks(doc: &str) -> Vec<String> {
    let mut blocks = vec![];
    let mut cur: Option<String> = None;
    for line in doc.lines() {
        match (cur.is_some(), line.trim()) {
            (false, "```mimas") => cur = Some(String::new()),
            (true, t) if t.starts_with("```") => blocks.push(cur.take().unwrap()),
            (true, _) => {
                let c = cur.as_mut().unwrap();
                c.push_str(line);
                c.push('\n');
            }
            _ => {}
        }
    }
    blocks
}

#[test]
fn native_doc_examples_execute() {
    let mut seen = 0;
    let mut skipped = 0;
    let mut failures = String::new();
    for meta in inventory::iter::<vm::api::NativeMeta> {
        let blocks = mimas_blocks(meta.doc);
        seen += blocks.len();
        // Blocks in one doc comment narrate each other: later ones use the
        // earlier ones' bindings, so a native's examples run as one program.
        let src = blocks.join("\n");
        if SKIP_IF.iter().any(|pat| src.contains(pat)) {
            skipped += blocks.len();
            continue;
        }
        if let Err(e) = vm::Vm::execute(&src, library::std) {
            failures.push_str(&format!("\n=== {}\n{src}{e:?}\n", meta.path));
        }
    }
    assert!(seen >= 50, "only {seen} doc blocks found — inventory pruned?");
    assert!(failures.is_empty(), "{seen} blocks ({skipped} skipped):{failures}");
}

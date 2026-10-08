//! Waffle-spike driver: `wemit <out.wasm> [src.mimas]` — compiles the source
//! through `wemit::emit_waffle`, prints emitted/skipped bodies, writes the
//! wasm, and (for comparison) also runs `emit_ir` on the same IR.

use mimas::vm::Vm;
use mimas_wasmgen::{Opts, irgen::emit_ir, wfull::emit_waffle_ir};

const SRC: &str = r#"
fn fib(n: int) -> int {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

fn sum(n: int) -> int {
    let s = 0;
    for i in 0 .. n { s += i; }
    s
}

fn wh(n: int) -> int {
    let i = 0;
    let s = 0;
    while i < n {
        i += 1;
        if i mod 2 == 0 { continue; }
        if s > 100 { break; }
        s += i;
    }
    s
}

fn sw(n: int) -> int {
    match n {
        0 => 10,
        1 => 20,
        2 => 30,
        _ => -1,
    }
}

fn mx(a: int, b: int) -> int {
    let r = if a > b { a } else { b };
    r
}

fn fl(x: float) -> float {
    let t = 0.0;
    for i in 0 .. 5 { t += x * i.to_float(); }
    t.sqrt()
}

let r = fib(10) + sum(20) + wh(30) + sw(2) + mx(3, 4) + fl(3.5).to_int();
"#;

fn main() {
    let out = std::env::args().nth(1).expect("usage: wemit <out.wasm> [src]");
    let src = std::env::args()
        .nth(2)
        .map(|p| std::fs::read_to_string(&p).expect("read src"))
        .unwrap_or_else(|| SRC.to_string());
    let (program, ir, _s) =
        Vm::compile_parts_ir(&[("<t>", &src)], |api| mimas::library::std(api)).expect("compile");

    let w = emit_waffle_ir(&ir, &program.strs, &Opts::default(), None, None, 0, 0)
        .expect("emit_waffle_ir");
    for b in &w.bodies {
        println!("wemit b{}", b.body);
    }
    for s in &w.skipped {
        println!("wemit skip b{} {}", s.body, s.reason);
    }
    std::fs::write(&out, &w.bytes).expect("write");
    eprintln!("wrote {} bytes to {out}", w.bytes.len());

    let mut names = std::collections::HashMap::new();
    fn walk(
        m: &mimas::vm::Module,
        prefix: &str,
        names: &mut std::collections::HashMap<usize, String>,
    ) {
        for (n, f) in &m.functions {
            names.insert(f.body.index(), format!("{prefix}{n}"));
        }
        for (n, sub) in &m.modules {
            walk(sub, &format!("{prefix}{n}::"), names);
        }
    }
    walk(&program.root, "", &mut names);
    names.insert(program.entry.index(), "<top-level>".into());

    let i = emit_ir(&ir, &program.strs, &Opts::default(), None, None, 0, 0).expect("emit_ir");
    for b in &i.bodies {
        println!(
            "irgen b{} -> {}",
            b.body,
            names.get(&b.body).cloned().unwrap_or_default()
        );
    }
    for s in &i.skipped {
        println!("irgen skip b{} {}", s.body, s.reason);
    }
}

//! Dump mimas Ir block structure: `dump_cfg [src.mimas]`

use mimas::vm::Vm;

const SRC: &str = r#"
fn ife(a: int) -> int {
    let r = 0;
    if a > 0 { r = 1; } else if a < 0 { r = -1; } else { r = 2; }
    r
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

let r = ife(3) + wh(10);
"#;

fn main() {
    let src = std::env::args()
        .nth(1)
        .map(|p| std::fs::read_to_string(&p).expect("read"))
        .unwrap_or_else(|| SRC.to_string());
    let (_program, ir, _s) =
        Vm::compile_parts_ir(&[("<t>", &src)], |api| mimas::library::std(api)).expect("compile");
    for (bid, body) in ir.bodies.iter() {
        println!(
            "=== body {} params={:?} locals={}",
            bid.index(),
            body.params,
            body.locals.len()
        );
        for (bkid, block) in body.blocks.iter() {
            println!("  block {}:", bkid.index());
            for &iid in &block.stream {
                println!("    i{} {:?}", iid.index(), body.instructions[iid]);
            }
        }
    }
}

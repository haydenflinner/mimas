use jit::JitSession;
use jit_test::natives;
use vm::Vm;

fn main() {
    let source = r#"
fn fadd(a: int, b: int) -> int { a + b }
fn fsub(a: int, b: int) -> int { a - b }
fn apply(sub: bool, x: int) -> int {
    let f = if sub { fsub } else { fadd };
    f(x, 2) * 3
}
let TEST_VALUE = apply(false, 5);
"#;
    let (program, sources) =
        Vm::compile_parts(&[("main", source)], natives::install).expect("compile");
    let jitprog = program.clone();
    let mut vm = Vm::new();
    vm.load_prebuilt(program, sources, natives::install);
    vm.run().unwrap();
    println!("after run ops_left={}", vm.ops_left());

    let mut sess = JitSession::observe(&mut vm, &jitprog).unwrap();
    for _ in 0..4 {
        let v: i64 = vm.call("apply", (false, 5i64)).unwrap();
        println!("call v={v} left={}", vm.ops_left());
    }
    let facts = sess.facts();
    sess.specialize(&mut vm, &jitprog, &facts).unwrap();
    println!("specialized, left={}", vm.ops_left());
    match vm.call::<i64>("apply", (false, 5i64)) {
        Ok(v) => println!("jit call v={v} left={}", vm.ops_left()),
        Err(e) => println!("jit call ERR {e}"),
    }
}

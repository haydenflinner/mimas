fn main() {
    let path = std::env::args().nth(1).unwrap();
    let source = std::fs::read_to_string(&path).unwrap();
    let (program, _) =
        vm::Vm::compile_parts(&[("main", &source)], |api| {
            jit_test::natives::install(api);
            #[vm::native]
            fn print<'gc>(_ctx: vm::Ctx<'gc>, _msg: vm::Val<'gc>) {}
            api.add_named("print", print);
        }).expect("compile");
    let _j = jit::compile(&program).expect("jit");
}

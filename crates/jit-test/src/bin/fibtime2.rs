use std::time::Instant;
fn main() {
    let source = std::fs::read_to_string("benchmarks/fib_rec/fib_rec.mim").unwrap();
    let (program, sources) =
        vm::Vm::compile_parts(&[("main", &source)], |api| {
            jit_test::natives::install(api);
            #[vm::native]
            fn print<'gc>(_ctx: vm::Ctx<'gc>, _msg: vm::Val<'gc>) {}
            api.add_named("print", print);
        }).expect("compile");
    let j = jit::compile(&program).expect("jit");
    let mut vm = vm::Vm::new();
    vm.load_prebuilt(program.clone(), sources.clone(), |api| {
        jit_test::natives::install(api);
        #[vm::native]
        fn print<'gc>(_ctx: vm::Ctx<'gc>, _msg: vm::Val<'gc>) {}
        api.add_named("print", print);
    });
    vm.install_bc(j.bodies());
    let t = Instant::now();
    loop {
        let mut vm2 = vm::Vm::new();
        vm2.load_prebuilt(program.clone(), sources.clone(), |api| {
            jit_test::natives::install(api);
            #[vm::native]
            fn print<'gc>(_ctx: vm::Ctx<'gc>, _msg: vm::Val<'gc>) {}
            api.add_named("print", print);
        });
        vm2.install_bc(j.bodies());
        vm2.run().unwrap();
        if t.elapsed().as_secs() > 20 { break; }
    }
    eprintln!("done {:?}", t.elapsed());
}

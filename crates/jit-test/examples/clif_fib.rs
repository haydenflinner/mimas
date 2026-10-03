//! Like-for-like codegen comparison: `fib(n) = n<2 ? n : fib(n-1)+fib(n-2)`
//! emitted as a bare Cranelift function — the same backend version, ISA,
//! and `opt_level=speed` flags `mimas_jit::compile` uses, but none of the
//! VM semantics (no BodyEnv, no frames, no quota, no dispatch). Times it
//! exactly like bc-bench's native lane so the three numbers split the
//! jit-vs-native gap into codegen quality vs VM-model overhead.
//!
//! Run: `cargo run -p mimas-jit-test --release --example clif_fib`

use cranelift_codegen::ir::{types, AbiParam, InstBuilder};
use cranelift_codegen::settings::Configurable;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::time::Instant;

fn main() {
    let mut flags = cranelift_codegen::settings::builder();
    flags.set("opt_level", "speed").unwrap();
    let isa = cranelift_native::builder()
        .unwrap()
        .finish(cranelift_codegen::settings::Flags::new(flags))
        .unwrap();
    let mut module =
        JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));

    // extern "C" fn(i64) -> i64
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::I64));
    let fid = module
        .declare_function("fib", Linkage::Export, &sig)
        .unwrap();

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fbc = FunctionBuilderContext::new();
    {
        let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fbc);
        let entry = fb.create_block();
        let ret_n = fb.create_block();
        let rec = fb.create_block();
        fb.append_block_params_for_function_params(entry);
        fb.switch_to_block(entry);
        let n = fb.block_params(entry)[0];
        let two = fb.ins().iconst(types::I64, 2);
        let lt = fb.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, n, two);
        fb.ins().brif(lt, ret_n, &[], rec, &[]);

        fb.switch_to_block(ret_n);
        fb.ins().return_(&[n]);

        fb.switch_to_block(rec);
        let fref = module.declare_func_in_func(fid, fb.func);
        let one = fb.ins().iconst(types::I64, 1);
        let nm1 = fb.ins().isub(n, one);
        let c1 = fb.ins().call(fref, &[nm1]);
        let r1 = fb.inst_results(c1)[0];
        let two2 = fb.ins().iconst(types::I64, 2);
        let nm2 = fb.ins().isub(n, two2);
        let c2 = fb.ins().call(fref, &[nm2]);
        let r2 = fb.inst_results(c2)[0];
        let sum = fb.ins().iadd(r1, r2);
        fb.ins().return_(&[sum]);
        fb.seal_all_blocks();
        let fe_cfg = module.isa().frontend_config();
        fb.finalize(fe_cfg);
    }
    module.define_function(fid, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();
    let fib: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(module.get_finalized_function(fid)) };

    // same work as bc-bench's fib_rec lane: fib(30) x20
    assert_eq!(fib(10), 55);
    let t = Instant::now();
    let mut acc = 0i64;
    for _ in 0..20 {
        acc += std::hint::black_box(fib(std::hint::black_box(30)));
    }
    let el = t.elapsed();
    assert_eq!(acc, 832040 * 20);
    println!("clif-fib fib(30)x20: {:.1}ms", el.as_secs_f64() * 1e3);
}

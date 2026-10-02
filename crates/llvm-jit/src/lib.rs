//! mimas-llvm-jit — LLVM (MCJIT) tier for mimas bytecode bodies. SPIKE.
//!
//! The third tier: same `BodyFn` contract as `mimas-jit` (Cranelift) — one
//! native function per chunk, installed via `Vm::install_bc`. Emission
//! mirrors `jit::emit`'s structure — a dense dispatch on `code.ip`, per-op
//! `bcn` quota bookkeeping, scalar-register shadows — but shadows are LLVM
//! allocas (mem2reg promotes them to SSA) and the module runs `default<O2>`
//! before MCJIT. Everything not specialized runs the interpreter's own
//! `step_one` through `vm::bc::jit::step_at`, so coverage is total.

mod emit;

use compile::Program;
use std::sync::LazyLock;
use vm::bc::{BodyFn, jit};

/// One compiled program: owns the execution engine (and therefore the code
/// memory) plus the install table for `Vm::install_bc`.
pub struct Jit {
    // `ExecutionEngine<'ctx>` borrows its `Context`; rather than a
    // self-referential struct the context is leaked — `compile` is a
    // once-per-program operation (spike simplification).
    _engine: inkwell::execution_engine::ExecutionEngine<'static>,
    bodies: Vec<Option<BodyFn>>,
    /// The `mj_call_body`/`mj_call_dyn` body-pointer table (`tbl` arg). The
    /// boxed slice's buffer is stable; emitted code passes its address to
    /// the megashims and `emit_program` fills it once addresses are known.
    _tbl: Box<[usize]>,
}

impl Jit {
    /// The `install_bc` table — one `BodyFn` per `BodyId`, all `Some`.
    pub fn bodies(&self) -> Vec<Option<BodyFn>> {
        self.bodies.clone()
    }
}

/// JIT compile failure — LLVM/target/module errors or a malformed program.
/// Compilation is all-or-nothing; a failed `compile` leaves the
/// interpreter-only path as the fallback.
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// C ABI parameter type for a helper signature: everything is pointer-sized
/// except `u8`/`u32` scalars and `Ctx`, which the `#[repr(C)]` two-pointer
/// struct passes as *two* consecutive pointer params.
#[derive(Clone, Copy)]
pub(crate) enum Pt {
    /// pointer / `usize` / `u64` / `i64`
    P,
    I8,
    I32,
    F64,
    /// `Ctx<'gc>` — `#[repr(C)]` pair of pointers, by value.
    Ctx,
}

/// Helper return type.
#[derive(Clone, Copy)]
pub(crate) enum Hr {
    Void,
    /// `u8` status.
    U8,
    /// pointer / `usize`.
    P,
}

/// One row of the `vm::bc::jit` ABI contract: the Rust fn's address and its
/// C-ABI signature. (Binding is by absolute address — the name only exists
/// for readability of the emitted IR.)
pub(crate) struct Spec {
    pub name: &'static str,
    pub addr: usize,
    pub params: &'static [Pt],
    pub ret: Hr,
}

// Safe: fn items coerce to thin code pointers.
macro_rules! spec {
    ($name:literal, $f:expr, [$($p:ident),*], $r:ident) => {
        Spec {
            name: $name,
            addr: $f as *const u8 as usize,
            params: &[$(Pt::$p),*],
            ret: Hr::$r,
        }
    };
}

/// The helper table — [`H`] indexes this. Verbatim copy of `mimas-jit`'s
/// `SPECS`: same `vm::bc::jit` shim layer, same C-ABI contract.
pub(crate) static SPECS: LazyLock<Vec<Spec>> = LazyLock::new(|| {
    vec![
        spec!("mj_paused_ptr", jit::paused_ptr, [Ctx], P),
        spec!("mj_ops_left_ptr", jit::ops_left_ptr, [P], P),
        spec!("mj_ip_ptr", jit::ip_ptr, [P], P),
        spec!("mj_frame_base", jit::frame_base, [P], P),
        spec!("mj_frame_nregs", jit::frame_nregs, [P, P], P),
        spec!("mj_regs_ptr", jit::regs_ptr, [P], P),
        spec!("mj_frames_len", jit::frames_len, [P], P),
        spec!("mj_ri", jit::ri, [P, P, P], U8),
        spec!("mj_rf", jit::rf, [P, P, P], U8),
        spec!("mj_rb", jit::rb, [P, P, P], U8),
        spec!("mj_is_bool", jit::is_bool, [P, P, I8], U8),
        spec!("mj_wr_i", jit::wr_i, [P, P, P], Void),
        spec!("mj_wr_f", jit::wr_f, [P, P, F64], Void),
        spec!("mj_wr_b", jit::wr_b, [P, P, I8], Void),
        spec!("mj_wr_null", jit::wr_null, [P, P], Void),
        spec!("mj_wr_fn", jit::wr_fn, [P, P, I32], Void),
        spec!("mj_wr_v", jit::wr_v, [P, P, P], Void),
        spec!("mj_mv", jit::mv, [P, P, P], Void),
        spec!("mj_rval", jit::rval, [P, P], P),
        spec!("mj_out_next", jit::out_next, [P], Void),
        spec!("mj_out_err", jit::out_err, [P, I8], Void),
        spec!("mj_out_return", jit::out_return, [P, P], Void),
        spec!("mj_out_call", jit::out_call, [P, P, P, P, P, P], Void),
        spec!(
            "mj_out_call_direct",
            jit::out_call_direct,
            [P, P, I32, P, P, P],
            Void
        ),
        spec!("mj_step_at", jit::step_at, [P, P, P, P, Ctx, P, P], U8),
        spec!("mj_len", jit::len, [P, P, P, P], U8),
        spec!("mj_is_raised", jit::is_raised, [P, P, P], Void),
        spec!("mj_unwrap_raised", jit::unwrap_raised, [P, P, P], Void),
        spec!("mj_unwrap", jit::unwrap, [P, P, P, P], U8),
        spec!("mj_unwrap_unit", jit::unwrap_unit, [P, P, P, P], U8),
        spec!("mj_raise", jit::raise, [P, P, P], Void),
        spec!("mj_bin", jit::bin, [P, P, P, I8, P, Ctx, P], U8),
        spec!("mj_unary", jit::unary, [P, P, I8, P, Ctx, P], U8),
        spec!("mj_get_index", jit::get_index, [P, P, P, P, I8, Ctx, P], U8),
        spec!("mj_set_index", jit::set_index, [P, P, P, P, Ctx, P], U8),
        spec!("mj_get_field", jit::get_field, [P, P, P, P, I8, P], U8),
        spec!("mj_set_field", jit::set_field, [P, P, P, P, Ctx, P], U8),
        spec!("mj_push", jit::push, [P, P, P, Ctx], Void),
        spec!("mj_insert", jit::insert, [P, P, I32, P, Ctx, P], Void),
        spec!("mj_contains", jit::contains_op, [P, P, P, P, I8], Void),
        spec!("mj_is_instance", jit::is_instance, [P, P, P, I32], Void),
        spec!("mj_new_array", jit::new_array, [P, P, Ctx], Void),
        spec!("mj_new_dict", jit::new_dict, [P, P, Ctx], Void),
        spec!(
            "mj_new_instance",
            jit::new_instance,
            [P, P, I32, P, P, Ctx],
            Void
        ),
        spec!(
            "mj_new_closure",
            jit::new_closure,
            [P, P, I32, P, P, Ctx],
            Void
        ),
        spec!(
            "mj_call_native",
            jit::call_native,
            [P, P, P, I32, P, P, P, Ctx, P],
            U8
        ),
        spec!(
            "mj_call_target",
            jit::call_target,
            [P, P, P, P, P, P, P],
            U8
        ),
        spec!(
            "mj_enter",
            jit::enter,
            [P, P, P, I32, P, P, P, P, P, P, P],
            U8
        ),
        spec!("mj_pop_return", jit::pop_return, [P, P, P], U8),
        spec!("mj_bin_str", jit::bin_str, [P, P, P, P, I8], U8),
        spec!(
            "mj_load_const_str",
            jit::load_const_str,
            [P, P, I32, Ctx, P],
            Void
        ),
        spec!("mj_flush", jit::flush, [P, P, P], Void),
        spec!(
            "mj_call_body",
            jit::call_body,
            [P, P, P, P, P, P, P, P, P, P, P, P, P, P, P],
            P
        ),
        spec!(
            "mj_call_dyn",
            jit::call_dyn,
            [P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P],
            P
        ),
    ]
});

/// Helper function indexes — mirrors SPECS 1:1 (not every entry is used).
#[derive(Clone, Copy)]
#[allow(dead_code)]
#[repr(usize)]
pub(crate) enum H {
    PausedPtr = 0,
    OpsLeftPtr,
    IpPtr,
    FrameBase,
    FrameNregs,
    RegsPtr,
    FramesLen,
    Ri,
    Rf,
    Rb,
    IsBool,
    WrI,
    WrF,
    WrB,
    WrNull,
    WrFn,
    WrV,
    Mv,
    Rval,
    OutNext,
    OutErr,
    OutReturn,
    OutCall,
    OutCallDirect,
    StepAt,
    Len,
    IsRaised,
    UnwrapRaised,
    Unwrap,
    UnwrapUnit,
    Raise,
    Bin,
    Unary,
    GetIndex,
    SetIndex,
    GetField,
    SetField,
    Push,
    Insert,
    ContainsOp,
    IsInstance,
    NewArray,
    NewDict,
    NewInstance,
    NewClosure,
    CallNative,
    CallTarget,
    Enter,
    PopReturn,
    BinStr,
    LoadConstStr,
    Flush,
    CallBody,
    CallDyn,
}

/// Compile every chunk of `program` to native code. The returned [`Jit`] owns
/// the code memory — keep it alive for as long as the bodies are installed.
pub fn compile(program: &Program) -> Result<Jit, Error> {
    debug_assert_eq!(SPECS.len(), H::CallDyn as usize + 1, "H/SPECS order drifted");
    emit::emit_program(program)
}

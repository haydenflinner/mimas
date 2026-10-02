//! mimas-jit — Cranelift JIT tier for mimas bytecode bodies.
//!
//! The second Futamura projection on the interpreter, next to bcgen: where
//! bcgen emits *Rust source* that must be recompiled ahead-of-time, this crate
//! emits machine code in-process via Cranelift. Each chunk of a [`Program`]
//! becomes one native function at the [`BodyFn`] ABI — the same
//! `extern "C"` signature the bcgen `body_N_abi` trampolines use — so
//! `Vm::install_bc` accepts JIT bodies interchangeably:
//!
//! ```ignore
//! let (program, sources) = Vm::compile_parts(files, std_install)?;
//! let jit = mimas_jit::compile(&program)?;          // native code for every chunk
//! vm.load_prebuilt(program, sources, std_install);
//! vm.install_bc(jit.bodies());
//! vm.run()
//! ```
//!
//! Semantics are single-sourced like bcgen's: per-op bookkeeping replicates
//! `run_dispatch` (pause/fuel/`ops_left` in the same order), scalar registers
//! shadow into SSA values with `ok` flags and deferred writeback flushed at
//! every observable boundary, and anything the emitter doesn't specialize runs
//! the interpreter's own `step_one` through the `vm::bc::jit` shim table — so
//! coverage is always total.

mod emit;
mod session;

use compile::Program;
use cranelift_codegen::ir::{AbiParam, types};
use cranelift_codegen::settings::Configurable;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, FuncId, Linkage, Module, default_libcall_names};
use std::sync::LazyLock;
use vm::bc::{BodyFn, jit};

/// One compiled program: owns the JIT module (and therefore the code memory)
/// plus the install table for `Vm::install_bc`.
pub struct Jit {
    _module: JITModule,
    bodies: Vec<Option<BodyFn>>,
}

impl Jit {
    /// The `install_bc` table — one `BodyFn` per `BodyId`, all `Some`.
    pub fn bodies(&self) -> Vec<Option<BodyFn>> {
        self.bodies.clone()
    }
}

/// JIT compile failure — module/layout errors from Cranelift or a malformed
/// program. `Error` is deliberately flat; compilation is all-or-nothing so a
/// failed `compile` leaves the interpreter-only path as the fallback.
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

// ---- runtime facts & specialization ----

use std::collections::HashMap;

pub use session::JitSession;

/// The `Val` discriminant the profiler (or the host) saw in a register at
/// body entry — one entry per tag. `Mixed` = more than one tag was seen;
/// `Unknown` = never observed (never entered).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ObsTag {
    /// No observation recorded — treated like `Mixed` (no speculation).
    #[default]
    Unknown,
    Null,
    Bool,
    Int,
    Float,
    Fn,
    Str,
    Array,
    /// `Val::IntArray`/`Val::FloatArray` — the typed-array tags.
    IntArray,
    /// See `IntArray`.
    FloatArray,
    Dict,
    Instance,
    Closure,
    /// Anything else (`Raised`, feature-gated payloads, ...).
    Other,
    /// More than one tag observed — no specialization.
    Mixed,
}

/// What one register held on every observed body entry: a merged tag plus,
/// when every observation carried the *same* GC payload, that payload's
/// address (`Gc::as_ptr()` — used to key [`Facts::frozen`] lookups; `0` =
/// "not one stable pointer").
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Obs {
    /// Merged tag — `Mixed` when more than one was seen.
    pub tag: ObsTag,
    /// The one GC payload address observed, else `0`.
    pub ptr: usize,
}

/// Specialization facts for one chunk of the program.
#[derive(Clone, Debug, Default)]
pub struct BodyFacts {
    /// `entry[r]` — the merged observation for register `r` at body entry.
    /// `Int`/`Float` entries force the register into the scalar-shadow set:
    /// emitted code then assumes the tag until a runtime guard says
    /// otherwise, deopting through `step` on a mismatch. Container tags
    /// (`Dict`, `Array`, `IntArray`, `FloatArray`, `Instance`, ...) record
    /// the GC pointer for [`Facts::frozen`] lookups.
    pub entry: Vec<Obs>,
    /// Monomorphic dynamic-call sites: the caller-frame resume offset (the
    /// `code.ip` the interpreter saved — i.e. the byte offset of the op
    /// *after* the `Op::Call`) → the one `BodyId` index ever observed as the
    /// `Val::Fn`/`Val::Closure` target. The emitter guards the callee's
    /// stored body index and calls it directly; misses run `mj_call_dyn`.
    pub calls: HashMap<usize, u32>,
    /// `sites[ip]` — the merged observation of the *receiver operand* of the
    /// `GetIndex`/`GetField` op at byte offset `ip`. Unlike `entry` this
    /// sees mid-body values, so a global `LoadEntry`'d into a reg every call
    /// still yields the container's stable payload pointer — the key
    /// [`Facts::frozen`] bakes on.
    pub sites: HashMap<usize, Obs>,
}

/// A GC-rooted container the host declares immutable for the lifetime of the
/// specialized bodies — reads from it may be constant-folded at compile time.
///
/// # ⚠️ THE FROZEN CONTRACT — READ THIS ⚠️
///
/// Marking an object frozen is a promise that **nothing mutates it** — no
/// `SetIndex`/`SetField`/`Push`/`Insert` from script, no `borrow_mut` from a
/// native, for the entire time the specialized bodies are installed and may
/// run. Breaking the promise makes baked reads return stale values — silent
/// wrong answers, not crashes.
///
/// The pointer (`Gc::as_ptr` of the `RefLock`-holding cell) is embedded in
/// generated code. The object must therefore outlive the [`Jit`]: keep its
/// arena alive and make sure it is a real GC root — a frozen address whose
/// object is collected and reused would alias a *different* container and
/// bake lies. In practice: freeze objects reachable from the program's own
/// globals/frame (`Gc` is a root through the arena) while the `Vm` lives.
#[derive(Clone, Copy, Debug)]
pub struct Frozen {
    /// What the payload address points at — selects which accessor bakes.
    pub kind: FrozenKind,
    /// `false` (default): every baked read still verifies the receiver's
    /// payload pointer at runtime — always correct, just cheaper than the
    /// general path. `true` additionally skips that check — the host is
    /// asserting the register provably holds this object at the site (e.g.
    /// an entry fact), which is only sound under the frozen contract above.
    pub unguarded: bool,
}

/// Which container type a [`Frozen`] address names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrozenKind {
    /// `Val::Array`'s `Gc<RefLock<Vec<Val>>>` payload.
    Array,
    /// `Val::IntArray`/`Val::FloatArray`'s `Gc<RefLock<ArrayStore>>` payload.
    Seq,
    /// `Val::Dict`'s `Gc<RefLock<DictMap<Val>>>` payload.
    Dict,
    /// `Val::Instance`'s `Gc<RefLock<InstanceData>>` payload.
    Instance,
}

/// Runtime specialization facts for [`compile_with`] — produced by
/// [`JitSession`]'s observation shims or built by hand.
///
/// Every fact is *guarded*: emitted code checks the speculation at runtime
/// and falls back to the interpreter's own semantics on a mismatch, so wrong
/// facts cost speed, never correctness — except [`Facts::frozen`], which is
/// a host contract (see [`Frozen`]).
#[derive(Clone, Debug, Default)]
pub struct Facts {
    /// `bodies[b]` — facts for `BodyId` `b`. Missing bodies get no
    /// specialization.
    pub bodies: Vec<BodyFacts>,
    /// Frozen GC roots the host guarantees immutable — keyed by
    /// `Gc::as_ptr` address. See [`Frozen`] for the contract.
    pub frozen: HashMap<usize, Frozen>,
}

impl From<cranelift_module::ModuleError> for Error {
    fn from(e: cranelift_module::ModuleError) -> Self {
        Error(e.to_string())
    }
}

/// C ABI parameter type for a helper signature: everything is pointer-sized
/// except `u8`/`u32` scalars and `Ctx`, which the `#[repr(C)]` two-pointer
/// struct passes as *two* consecutive i64 params.
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

/// One row of the `vm::bc::jit` ABI contract: symbol name (as bound through
/// `JITBuilder::symbol`), the Rust fn's address, and its C-ABI signature.
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

/// The helper table — [`H`] indexes this. Names are arbitrary (binding is by
/// address); the `mj_*` prefix keeps them recognizable in profiles. Function
/// addresses aren't const-evaluable, so this is lazily initialized.
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

/// Helper function ids — indexes [`SPECS`]. (Not every table entry is used by
/// the emitter yet — the enum mirrors SPECS so the mapping stays 1:1.)
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

impl H {
    #[allow(dead_code)]
    pub(crate) fn spec(self) -> &'static Spec {
        &SPECS[self as usize]
    }
}

/// Compile every chunk of `program` to native code. The returned [`Jit`] owns
/// the code memory — keep it alive for as long as the bodies are installed.
pub fn compile(program: &Program) -> Result<Jit, Error> {
    compile_with(program, &Facts::default())
}

/// [`compile`](Self::compile) specialized by runtime [`Facts`]:
///
/// - `bodies[b].entry[r]` observed `Int`/`Float` on every entry → the
///   register joins the scalar-shadow set; entry probes the tag once and
///   reads deopt through `step` on a mismatch.
/// - `bodies[b].calls` → a monomorphic inline cache on that `Op::Call` site:
///   the callee's stored body index is checked and the body called directly;
///   a miss runs `mj_call_dyn`.
/// - `frozen` container addresses known from `entry` observations (or hit
///   via an `unguarded` declaration) let `GetField`/`GetIndex` bake the
///   resolved value — `Int`/`Float`/`Bool`/`Null`/`Fn` only — as an
///   immediate, pointer-guarded unless `unguarded`.
///
/// Facts carry no lifetimes — the addresses in them must stay live and
/// truthful for as long as the returned [`Jit`] is installed (see
/// [`Frozen`]).
pub fn compile_with(program: &Program, facts: &Facts) -> Result<Jit, Error> {
    debug_assert_eq!(
        SPECS.len(),
        H::CallDyn as usize + 1,
        "H/SPECS order drifted"
    );
    let mut flags = cranelift_codegen::settings::builder();
    flags
        .set("opt_level", "speed")
        .map_err(|e| Error(format!("cranelift flag: {e}")))?;
    let isa = cranelift_native::builder()
        .map_err(|e| Error(format!("native isa: {e}")))?
        .finish(cranelift_codegen::settings::Flags::new(flags))
        .map_err(|e| Error(format!("isa finish: {e}")))?;
    let mut jb = JITBuilder::with_isa(isa, default_libcall_names());
    for s in SPECS.iter() {
        jb.symbol(s.name, s.addr as *const u8);
    }
    let mut module = JITModule::new(jb);

    // The BodyFn C ABI: (thread, code, Ctx-by-value as two i64s, strs, chunks,
    // signatures, fuel, op_ip, out) — ten pointer-width params, no return.
    let mut body_sig = module.make_signature();
    for _ in 0..10 {
        body_sig.params.push(AbiParam::new(types::I64));
    }

    // Helper imports — named symbols resolved through JITBuilder.
    let helper_ids: Vec<FuncId> = SPECS
        .iter()
        .map(|s| {
            let mut sig = module.make_signature();
            for p in s.params {
                match p {
                    Pt::P => sig.params.push(AbiParam::new(types::I64)),
                    Pt::I8 => sig.params.push(AbiParam::new(types::I8)),
                    Pt::I32 => sig.params.push(AbiParam::new(types::I32)),
                    Pt::F64 => sig.params.push(AbiParam::new(types::F64)),
                    Pt::Ctx => {
                        sig.params.push(AbiParam::new(types::I64));
                        sig.params.push(AbiParam::new(types::I64));
                    }
                }
            }
            match s.ret {
                Hr::Void => {}
                Hr::U8 => sig.returns.push(AbiParam::new(types::I8)),
                Hr::P => sig.returns.push(AbiParam::new(types::I64)),
            }
            module.declare_function(s.name, Linkage::Import, &sig)
        })
        .collect::<Result<_, _>>()?;

    // Body decls — anonymous (called by FuncId/pointer, never by name).
    let nbodies = program.chunks.len();
    let body_ids: Vec<FuncId> = (0..nbodies)
        .map(|_| module.declare_anonymous_function(&body_sig))
        .collect::<Result<_, _>>()?;

    // `bodies` table: fn pointers for dynamic `Call` (the callee body index is
    // only known at runtime), mirroring bcgen's `static BODIES`.
    let bodies_data = module.declare_anonymous_data(false, false)?;
    let mut dd = DataDescription::new();
    dd.define(vec![0u8; nbodies * 8].into());
    dd.set_align(8);
    for (i, id) in body_ids.iter().enumerate() {
        let fref = module.declare_func_in_data(*id, &mut dd);
        dd.write_function_addr((i * 8) as u32, fref);
    }
    module.define_data(bodies_data, &dd)?;

    let mut fbc = cranelift_frontend::FunctionBuilderContext::new();
    let mut ctx = module.make_context();
    // Probed layouts (`Val` tag/payload, `ThreadState`/`Frame`/`Decoder`
    // fields, `Vec` header order) — emitted code reads/writes these inline.
    let lyt = jit::layout();
    let empty_bf = BodyFacts::default();
    for body in 0..nbodies {
        let bfacts = facts.bodies.get(body).unwrap_or(&empty_bf);
        let spec = emit::BodySpec {
            facts: bfacts,
            frozen: &facts.frozen,
        };
        // emit_body defines the function itself (so it can map verifier
        // errors to the offending chunk's CLIF).
        emit::emit_body(
            &mut module,
            program,
            body,
            &body_sig,
            &helper_ids,
            &body_ids,
            bodies_data,
            &lyt,
            &mut fbc,
            &mut ctx,
            &spec,
        )?;
    }
    module.finalize_definitions()?;

    let bodies = body_ids
        .iter()
        .map(|id| {
            let p = module.get_finalized_function(*id);
            debug_assert!(!p.is_null());
            // SAFETY: the emitted function implements exactly the extern "C"
            // BodyFn signature (`body_sig` above); the module outlives it via
            // `Jit`.
            Some(unsafe { std::mem::transmute::<*const u8, BodyFn>(p) })
        })
        .collect();
    Ok(Jit {
        _module: module,
        bodies,
    })
}

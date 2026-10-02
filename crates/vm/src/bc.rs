//! Surface for bcgen-specialized bodies — the first Futamura projection on this
//! interpreter.
//!
//! A bcgen-emitted module provides one [`BodyFn`] per `BodyId`: a chunk-specialized
//! `step_one` that matches on `code.ip` (a *constant* by construction at codegen
//! time) and either runs the op inline or delegates to [`step`], which is the real
//! interpreter arm. Both produce the same [`Flow`], so calls, returns, register
//! windows, fuel, pauses and snapshots ride `run_dispatch` unchanged — `Vm::run`
//! and `Vm::run_frame` consult the installed table where `Vm::install_bc` put it.
//!
//! The point is that semantics stay single-sourced: unhandled ops execute the
//! interpreter's own code path, so a partially specialized module can never drift
//! from the VM — only from not being faster yet.

use crate::vm::step_one;
use crate::State;
use gc_arena::Mutation;

/// Frame depth at which generated bodies still enter calls inline (push the
/// frame and recurse into the callee's body) instead of returning
/// `Flow::Call` for the driver. Above it, the interpreter-side stack is
/// unbounded but the real Rust stack is not — inline calls stop here and every
/// deeper frame rides the heap-side `Flow::Call` path as before.
///
/// The cap is purely a host-stack/perf knob — inline and driver-dispatched
/// calls are semantically identical — so debug builds run a smaller one:
/// unoptimized body frames are several times fatter and callers (like the
/// test harness) may run on small thread stacks.
#[cfg(not(debug_assertions))]
pub const INLINE_CALL_DEPTH: usize = 256;
/// See the non-debug const above.
#[cfg(debug_assertions)]
pub const INLINE_CALL_DEPTH: usize = 48;

/// One bcgen-specialized body — a chunk-local `step_one`. The driver invokes it
/// with `code.ip` at the body's next unexecuted op and the body's frame on top
/// of `thread.frames`.
///
/// A body loops internally over ops: per op it replicates the dispatch loop's
/// bookkeeping in order — `ctx.state().paused` (when set, return `Ok(Flow::Next)`
/// immediately; the driver syncs the frame's save slot), `fuel` (at 0, return
/// `Ok(Flow::Next)` so the host regains control; else decrement), then
/// `thread.ops_left` (at 0 return `Err(RtErr::OutOfFuel)` with `*op_ip` at the
/// current op; else decrement).
///
/// Calls whose callee has a specialized body are entered *inline* below
/// [`INLINE_CALL_DEPTH`]: the body runs `enter_call` itself, calls the
/// callee's body fn (a module-level `static BODIES` table for dynamic calls,
/// a direct `body_N` call for `CallDirect`), and on `Flow::Return` pops the
/// callee frame, writes the caller's `dst` and resumes — the driver's exact
/// `Flow::Call`/`Flow::Return` handling, kept inside generated code. Deeper
/// calls fall back to returning `Flow::Call` so the driver does it.
/// `Flow::Next`/`Err` from an inlined callee propagate straight to the driver
/// (the callee frame is still on top, exactly as if it had been dispatched
/// itself).
///
/// The body's own register window is `thread.regs[base..base + regs]` like
/// `run_dispatch`'s `window()` — build it from a raw pointer and *rebuild it
/// after every `thread.regs` resize/truncate* (inline `enter_call`, callee
/// returns). `op_ip` is the body's output slot for the faulting op's byte
/// offset — set it to the current op before any `Err` or `Flow::Call` return
/// so the driver locates errors exactly as it does for `step_one`.
/// The driver-facing ABI: `extern "C"` with raw pointers so a JIT-emitted
/// function (mimas-jit/Cranelift) can be installed interchangeably with a
/// bcgen Rust body. The `RtResult<Flow>` result goes through `out` — the enum
/// has no stable layout, so generated/JIT code never *constructs* it: Rust-side
/// trampolines (`bcgen`'s `body_N_abi`) and `jit_*` shims write the slot.
///
/// Contract for an implementation:
/// - every pointer is valid for the call's duration and borrowed from the
///   driver's live structures; `thread`/`code`/`fuel`/`op_ip` may be written,
///   `strs`/`chunks`/`signatures` are read-only;
/// - `out` must be fully written before returning (the driver `assume_init`s
///   it unconditionally);
/// - all the semantic rules in the doc comment above apply unchanged.
///
/// The dispatch-time driver-state bundle every `BodyFn` reads through.
/// Every input is a per-dispatch constant — the same ten words reach
/// every body a `run` dispatches, including each body→body call on the
/// inline fast path — so they ride one `*const`: bodies load the fields
/// they use at entry rather than pinning ten argument registers across
/// the whole function (and re-marshaling them before every nested call).
/// Same pointer contract as the flat params it replaces.
///
/// All fields are pointer-width, `repr(C)` — generated code reads them
/// by `offset_of!`.
#[repr(C)]
pub struct BodyEnv<'gc> {
    pub thread: *mut ThreadState<'gc>,
    pub code: *mut Decoder,
    /// `Ctx`'s first word (`Ctx::from_parts` reassembles it).
    pub mutation: *const Mutation<'gc>,
    /// `Ctx`'s second word.
    pub state: *const State<'gc>,
    pub strs: *const StrInterner,
    pub chunks: *const IdVec<BodyId, Chunk>,
    pub signatures: *const IdVec<BodyId, Option<Function>>,
    pub fuel: *mut usize,
    pub op_ip: *mut usize,
    pub out: *mut RtResult<Flow<'gc>>,
}

/// The driver-facing ABI — see the `BodyFn` contract above. One param:
/// the [`BodyEnv`] bundle.
pub type BodyFn = for<'gc> unsafe extern "C" fn(env: *const BodyEnv<'gc>);

/// The ordinary Rust signature a specialized body's *inner* implementation
/// uses — bcgen bodies call each other at this ABI (no trampoline round-trip
/// on the inlined-call path); each also emits a `BodyFn` extern-"C" shim for
/// the driver table. Not part of any FFI contract.
///
/// The inner convention avoids building the ~`Flow`-sized `RtResult` on the
/// hot return path: `gout` is an out-slot for results that must propagate to
/// the driver (`Flow::Next`/`Err`/`Flow::Call`/a `Flow::Return` seen by an
/// ABI-entered body), and the return tag is `0` when an `inl`-entered callee
/// finished a call itself — its frame already popped, `regs` truncated,
/// `code.ip` restored, and the value written into the caller's `return_reg`
/// slot. `1` means `gout` holds the `RtResult<Flow>` to propagate verbatim.
/// `inl` is `true` only for generated-caller invocations (they pushed the
/// frame); `false` entries always produce tag `1`.
///
/// `qp` chains the batched op quota across an `inl` call so neither side
/// settles `*fuel`/`ops_left` at the boundary: the caller leaves
/// `[armed_baseline, remaining]` in it and the callee runs its `bcn`/`bcn0`
/// pair straight from those values, writing the pair back on a tag-0 exit
/// (tag-1 exits settle the whole chain — caller's pending spend included —
/// before landing in `gout`, so the propagate side reads nothing back).
/// `inl = false` callers pass a scratch cell; it is never read.
pub type InnerBodyFn = for<'gc> fn(
    thread: &mut ThreadState<'gc>,
    code: &mut Decoder,
    ctx: Ctx<'gc>,
    strs: &StrInterner,
    chunks: &IdVec<BodyId, Chunk>,
    signatures: &IdVec<BodyId, Option<Function>>,
    io: &mut GenIo<'_, 'gc>,
) -> u8;

/// The generated-body boundary bundle: every per-call input/output that is
/// not `thread`/`code`/the read-only environment rides one `&mut` so the
/// inner-call ABI stays inside the arg registers — and the caller-to-callee
/// handoff is a couple of stores into this one already-live cell rather than
/// stack-arg marshaling.
///
/// - `out`: the tag-1 `RtResult<Flow>` payload slot (see `InnerBodyFn`);
/// - `fuel`/`op_ip`: the driver's counters — written at body exits exactly
///   where `run_dispatch` would have read them;
/// - `qp`: the `[bcn0, bcn]` quota chain handed to `inl` callees;
/// - `inl`: `true` when a generated caller pushed the frame (the body's own
///   entry mode — a caller re-arms it to `true` before each inline call;
///   bodies cache it into a local at entry).
#[doc(hidden)]
pub struct GenIo<'a, 'gc> {
    /// Tag-1 payload slot.
    pub out: std::mem::MaybeUninit<RtResult<Flow<'gc>>>,
    /// The driver's fuel counter.
    pub fuel: &'a mut usize,
    /// The driver's faulting-op slot.
    pub op_ip: &'a mut usize,
    /// Quota chain `[armed_baseline, remaining]` — see `InnerBodyFn`.
    pub qp: [u64; 2],
    /// Generated-caller entry flag.
    pub inl: bool,
}

/// The callee-side `Flow::Return` for `inl`-entered bodies (`InnerBodyFn`
/// tag-0 exits): pop our frame, truncate `regs` back to the caller's window,
/// restore the caller's saved `code.ip`, and write `rv` straight into its
/// `return_reg` slot — the driver's exact pop sequence, run early so the
/// `RtResult<Flow>` round-trip never happens for generated-caller returns.
///
/// `rv` must already be materialized: it may read the dying window.
#[doc(hidden)]
#[inline]
pub fn gen_return_pop<'gc>(t: &mut ThreadState<'gc>, code: &mut Decoder, rv: Val<'gc>) {
    let popped = t.frames.pop().unwrap();
    t.regs.truncate(popped.base);
    let caller = t.frames.last().unwrap();
    code.ip = caller.ip;
    let caller_base = caller.base;
    t.regs[caller_base + popped.return_reg as usize] = rv;
}

/// The interpreter's own `step_one`, exposed so generated bodies can delegate any
/// op they don't specialize — `code.ip` sits at the op's first byte. Never
/// inlined: the point is sharing one codegen-verbatim implementation.
#[doc(hidden)]
#[inline(never)]
pub fn step<'gc>(
    regs: &mut [Val<'gc>],
    code: &mut Decoder,
    ctx: Ctx<'gc>,
    strs: &StrInterner,
    frames: &[Frame],
) -> RtResult<Flow<'gc>> {
    step_one(regs, code, ctx, strs, frames)
}

/// Register read, mirroring the dispatch loop's `rd!` — register indices are
/// compiler-allocated inside `0..chunk.regs == regs.len()`, so the bounds check
/// is debug-only here too.
#[inline(always)]
pub fn rd<'gc>(regs: &[Val<'gc>], r: Reg) -> Val<'gc> {
    debug_assert!(r.index() < regs.len());
    // SAFETY: register indices are compiler-allocated in 0..chunk.regs == window len.
    unsafe { *regs.get_unchecked(r.index()) }
}

/// Register write, mirroring `wr!` — same invariant as [`rd`].
#[inline(always)]
pub fn wr<'gc>(regs: &mut [Val<'gc>], r: Reg, v: Val<'gc>) {
    debug_assert!(r.index() < regs.len());
    // SAFETY: as in `rd`.
    unsafe { *regs.get_unchecked_mut(r.index()) = v };
}

// Everything a generated module needs to name — one `use` line against the
// embedder's vm path (`mimas::vm::bc::*`, or `vm::bc::*` with a mimas-vm dep).
// The `*_cold` fns are `step_one`'s own cold-path helpers: a specialized op
// whose operand types miss the fast path tail-calls the same helper the
// interpreter would (after pointing `code.ip` past the op, which the
// interpreter's decode had already done).
pub use crate::{
    CallTarget, Closure, Ctx, DebugInfo, DictMap, Fields, Flow, Frame, INLINE_FIELDS, Native,
    RtErr, RtResult, ThreadState, Val, bin, bin_cold, bin_cold_imm_float, bin_cold_imm_int,
    branch_cold, branch_cold_imm_float, branch_cold_imm_int, constant_to_val, contains, enter_call,
    enter_call_regs, get_index, not_callable, seq_push, set_index, unary,
};
pub use api::NativeId;
pub use compile::{
    AccessKind, BinOp, BlockTarget, BodyId, Chunk, Constant, Decoder, Function, Op, OpCode,
    OpFormatPart, Reg, UnaryOp,
};
pub use gc_arena::Gc;
pub use shared::{IdVec, StrId, StrInterner};
pub use smallvec::SmallVec;

/// `extern "C"` helpers for Cranelift-JIT bodies (mimas-jit) — the FFI twin of
/// the helper surface above. Every signature is C-shaped: raw pointers and
/// scalars, with `RtResult<Flow>` and `RtErr` moving through `out` slots JIT
/// code never constructs. `mimas_jit::compile` binds these by address through
/// `JITBuilder::symbol`, so the names/signatures here are the ABI contract.
///
/// Conventions:
/// - `regs` parameters are the *current frame's window base* — `regs[i]` is a
///   relative register index, matching `rd`/`wr` above. JIT bodies must flush
///   their scalar shadows to the window before calling anything that reads it.
/// - `u8` returns: `0` = the op completed like `Flow::Next` (the body may
///   continue to the next op), `1` = `out` holds a `Flow`/`Err` to propagate
///   (return immediately).
/// - `Frame`/`ThreadState`/`Decoder` internals are only ever touched here —
///   JIT code keeps pointers, never layouts.
#[doc(hidden)]
pub mod jit {
    use super::*;
    use crate::State;
    use gc_arena::Mutation;

    // ---- environment pointers (hoisted once per body invocation) ----

    /// `&ctx.state().paused` — the cooperative-pause cell. Stable for the life
    /// of the `Vm`'s arena (the GC never moves `State`).
    pub unsafe extern "C" fn paused_ptr(ctx: Ctx<'_>) -> *const bool {
        ctx.state().paused.as_ptr() as *const bool
    }

    /// `&mut thread.ops_left` — the host op budget.
    pub unsafe extern "C" fn ops_left_ptr(thread: *mut ThreadState<'_>) -> *mut u64 {
        unsafe { &mut (*thread).ops_left }
    }

    /// `&mut code.ip`.
    pub unsafe extern "C" fn ip_ptr(code: *mut Decoder) -> *mut usize {
        unsafe { &mut (*code).ip }
    }

    /// The top frame's register-window base (`frames.last().base`).
    pub unsafe extern "C" fn frame_base(thread: *const ThreadState<'_>) -> usize {
        unsafe { (*thread).frames.last().unwrap().base }
    }

    /// The top frame's register count (`chunks[frame.chunk].regs`).
    pub unsafe extern "C" fn frame_nregs(
        thread: *const ThreadState<'_>,
        chunks: *const IdVec<BodyId, Chunk>,
    ) -> usize {
        unsafe { (&*chunks)[(*thread).frames.last().unwrap().chunk].regs as usize }
    }

    /// `thread.regs.as_mut_ptr()` — re-fetch after any resize (enter_call,
    /// callee return pop).
    pub unsafe extern "C" fn regs_ptr<'gc>(thread: *mut ThreadState<'gc>) -> *mut Val<'gc> {
        unsafe { (*thread).regs.as_mut_ptr() }
    }

    /// `thread.frames.len()` — for the `INLINE_CALL_DEPTH` cap and the
    /// root-frame check on `Return`.
    pub unsafe extern "C" fn frames_len(thread: *const ThreadState<'_>) -> usize {
        unsafe { (*thread).frames.len() }
    }

    // ---- register access (`regs` = window base, `i` = relative index) ----

    /// `regs[i]` as `Val::Int`: writes `*v` and returns 1, else returns 0.
    pub unsafe extern "C" fn ri<'gc>(regs: *const Val<'gc>, i: usize, v: *mut i64) -> u8 {
        unsafe {
            match *regs.add(i) {
                Val::Int(x) => {
                    *v = x;
                    1
                }
                _ => 0,
            }
        }
    }

    /// `regs[i]` as `Val::Float`.
    pub unsafe extern "C" fn rf<'gc>(regs: *const Val<'gc>, i: usize, v: *mut f64) -> u8 {
        unsafe {
            match *regs.add(i) {
                Val::Float(x) => {
                    *v = x;
                    1
                }
                _ => 0,
            }
        }
    }

    /// `regs[i]` as `Val::Bool`.
    pub unsafe extern "C" fn rb<'gc>(regs: *const Val<'gc>, i: usize, v: *mut u8) -> u8 {
        unsafe {
            match *regs.add(i) {
                Val::Bool(x) => {
                    *v = x as u8;
                    1
                }
                _ => 0,
            }
        }
    }

    /// `regs[i] == Val::Bool(b != 0)` — the `JumpIf` condition.
    pub unsafe extern "C" fn is_bool<'gc>(regs: *const Val<'gc>, i: usize, b: u8) -> u8 {
        unsafe { (*regs.add(i) == Val::Bool(b != 0)) as u8 }
    }

    pub unsafe extern "C" fn wr_i<'gc>(regs: *mut Val<'gc>, i: usize, v: i64) {
        unsafe { *regs.add(i) = Val::Int(v) }
    }

    pub unsafe extern "C" fn wr_f<'gc>(regs: *mut Val<'gc>, i: usize, v: f64) {
        unsafe { *regs.add(i) = Val::Float(v) }
    }

    pub unsafe extern "C" fn wr_b<'gc>(regs: *mut Val<'gc>, i: usize, v: u8) {
        unsafe { *regs.add(i) = Val::Bool(v != 0) }
    }

    pub unsafe extern "C" fn wr_null<'gc>(regs: *mut Val<'gc>, i: usize) {
        unsafe { *regs.add(i) = Val::Null }
    }

    pub unsafe extern "C" fn wr_fn<'gc>(regs: *mut Val<'gc>, i: usize, body: u32) {
        unsafe { *regs.add(i) = Val::Fn(BodyId::from(body)) }
    }

    /// `regs[d] = *v`.
    pub unsafe extern "C" fn wr_v<'gc>(regs: *mut Val<'gc>, i: usize, v: *const Val<'gc>) {
        unsafe { *regs.add(i) = *v }
    }

    /// `regs[d] = regs[s]` — the `Move` op.
    pub unsafe extern "C" fn mv<'gc>(regs: *mut Val<'gc>, d: usize, s: usize) {
        unsafe { *regs.add(d) = *regs.add(s) }
    }

    /// `&regs[i]` — a stable read pointer for `out_return`/`raise`.
    pub unsafe extern "C" fn rval<'gc>(regs: *const Val<'gc>, i: usize) -> *const Val<'gc> {
        unsafe { regs.add(i) }
    }

    /// One [`flush`] entry — `#[repr(C)]` so JIT code can pack entries into a
    /// stack slot without the Val layout.
    #[repr(C)]
    pub struct FlushEnt {
        /// Register index in the window.
        pub idx: u32,
        /// 0 = skip (shadow's `ok` flag clear), 1 = `Val::Int`, 2 = `Val::Float`.
        pub tag: u8,
        pub _pad: [u8; 3],
        /// Payload — `f64::to_bits` for `tag == 2`.
        pub val: i64,
    }

    /// Bulk shadow writeback — replaces `n` per-reg `wr_i`/`wr_f` calls with a
    /// single helper: `buf` holds `n` packed entries built by the JIT body;
    /// `tag` 0 skips (shadow dead), 1 writes `Val::Int`, 2 `Val::Float`.
    pub unsafe extern "C" fn flush<'gc>(regs: *mut Val<'gc>, buf: *const FlushEnt, n: usize) {
        unsafe {
            for j in 0..n {
                let e = &*buf.add(j);
                match e.tag {
                    1 => *regs.add(e.idx as usize) = Val::Int(e.val),
                    2 => *regs.add(e.idx as usize) = Val::Float(f64::from_bits(e.val as u64)),
                    _ => {}
                }
            }
        }
    }

    /// Layout facts JIT-emitted code bakes in as immediates so hot paths can
    /// touch `Val`s, `thread.regs`/`frames`/`ops_left` and `code.ip` with raw
    /// loads/stores instead of FFI round-trips. Nothing here is *assumed*:
    /// `Val`'s tag/payload placement and `Vec`'s header order are probed by
    /// inspecting known values of the same build, and the struct offsets come
    /// from `offset_of!` — the compiler's own answer, so a field reorder or a
    /// rustc layout change yields different numbers, not silent breakage.
    /// Anything the probe can't verify panics at compile time.
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct Layout {
        /// `size_of::<Val>` — the window stride (multiple of 8).
        pub val_size: usize,
        /// Byte offset of the discriminant inside a `Val`.
        pub val_tag: usize,
        /// Width of the discriminant in bytes (1/2/4/8).
        pub tag_size: usize,
        /// Byte offset of the `i64`/`f64` payload inside `Val::Int`/`Val::Float`
        /// (the two share the union's 8-byte slot).
        pub val_pay: usize,
        /// Byte offset of the `u8` payload inside `Val::Bool` — rustc is free
        /// to place a 1-byte payload somewhere other than the union's base,
        /// so it's probed separately rather than assumed equal to `val_pay`.
        pub bool_pay: usize,
        /// Byte offset of the `u32` payload inside `Val::Fn` (ditto).
        pub fn_pay: usize,
        /// Discriminant value for `Val::Null` (`tag_size`-byte LE).
        pub t_null: u64,
        /// Discriminant value for `Val::Bool`.
        pub t_bool: u64,
        /// Discriminant value for `Val::Int`.
        pub t_int: u64,
        /// Discriminant value for `Val::Float`.
        pub t_float: u64,
        /// Discriminant value for `Val::Fn`.
        pub t_fn: u64,
        /// `offset_of!(ThreadState, regs)` — a `Vec<Val>` header.
        pub regs_off: usize,
        /// `offset_of!(ThreadState, frames)` — a `Vec<Frame>` header.
        pub frames_off: usize,
        /// `offset_of!(ThreadState, ops_left)`.
        pub ops_left_off: usize,
        /// Byte offset of the buffer pointer inside a `Vec<T>` header.
        pub vec_ptr: usize,
        /// Byte offset of `len` inside a `Vec<T>` header.
        pub vec_len: usize,
        /// `size_of::<Frame>`.
        pub frame_size: usize,
        /// `offset_of!(Frame, base)`.
        pub frame_base: usize,
        /// `offset_of!(Decoder, ip)`.
        pub code_ip: usize,
        /// Byte offset of `cap` inside a `Vec<T>` header — the fast-path
        /// frame push / window grow checks capacity inline and defers to
        /// `Flow::Call` (the driver's own `enter_call`) on a miss.
        pub vec_cap: usize,
        /// `offset_of!(Frame, chunk)` — a `BodyId` (u32) field.
        pub frame_chunk: usize,
        /// `offset_of!(Frame, ip)` — the `usize` resume offset.
        pub frame_ip: usize,
        /// `offset_of!(Frame, return_reg)` — a `u32`.
        pub frame_ret: usize,
        /// `offset_of!(State, paused)` — the pause cell sits directly inside
        /// `State`, reachable from `Ctx`'s second word with a single add —
        /// no FFI hop per body entry.
        pub state_paused: usize,
        /// Discriminant values for the Gc-carrying `Val` variants inlined
        /// by typed container access (tag + payload writes stay
        /// helper-side, but reads of the tag are inline).
        pub t_str: u64,
        /// Discriminant for `Val::Array`.
        pub t_array: u64,
        /// Discriminant for `Val::Instance`.
        pub t_instance: u64,
        /// Byte offset of the `Gc` payload inside `Val::Array` — the pointer
        /// is a plain machine pointer to the `RefLock<Vec<Val>>`.
        pub arr_pay: usize,
        /// Byte offset of the `Gc` payload inside `Val::Instance`
        /// (`RefLock<InstanceData>`).
        pub inst_pay: usize,
        /// Offset of the `Vec<Val>` inside `RefLock<Vec<Val>>` — `Gc` points
        /// straight at the `RefLock`, so `gc + rl_vec` is the Vec header
        /// (probe fields `vec_ptr`/`vec_len`/`vec_cap` apply inside it).
        pub rl_vec: usize,
        /// Offset of the `RefCell` borrow flag (`isize`, `0` unborrowed,
        /// `>0` shared, `<0` mutable) inside `RefLock<Vec<Val>>`. JIT code
        /// may read through the lock; a live mutable borrow is impossible
        /// at an op boundary, so a negative flag routes to `step` (which
        /// panics identically to the interpreter's `borrow()`).
        pub rl_flag: usize,
        /// `offset_of!(InstanceData, struct_id)` — `u32`.
        pub id_sid: usize,
        /// `offset_of!(InstanceData, fields)` — the `Fields` enum.
        pub id_fields: usize,
        /// `offset_of!(RefLock<InstanceData>'s inner InstanceData)` — the
        /// `Gc` payload of `Val::Instance` points at the RefLock.
        pub rl_inst: usize,
        /// The `RefCell` borrow flag's offset inside `RefLock<InstanceData>`
        /// — probed separately from `rl_flag`; rustc is free to order the
        /// `RefCell`'s fields differently per `T`.
        pub rl_flag_i: usize,
        /// `Fields` discriminant offset/width and the tag value selecting
        /// `Fields::Inline` (`Spilled` is the only other variant).
        pub fld_tag: usize,
        /// Discriminant width in bytes for `Fields` (see `fld_tag`).
        pub fld_tsz: usize,
        /// Discriminant value of `Fields::Inline`.
        pub fld_inline: u64,
        /// Byte offset of the `len: u8` field inside `Fields::Inline`.
        pub fld_len: usize,
        /// Byte offset of the `data: [Val; INLINE_FIELDS]` inside
        /// `Fields::Inline`.
        pub fld_data: usize,
        /// Byte offset of the `Vec<Val>` inside `Fields::Spilled`.
        pub fld_spilled: usize,
        /// Byte offset/width of the discriminant inside `RtResult<Flow>`
        /// that selects `Ok(Flow::Return(_))` from every other outcome —
        /// the inlined-call tail classifies `*out` without an FFI hop.
        pub out_tag: usize,
        /// Width in bytes of that discriminant (see `out_tag`).
        pub out_tsz: usize,
        /// The `Ok(Flow::Return(_))` discriminant value.
        pub out_ret: u64,
        /// Byte offset of the returned `Val` inside `Ok(Flow::Return(..))`.
        pub out_ret_pay: usize,
        /// Discriminants for the typed-array `Val` variants (`Seq` payloads)
        /// and `Val::Dict` — the raw-load fast path and frozen-constant
        /// pointer guards compare these.
        pub t_intarray: u64,
        /// See `t_intarray`.
        pub t_floatarray: u64,
        /// See `t_intarray`.
        pub t_dict: u64,
        /// Byte offset of the `Gc` payload inside `Val::IntArray`/
        /// `Val::FloatArray` — a pointer to `RefLock<ArrayStore>`. Both tags
        /// share one payload position (verified by the probe).
        pub seq_pay: usize,
        /// Byte offset of the `Gc` payload inside `Val::Dict`.
        pub dict_pay: usize,
        /// Byte offset of the `Gc` payload inside `Val::Str` — the
        /// baked-key guard compares it.
        pub str_pay: usize,
        /// Offset of the `ArrayStore` inside `RefLock<ArrayStore>` — `Gc`
        /// points at the RefLock, so `gc + rl_seq` is the enum.
        pub rl_seq: usize,
        /// The `RefCell` borrow flag (`isize`) inside `RefLock<ArrayStore>` —
        /// probed per-`T` like `rl_flag`/`rl_flag_i`.
        pub rl_flag_s: usize,
        /// `ArrayStore` discriminant position/width — same semantic
        /// round-trip probe as `Val`'s.
        pub as_tag: usize,
        /// Discriminant width in bytes for `ArrayStore` (see `as_tag`).
        pub as_tsz: usize,
        /// Discriminant value of `ArrayStore::Ints`.
        pub as_ints: u64,
        /// Discriminant value of `ArrayStore::Floats`.
        pub as_floats: u64,
        /// Byte offset of the `Vec<i64>` inside `ArrayStore::Ints` —
        /// `vec_ptr`/`vec_len` apply inside it.
        pub as_ints_vec: usize,
        /// Byte offset of the `Vec<f64>` inside `ArrayStore::Floats`.
        pub as_floats_vec: usize,
        /// Discriminant value of `ArrayStore::Vals`.
        pub as_vals: u64,
        /// Byte offset of the `Array` inside `ArrayStore::Vals` — a `Gc`
        /// pointer to `RefLock<Vec<Val>>`, so `rl_flag`/`rl_vec` apply
        /// through it (one more indirection than the primitive stores).
        pub as_vals_arr: usize,
        /// Discriminant for `Val::Closure` — monomorphic dynamic-call
        /// caches branch on it to reach `ClosureData`.
        pub t_closure: u64,
        /// Byte offset of the `Gc` payload inside `Val::Closure` — a
        /// pointer to `ClosureData` (no `RefLock`).
        pub cl_pay: usize,
        /// `offset_of!(ClosureData, function)` — the `BodyId` (u32) the
        /// monomorphic call IC compares against.
        pub cl_func: usize,
        /// `offset_of!(ClosureData, captures)` — a `Vec<Val>` header
        /// (`vec_ptr`/`vec_len` apply) for the IC's capture copy.
        pub cl_caps: usize,
    }

    /// Probe this build's layouts — see [`Layout`]. Called once per
    /// `mimas_jit::compile`, not from emitted code.
    pub fn layout() -> Layout {
        use crate::val::InstanceData;
        use std::mem::{offset_of, size_of};

        const N: usize = size_of::<Val<'static>>();
        let vsz = N;
        assert_eq!(vsz % 8, 0, "Val size not a multiple of 8");
        assert!(vsz <= 64, "Val unexpectedly large: {vsz}");
        // Raw bytes of a live `Val`. Uninit padding may hold garbage, so
        // nothing is *assumed* from these bytes — every derived offset is
        // verified by a `bake` round-trip through rustc's own decode below.
        fn bytes<'g>(v: &Val<'g>) -> [u8; N] {
            let mut b = [0u8; N];
            unsafe {
                std::ptr::copy_nonoverlapping(v as *const Val<'g> as *const u8, b.as_mut_ptr(), N);
            }
            b
        }
        // Decoded probe result. `Bad` = byte pattern whose discriminant
        // matches no probed variant — how corrupt encodings surface.
        #[derive(PartialEq)]
        enum Pb {
            Bad,
            Null,
            Bool(bool),
            Int(i64),
            Float(f64),
            Fn(u32),
            Other,
        }
        // Raw bytes → which `Val` variant they encode. `ptr::read` (unlike
        // `transmute`) performs no validity check, so a pattern with a
        // corrupt discriminant — e.g. a candidate offset that overlaps the
        // tag — yields `Bad` instead of a hard `invalid value` abort. The
        // payload is matched out only once the variant is identified by its
        // `Discriminant`, at which point the read is fully determined.
        let decode = |raw: [u8; N]| -> Pb {
            let v: Val<'static> = unsafe { std::ptr::read_unaligned(raw.as_ptr().cast()) };
            use std::mem::discriminant;
            let d = discriminant(&v);
            if d == discriminant(&Val::Null) {
                return Pb::Null;
            }
            if d == discriminant(&Val::Bool(false)) {
                return match v {
                    Val::Bool(b) => Pb::Bool(b),
                    _ => Pb::Bad,
                };
            }
            if d == discriminant(&Val::Int(0)) {
                return match v {
                    Val::Int(x) => Pb::Int(x),
                    _ => Pb::Bad,
                };
            }
            if d == discriminant(&Val::Float(0.0)) {
                return match v {
                    Val::Float(x) => Pb::Float(x),
                    _ => Pb::Bad,
                };
            }
            if d == discriminant(&Val::Fn(BodyId::ZERO)) {
                return match v {
                    Val::Fn(x) => Pb::Fn(u32::from(x)),
                    _ => Pb::Bad,
                };
            }
            Pb::Other
        };
        let i0 = bytes(&Val::Int(7));
        let f0 = bytes(&Val::Float(0.0));
        // Discriminant position/width, found by *semantic* round-trip:
        // rewriting a candidate byte range of an `Int`'s encoding with
        // `Float`'s bytes must produce `Float` with the same payload bits;
        // candidates that miss the tag decode as `Int`/`Bad` and are
        // rejected. Probe the *smallest* width first —
        // `size_of::<Discriminant<Val>>` overstates the field width when the
        // enum's tag sits next to padding (rustc then leaves stale bytes
        // there, which would poison an 8-byte tag compare). A tag narrower
        // than the true discriminant is still correct here: every probed
        // variant differs within its low byte(s), and tag writes only need
        // to set those same low bytes — the rest of the discriminant is
        // already `0` in every `Val` encoding this register file can hold.
        let mut found = None;
        'widths: for w in [1usize, 2, 4, 8] {
            if w > vsz {
                continue;
            }
            for p in 0..=vsz - w {
                if i0[p..p + w] == f0[p..p + w] {
                    continue;
                }
                let mut raw = i0;
                raw[p..p + w].copy_from_slice(&f0[p..p + w]);
                if matches!(decode(raw), Pb::Float(x) if x.to_bits() == 7) {
                    assert!(found.is_none(), "ambiguous tag positions for Val");
                    found = Some((p, w));
                }
            }
            if found.is_some() {
                break 'widths;
            }
        }
        let (tag, dsz) = found.expect("no inline discriminant found in Val");
        let tagv = |b: &[u8; N]| -> u64 {
            let mut w = [0u8; 8];
            w[..dsz].copy_from_slice(&b[tag..tag + dsz]);
            u64::from_le_bytes(w)
        };
        let t_int = tagv(&i0);
        let t_float = tagv(&f0);
        // Tag values for the other variants the JIT touches — verified by
        // writing the discriminant onto an Int's encoding and reading it back.
        let b0 = bytes(&Val::Bool(false));
        let null = bytes(&Val::Null);
        let fn0 = bytes(&Val::Fn(BodyId::ZERO));
        let t_bool = tagv(&b0);
        let t_null = tagv(&null);
        let t_fn = tagv(&fn0);
        let wtag = |raw: &mut [u8; N], v: u64| {
            raw[tag..tag + dsz].copy_from_slice(&v.to_le_bytes()[..dsz]);
        };
        {
            let mut raw = i0;
            wtag(&mut raw, t_bool);
            assert!(
                matches!(decode(raw), Pb::Bool(_)),
                "t_bool is not the Bool tag"
            );
            let mut raw = i0;
            wtag(&mut raw, t_null);
            assert!(
                matches!(decode(raw), Pb::Null),
                "t_null is not the Null tag"
            );
            let mut raw = i0;
            wtag(&mut raw, t_fn);
            assert!(matches!(decode(raw), Pb::Fn(_)), "t_fn is not the Fn tag");
        }
        // And no two tags may coincide — the emitted `tag == t_int` checks
        // rely on it.
        for (a, b, name) in [
            (t_int, t_float, "int/float"),
            (t_int, t_bool, "int/bool"),
            (t_int, t_null, "int/null"),
            (t_int, t_fn, "int/fn"),
            (t_float, t_bool, "float/bool"),
            (t_float, t_null, "float/null"),
            (t_float, t_fn, "float/fn"),
            (t_bool, t_null, "bool/null"),
            (t_bool, t_fn, "bool/fn"),
            (t_null, t_fn, "null/fn"),
        ] {
            assert_ne!(a, b, "indistinguishable Val tags: {name}");
        }
        // Payload offsets, again by semantic round-trip: start from a Null's
        // encoding, write the tag + a candidate payload position, and see
        // what value comes back out. Anything rustc puts elsewhere survives
        // unchanged, so only the true payload offset yields a match;
        // candidate ranges overlapping the tag corrupt the discriminant and
        // decode as `Bad`.
        let mut pay = None;
        for p in 0..=vsz - 8 {
            let mut raw = null;
            wtag(&mut raw, t_int);
            raw[p..p + 8].copy_from_slice(&42i64.to_ne_bytes());
            if matches!(decode(raw), Pb::Int(42)) {
                assert!(pay.is_none(), "ambiguous int payload offsets");
                pay = Some(p);
            }
        }
        let pay = pay.expect("no 8-byte int payload found in Val");
        let mut bool_pay = None;
        for p in 0..vsz {
            let mut raw = null;
            wtag(&mut raw, t_bool);
            raw[p] = 1;
            if matches!(decode(raw), Pb::Bool(true)) {
                assert!(bool_pay.is_none(), "ambiguous bool payload offsets");
                bool_pay = Some(p);
            }
        }
        let bool_pay = bool_pay.expect("no bool payload byte found in Val");
        let mut fn_pay = None;
        for p in 0..=vsz - 4 {
            let mut raw = null;
            wtag(&mut raw, t_fn);
            raw[p..p + 4].copy_from_slice(&0x1122_3344u32.to_ne_bytes());
            if matches!(decode(raw), Pb::Fn(0x1122_3344)) {
                assert!(fn_pay.is_none(), "ambiguous fn payload offsets");
                fn_pay = Some(p);
            }
        }
        let fn_pay = fn_pay.expect("no u32 fn payload found in Val");
        // Every emitted write touches tag + payload only; prove that's a
        // complete encoding by checking a read-back on a fourth variant.
        assert_eq!(tagv(&bytes(&Val::Float(1.5))), t_float);
        let mut raw = null;
        wtag(&mut raw, t_float);
        raw[pay..pay + 8].copy_from_slice(&1.5f64.to_ne_bytes());
        assert!(
            matches!(decode(raw), Pb::Float(x) if x == 1.5),
            "float tag+payload write does not round-trip"
        );
        // Vec<T> = three words {ptr, cap, len} in rustc's chosen order —
        // identify each against a vec with unambiguous values.
        let mut v: Vec<u64> = Vec::with_capacity(61);
        v.push(7);
        let words: [usize; 3] = unsafe { std::mem::transmute_copy(&v) };
        let pos = |w: usize| {
            words
                .iter()
                .position(|&x| x == w)
                .unwrap_or_else(|| panic!("Vec word {w:#x} not found in {words:x?}"))
        };
        let (vec_ptr, vec_len, vec_cap) = (
            pos(v.as_ptr() as usize) * 8,
            pos(v.len()) * 8,
            pos(v.capacity()) * 8,
        );
        assert!(
            vec_ptr != vec_len && vec_len != vec_cap && vec_ptr != vec_cap,
            "Vec header words not distinct: ptr={vec_ptr} len={vec_len} cap={vec_cap}"
        );
        assert_eq!(size_of::<BodyId>(), 4, "Frame.chunk assumed u32");
        assert_eq!(size_of::<bool>(), 1);
        // Gc-carrying variants and the containers behind them — probed on
        // live arena objects, not assumed. `Gc` is a plain pointer to T, so
        // `Val::Array`'s payload IS the `RefLock<Vec<Val>>` address.
        let (t_str, t_array, t_instance, arr_pay, inst_pay, rl_vec, rl_flag, rl_inst, rl_flag_i) =
            gc_arena::arena::rootless_mutate(|mc| {
                use crate::val::{Array, Instance, InstanceData, SharedStr, Str};
                let s = Str(Gc::new(mc, SharedStr::from("p")));
                let a = Array(Gc::new(
                    mc,
                    gc_arena::RefLock::new(vec![Val::Int(11), Val::Int(22)]),
                ));
                let i = Instance(Gc::new(
                    mc,
                    gc_arena::RefLock::new(InstanceData {
                        struct_id: 7,
                        fields: Fields::Inline {
                            len: 1,
                            data: [Val::Int(5), Val::Null, Val::Null, Val::Null],
                        },
                    }),
                ));
                let t_str = tagv(&bytes(&Val::Str(s)));
                let t_array = tagv(&bytes(&Val::Array(a)));
                let t_instance = tagv(&bytes(&Val::Instance(i)));
                // payload offset: the variant's Gc pointer is the only
                // machine word in the encoding holding its address
                let pay_of = |v: &Val<'_>, p: usize| -> usize {
                    let b = bytes(v);
                    let want = p.to_ne_bytes();
                    let mut found = None;
                    for off in 0..=vsz - 8 {
                        if b[off..off + 8] == want {
                            assert!(found.is_none(), "ambiguous gc payload offset");
                            found = Some(off);
                        }
                    }
                    found.expect("gc payload not found in Val")
                };
                let arr_pay = pay_of(&Val::Array(a), Gc::as_ptr(a.0) as usize);
                let inst_pay = pay_of(&Val::Instance(i), Gc::as_ptr(i.0) as usize);
                // RefLock<T> = repr(transparent) RefCell<T>; `as_ptr` hands
                // the inner T's address directly.
                let rl: &gc_arena::RefLock<Vec<Val>> = &*a.0;
                let rl_vec = rl.as_ptr() as usize - rl as *const _ as usize;
                let rli: &gc_arena::RefLock<InstanceData> = &*i.0;
                let rl_inst = rli.as_ptr() as usize - rli as *const _ as usize;
                // borrow flag: the word that goes 0 → nonzero on `borrow()`.
                let nwords = size_of::<gc_arena::RefLock<Vec<Val>>>() / 8;
                let words = |p: *const u8| -> Vec<i64> {
                    (0..nwords)
                        .map(|w| unsafe { p.add(w * 8).cast::<i64>().read_unaligned() })
                        .collect()
                };
                let rlb = rl as *const _ as *const u8;
                let before = words(rlb);
                let during = {
                    let r = rl.borrow();
                    let w = words(rlb);
                    drop(r);
                    w
                };
                let mut flag = None;
                for w in 0..nwords {
                    if before[w] == 0 && during[w] == 1 {
                        // one shared borrow: flag == 1; mut borrow is -1
                        assert!(flag.is_none(), "ambiguous borrow flag word");
                        flag = Some(w * 8);
                    }
                }
                let rl_flag = flag.expect("RefCell borrow flag not found");
                // same flag probe for `RefLock<InstanceData>` — `RefCell`'s
                // field order is rustc's to choose per T
                let nwords_i = size_of::<gc_arena::RefLock<InstanceData>>() / 8;
                let rlib = rli as *const _ as *const u8;
                let words_i = |p: *const u8| -> Vec<i64> {
                    (0..nwords_i)
                        .map(|w| unsafe { p.add(w * 8).cast::<i64>().read_unaligned() })
                        .collect()
                };
                let before_i = words_i(rlib);
                let during_i = {
                    let r = rli.borrow();
                    let w = words_i(rlib);
                    drop(r);
                    w
                };
                let mut flag_i = None;
                for w in 0..nwords_i {
                    if before_i[w] == 0 && during_i[w] == 1 {
                        assert!(flag_i.is_none(), "ambiguous borrow flag word");
                        flag_i = Some(w * 8);
                    }
                }
                let rl_flag_i = flag_i.expect("RefCell borrow flag not found");
                (
                    t_str, t_array, t_instance, arr_pay, inst_pay, rl_vec, rl_flag, rl_inst,
                    rl_flag_i,
                )
            });
        // Typed arrays + dicts — `Seq`/`Dict` payloads, the `ArrayStore`
        // enum behind the typed tags, and `Str`'s payload for baked-key
        // guards. Same discipline as above: probes on live arena objects.
        let (
            t_intarray,
            t_floatarray,
            t_dict,
            seq_pay,
            dict_pay,
            str_pay,
            rl_seq,
            rl_flag_s,
            as_tag,
            as_tsz,
            as_ints,
            as_floats,
            as_ints_vec,
            as_floats_vec,
            as_vals,
            as_vals_arr,
            t_closure,
            cl_pay,
        ) = gc_arena::arena::rootless_mutate(|mc| {
            use crate::val::{ArrayStore, Closure, ClosureData, Dict, DictMap, Seq};
            let dg: Dict = Dict(Gc::new(mc, gc_arena::RefLock::new(DictMap::new())));
            let di = Val::Dict(dg);
            let si = Seq(Gc::new(
                mc,
                gc_arena::RefLock::new(ArrayStore::Ints(vec![3, 4])),
            ));
            let sf = Seq(Gc::new(
                mc,
                gc_arena::RefLock::new(ArrayStore::Floats(vec![1.5, 2.5])),
            ));
            let t_intarray = tagv(&bytes(&Val::IntArray(si)));
            let t_floatarray = tagv(&bytes(&Val::FloatArray(sf)));
            let t_dict = tagv(&bytes(&di));
            let pay_of = |v: &Val<'_>, p: usize| -> usize {
                let b = bytes(v);
                let want = p.to_ne_bytes();
                let mut found = None;
                for off in 0..=vsz - 8 {
                    if b[off..off + 8] == want {
                        assert!(found.is_none(), "ambiguous gc payload offset");
                        found = Some(off);
                    }
                }
                found.expect("gc payload not found in Val")
            };
            // Both Seq tags must carry the Gc at one offset — the emitted
            // tag check only selects which primitive store to try.
            let seq_pay = pay_of(&Val::IntArray(si), Gc::as_ptr(si.0) as usize);
            assert_eq!(
                seq_pay,
                pay_of(&Val::FloatArray(sf), Gc::as_ptr(sf.0) as usize),
                "Seq payload must sit at one offset for both tags"
            );
            let dict_pay = pay_of(&di, Gc::as_ptr(dg.0) as usize);
            let sv_str = Val::Str(crate::val::Str(Gc::new(
                mc,
                crate::val::SharedStr::from("k"),
            )));
            let Val::Str(key_str) = sv_str else {
                unreachable!()
            };
            let str_pay = pay_of(&sv_str, Gc::as_ptr(key_str.0) as usize);
            // RefLock<ArrayStore>: inner-enum offset + per-T borrow flag —
            // the same trick `rl_vec`/`rl_flag` used for RefLock<Vec<Val>>.
            let rls = &*si.0;
            let rl_seq = rls.as_ptr() as usize - rls as *const _ as usize;
            let nwords_s = size_of_val(rls) / 8;
            let rlsb = rls as *const _ as *const u8;
            let words_s = |p: *const u8| -> Vec<i64> {
                (0..nwords_s)
                    .map(|w| unsafe { p.add(w * 8).cast::<i64>().read_unaligned() })
                    .collect()
            };
            let before_s = words_s(rlsb);
            let during_s = {
                let r = rls.borrow();
                let w = words_s(rlsb);
                drop(r);
                w
            };
            let mut flag_s = None;
            for w in 0..nwords_s {
                if before_s[w] == 0 && during_s[w] == 1 {
                    assert!(flag_s.is_none(), "ambiguous borrow flag word");
                    flag_s = Some(w * 8);
                }
            }
            let rl_flag_s = flag_s.expect("RefLock<ArrayStore> flag probe failed");
            // `ArrayStore`'s discriminant: rewrite a candidate window of an
            // `Empty`'s encoding with each variant's bytes and require the
            // result's discriminant to be that variant's. `Ints`/`Floats`
            // payloads are Vecs, `Vals`'s is a `Gc`-holding `Array` — all
            // read via ManuallyDrop so nothing fabricated gets dropped.
            const ASZ: usize = size_of::<ArrayStore<'static>>();
            assert_eq!(ASZ % 8, 0, "ArrayStore size not a multiple of 8");
            let abytes = |v: &ArrayStore<'_>| -> [u8; ASZ] {
                let mut b = [0u8; ASZ];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        v as *const ArrayStore<'_> as *const u8,
                        b.as_mut_ptr(),
                        ASZ,
                    );
                }
                b
            };
            let av = ArrayStore::Vals(crate::val::Array(Gc::new(
                mc,
                gc_arena::RefLock::new(vec![Val::Int(1)]),
            )));
            let (ae_b, ai_b, af_b, av_b) = (
                abytes(&ArrayStore::Empty),
                abytes(&ArrayStore::Ints(vec![0])),
                abytes(&ArrayStore::Floats(vec![0.0])),
                abytes(&av),
            );
            let discs = [
                std::mem::discriminant(&ArrayStore::Empty),
                std::mem::discriminant(&ArrayStore::Ints(vec![0])),
                std::mem::discriminant(&ArrayStore::Floats(vec![0.0])),
                std::mem::discriminant(&av),
            ];
            let aclass = |raw: &[u8; ASZ]| -> Option<usize> {
                let v: std::mem::ManuallyDrop<ArrayStore<'_>> =
                    unsafe { std::ptr::read_unaligned(raw.as_ptr().cast()) };
                discs.iter().position(|d| *d == std::mem::discriminant(&*v))
            };
            let mut afound = None;
            'aw: for w in [1usize, 2, 4, 8] {
                if w > ASZ {
                    continue;
                }
                for p in 0..=ASZ - w {
                    let srcs: [&[u8; ASZ]; 4] = [&ae_b, &ai_b, &af_b, &av_b];
                    let mut ok = true;
                    for (j, src) in srcs.iter().enumerate() {
                        let mut raw = ae_b;
                        raw[p..p + w].copy_from_slice(&src[p..p + w]);
                        if aclass(&raw) != Some(j) {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        assert!(afound.is_none(), "ambiguous ArrayStore tag positions");
                        afound = Some((p, w));
                    }
                }
                if afound.is_some() {
                    break 'aw;
                }
            }
            let (as_tag, as_tsz) = afound.expect("no discriminant found in ArrayStore");
            let stagv = |b: &[u8; ASZ]| -> u64 {
                let mut w = [0u8; 8];
                w[..as_tsz].copy_from_slice(&b[as_tag..as_tag + as_tsz]);
                u64::from_le_bytes(w)
            };
            let as_ints = stagv(&abytes(&*si.0.borrow()));
            let as_floats = stagv(&abytes(&*sf.0.borrow()));
            assert_ne!(as_ints, as_floats, "indistinguishable ArrayStore tags");
            // `Vec<T>` payload offsets inside each variant — the `vec_*`
            // header probes apply within them.
            let as_ints_vec = match &*si.0.borrow() {
                ArrayStore::Ints(v) => {
                    let v: &Vec<i64> = v;
                    v as *const Vec<i64> as usize - rls.as_ptr() as usize
                }
                _ => unreachable!(),
            };
            let as_floats_vec = match &*sf.0.borrow() {
                ArrayStore::Floats(v) => {
                    let v: &Vec<f64> = v;
                    v as *const Vec<f64> as usize - (*sf.0).as_ptr() as usize
                }
                _ => unreachable!(),
            };
            // `Vals` holds an `Array` — the emitted code dereferences the
            // `Gc` at this offset, then `rl_flag`/`rl_vec` reach the
            // `Vec<Val>` exactly like a plain `Val::Array` payload.
            let as_vals = stagv(&abytes(&av));
            let as_vals_arr = match &av {
                ArrayStore::Vals(a) => a as *const _ as usize - &av as *const _ as usize,
                _ => unreachable!(),
            };
            // `Val::Closure` — a plain `Gc<ClosureData>` payload; the call
            // IC guards tag + `ClosureData.function` then copies `captures`.
            let cv = Closure(Gc::new(
                mc,
                ClosureData {
                    function: BodyId::ZERO,
                    captures: vec![Val::Int(9)],
                },
            ));
            let t_closure = tagv(&bytes(&Val::Closure(cv)));
            let cl_pay = pay_of(&Val::Closure(cv), Gc::as_ptr(cv.0) as usize);
            (
                t_intarray,
                t_floatarray,
                t_dict,
                seq_pay,
                dict_pay,
                str_pay,
                rl_seq,
                rl_flag_s,
                as_tag,
                as_tsz,
                as_ints,
                as_floats,
                as_ints_vec,
                as_floats_vec,
                as_vals,
                as_vals_arr,
                t_closure,
                cl_pay,
            )
        });
        // sanity: Gc-variant tags differ from everything else probed
        for (a, b, name) in [
            (t_array, t_int, "array/int"),
            (t_array, t_float, "array/float"),
            (t_array, t_bool, "array/bool"),
            (t_array, t_null, "array/null"),
            (t_array, t_fn, "array/fn"),
            (t_array, t_instance, "array/instance"),
            (t_array, t_str, "array/str"),
            (t_instance, t_str, "instance/str"),
            (t_instance, t_null, "instance/null"),
            (t_closure, t_fn, "closure/fn"),
            (t_closure, t_int, "closure/int"),
        ] {
            assert_ne!(a, b, "indistinguishable Val tags: {name}");
        }
        // `Fields` — tag position/width by Inline↔Spilled rewrite; the inner
        // field offsets by direct address arithmetic on each variant.
        const FSZ: usize = size_of::<Fields<'static>>();
        let fbytes = |f: &Fields<'static>| -> [u8; FSZ] {
            let mut b = [0u8; FSZ];
            unsafe {
                std::ptr::copy_nonoverlapping(f as *const _ as *const u8, b.as_mut_ptr(), FSZ);
            }
            b
        };
        let fi = Fields::Inline {
            len: 2,
            data: [Val::Int(9), Val::Null, Val::Null, Val::Null],
        };
        let fs = Fields::Spilled(vec![Val::Int(1)]);
        let d_inline = std::mem::discriminant(&fi);
        let d_spilled = std::mem::discriminant(&fs);
        let (fi_b, fs_b) = (fbytes(&fi), fbytes(&fs));
        let mut fld = None;
        'fw: for w in [1usize, 2, 4, 8] {
            if w > FSZ {
                continue;
            }
            for p in 0..=FSZ - w {
                if fi_b[p..p + w] == fs_b[p..p + w] {
                    continue;
                }
                let mut raw = fi_b;
                raw[p..p + w].copy_from_slice(&fs_b[p..p + w]);
                let v: std::mem::ManuallyDrop<Fields<'static>> =
                    unsafe { std::ptr::read_unaligned(raw.as_ptr().cast()) };
                if std::mem::discriminant(&*v) == d_spilled {
                    assert!(fld.is_none(), "ambiguous Fields tag positions");
                    fld = Some((p, w));
                }
            }
            if fld.is_some() {
                break 'fw;
            }
        }
        let (fld_tag, fld_tsz) = fld.expect("no discriminant found in Fields");
        let fld_inline = {
            let mut w = [0u8; 8];
            w[..fld_tsz].copy_from_slice(&fi_b[fld_tag..fld_tag + fld_tsz]);
            u64::from_le_bytes(w)
        };
        // the rewrite must identify Inline too (tags are total here)
        {
            let mut raw = fs_b;
            raw[fld_tag..fld_tag + fld_tsz].copy_from_slice(&fi_b[fld_tag..fld_tag + fld_tsz]);
            let v: std::mem::ManuallyDrop<Fields<'static>> =
                unsafe { std::ptr::read_unaligned(raw.as_ptr().cast()) };
            assert!(
                std::mem::discriminant(&*v) == d_inline,
                "Fields tag rewrite is not reversible"
            );
        }
        let (fld_len, fld_data) = {
            let faddr = &fi as *const _ as usize;
            let Fields::Inline { len, data } = &fi else {
                unreachable!()
            };
            (
                len as *const u8 as usize - faddr,
                data.as_ptr() as usize - faddr,
            )
        };
        let fld_spilled = {
            let faddr = &fs as *const _ as usize;
            let Fields::Spilled(v) = &fs else {
                unreachable!()
            };
            v as *const Vec<Val> as usize - faddr
        };
        // round-trip check: `data` addressing equals `as_slice` indexing
        assert_eq!(
            (((&fi as *const Fields) as usize + fld_data) as *const Val)
                .align_offset(std::mem::align_of::<Val>()),
            0,
            "Fields::Inline.data is not Val-aligned"
        );
        // `RtResult<Flow>` — the `out` slot's layout. The probe works exactly
        // like `Val`'s: rewrite a candidate tag window of an `Ok(Return)`'s
        // encoding with the other outcomes' bytes and require the result to
        // decode correctly (through `ManuallyDrop` — the enum can hold a
        // `SmallVec` and must not be dropped from a fabricated bit pattern).
        const OSZ: usize = size_of::<RtResult<Flow<'static>>>();
        let obytes = |v: &RtResult<Flow<'static>>| -> [u8; OSZ] {
            let mut b = [0u8; OSZ];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    v as *const RtResult<Flow<'static>> as *const u8,
                    b.as_mut_ptr(),
                    OSZ,
                );
            }
            b
        };
        let oret: RtResult<Flow> = Ok(Flow::Return(Val::Int(0)));
        let onext: RtResult<Flow> = Ok(Flow::Next);
        let oerr: RtResult<Flow> = Err(RtErr::DivByZero);
        let ocall: RtResult<Flow> = Ok(Flow::Call {
            target: CallTarget::Fn(BodyId::ZERO),
            dst: Reg::from(0u32),
            args: smallvec::SmallVec::new(),
        });
        let (oret_b, onext_b, oerr_b, ocall_b) =
            (obytes(&oret), obytes(&onext), obytes(&oerr), obytes(&ocall));
        let oclass = |raw: &[u8; OSZ]| -> u8 {
            let v: std::mem::ManuallyDrop<RtResult<Flow<'static>>> =
                unsafe { std::ptr::read_unaligned(raw.as_ptr().cast()) };
            match &*v {
                Ok(Flow::Next) => 0,
                Ok(Flow::Call { .. }) => 1,
                Ok(Flow::Return(_)) => 2,
                Err(_) => 3,
            }
        };
        let mut ofound = None;
        'ow: for w in [1usize, 2, 4, 8] {
            if w > OSZ {
                continue;
            }
            for p in 0..=OSZ - w {
                if oret_b[p..p + w] == onext_b[p..p + w]
                    && oret_b[p..p + w] == oerr_b[p..p + w]
                    && oret_b[p..p + w] == ocall_b[p..p + w]
                {
                    continue;
                }
                let mut raw = oret_b;
                raw[p..p + w].copy_from_slice(&onext_b[p..p + w]);
                if oclass(&raw) != 0 {
                    continue;
                }
                let mut raw = oret_b;
                raw[p..p + w].copy_from_slice(&oerr_b[p..p + w]);
                if oclass(&raw) != 3 {
                    continue;
                }
                let mut raw = oret_b;
                raw[p..p + w].copy_from_slice(&ocall_b[p..p + w]);
                if oclass(&raw) != 1 {
                    continue;
                }
                // and the return tag written onto a Next must decode as Return
                let mut raw = onext_b;
                raw[p..p + w].copy_from_slice(&oret_b[p..p + w]);
                if oclass(&raw) != 2 {
                    continue;
                }
                assert!(ofound.is_none(), "ambiguous out tag positions");
                ofound = Some((p, w));
            }
            if ofound.is_some() {
                break 'ow;
            }
        }
        let (out_tag, out_tsz) = ofound.expect("no discriminant found in RtResult<Flow>");
        let out_ret = {
            let mut w = [0u8; 8];
            w[..out_tsz].copy_from_slice(&oret_b[out_tag..out_tag + out_tsz]);
            u64::from_le_bytes(w)
        };
        let out_ret_pay = match &oret {
            Ok(Flow::Return(v)) => v as *const Val as usize - &oret as *const _ as usize,
            _ => unreachable!(),
        };
        Layout {
            val_size: vsz,
            val_tag: tag,
            tag_size: dsz,
            val_pay: pay,
            bool_pay,
            fn_pay,
            t_null,
            t_bool,
            t_int,
            t_float,
            t_fn,
            regs_off: offset_of!(ThreadState, regs),
            frames_off: offset_of!(ThreadState, frames),
            ops_left_off: offset_of!(ThreadState, ops_left),
            vec_ptr,
            vec_len,
            frame_size: size_of::<Frame>(),
            frame_base: offset_of!(Frame, base),
            code_ip: offset_of!(Decoder, ip),
            vec_cap,
            frame_chunk: offset_of!(Frame, chunk),
            frame_ip: offset_of!(Frame, ip),
            frame_ret: offset_of!(Frame, return_reg),
            state_paused: offset_of!(State, paused),
            t_str,
            t_array,
            t_instance,
            arr_pay,
            inst_pay,
            rl_vec,
            rl_flag,
            id_sid: offset_of!(InstanceData, struct_id),
            id_fields: offset_of!(InstanceData, fields),
            rl_inst,
            rl_flag_i,
            fld_tag,
            fld_tsz,
            fld_inline,
            fld_len,
            fld_data,
            fld_spilled,
            out_tag,
            out_tsz,
            out_ret,
            out_ret_pay,
            t_intarray,
            t_floatarray,
            t_dict,
            seq_pay,
            dict_pay,
            str_pay,
            rl_seq,
            rl_flag_s,
            as_tag,
            as_tsz,
            as_ints,
            as_floats,
            as_ints_vec,
            as_floats_vec,
            as_vals,
            as_vals_arr,
            t_closure,
            cl_pay,
            cl_func: offset_of!(crate::val::ClosureData, function),
            cl_caps: offset_of!(crate::val::ClosureData, captures),
        }
    }

    // ---- result-slot writers (`out: *mut RtResult<Flow>`) ----

    /// `*out = Ok(Flow::Next)` — pause/fuel exits and plain propagation.
    pub unsafe extern "C" fn out_next<'gc>(out: *mut RtResult<Flow<'gc>>) {
        unsafe { *out = Ok(Flow::Next) }
    }

    /// `*out = Err(<the payload-free RtErr for `kind`>)`.
    pub unsafe extern "C" fn out_err<'gc>(out: *mut RtResult<Flow<'gc>>, kind: u8) {
        let e = match kind {
            0 => RtErr::MatchPanicReached,
            1 => RtErr::DivByZero,
            2 => RtErr::ModByZero,
            3 => RtErr::InvalidShift,
            4 => RtErr::IndexOutOfBounds,
            5 => RtErr::UnwrappedNull,
            6 => RtErr::InvalidUnaryOperand,
            7 => RtErr::IntegerOverflow,
            8 => RtErr::DisplayTooDeep,
            9 => RtErr::UserPanic,
            _ => RtErr::OutOfFuel,
        };
        unsafe { *out = Err(e) }
    }

    /// `*out = Ok(Flow::Return(*v))`.
    pub unsafe extern "C" fn out_return<'gc>(out: *mut RtResult<Flow<'gc>>, v: *const Val<'gc>) {
        unsafe { *out = Ok(Flow::Return(*v)) }
    }

    /// The tag of the `RtResult<Flow>` in `out` — 0 `Next`, 1 `Call`,
    /// 2 `Return`, 3 `Err`. For the inlined-call path: a callee body's result
    /// the caller has to classify before deciding to continue or propagate.
    pub unsafe extern "C" fn out_kind<'gc>(out: *const RtResult<Flow<'gc>>) -> u8 {
        unsafe {
            match &*out {
                Ok(Flow::Next) => 0,
                Ok(Flow::Call { .. }) => 1,
                Ok(Flow::Return(_)) => 2,
                Err(_) => 3,
            }
        }
    }

    /// `*out = Ok(Flow::Call{ .. })` for a `Call` op at the depth cap — the
    /// callee register is re-read so the `CallTarget` keeps its real `Val`
    /// (no bare `Gc` pointers cross the boundary).
    pub unsafe extern "C" fn out_call<'gc>(
        out: *mut RtResult<Flow<'gc>>,
        regs: *const Val<'gc>,
        callee: usize,
        dst: usize,
        args_idx: *const u32,
        nargs: usize,
    ) {
        unsafe {
            let target = match *regs.add(callee) {
                Val::Fn(b) => CallTarget::Value(b),
                Val::Closure(c) => CallTarget::Closure(c),
                other => {
                    *out = Err(not_callable(other));
                    return;
                }
            };
            let mut args = SmallVec::<[Val; 8]>::new();
            for i in 0..nargs {
                args.push(*regs.add(*args_idx.add(i) as usize));
            }
            *out = Ok(Flow::Call {
                target,
                dst: Reg::from(dst as u32),
                args,
            });
        }
    }

    /// `*out = Ok(Flow::Call{ target: CallTarget::Fn(body), .. })` — the
    /// `CallDirect` depth-cap exit.
    pub unsafe extern "C" fn out_call_direct<'gc>(
        out: *mut RtResult<Flow<'gc>>,
        regs: *const Val<'gc>,
        body: u32,
        dst: usize,
        args_idx: *const u32,
        nargs: usize,
    ) {
        unsafe {
            let mut args = SmallVec::<[Val; 8]>::new();
            for i in 0..nargs {
                args.push(*regs.add(*args_idx.add(i) as usize));
            }
            *out = Ok(Flow::Call {
                target: CallTarget::Fn(BodyId::from(body)),
                dst: Reg::from(dst as u32),
                args,
            });
        }
    }

    // ---- interpreter fallback ----

    /// Run one op of the interpreter (`step_one`/`step`) at `code.ip` over the
    /// window `regs[..nregs]` of `thread`'s top frame; write the result to
    /// `out` and return 0 if it was `Flow::Next` (caller continues) or 1
    /// otherwise (caller returns `out` verbatim).
    ///
    /// This is the coverage backstop — any op a JIT body doesn't inline runs
    /// the exact interpreter arm, so semantics can never drift.
    pub unsafe extern "C" fn step_at<'gc>(
        thread: *mut ThreadState<'gc>,
        regs: *mut Val<'gc>,
        nregs: usize,
        code: *mut Decoder,
        ctx: Ctx<'gc>,
        strs: *const StrInterner,
        out: *mut RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            let t = &mut *thread;
            // SAFETY: `regs`/`nregs` are the top frame's window, handed up
            // from the caller that built it exactly like `run_dispatch`'s
            // `window()`; `&t.frames` borrows a disjoint field.
            let window = std::slice::from_raw_parts_mut(regs, nregs);
            *out = step(window, &mut *code, ctx, &*strs, &t.frames);
            // 0 = Flow::Next (caller continues dispatching), 1 = propagate.
            (!matches!(&*out, Ok(Flow::Next))) as u8
        }
    }

    // ---- ops worth a helper rather than full inlining ----
    //
    // Each mirrors its `step_one`/`cold_dispatch` arm verbatim, reading and
    // writing the caller's register window by index. Errors go to `out` with a
    // 1 return (propagate); success writes `regs[dst]` and returns 0.

    /// `Op::Len` — the length of `regs[s]` lands in `*v`; the destination may
    /// be shadowed so the caller writes it (regs or shadow) itself. Errors go
    /// to `out` with a 1 return.
    pub unsafe extern "C" fn len<'gc>(
        regs: *const Val<'gc>,
        s: usize,
        v: *mut i64,
        out: *mut RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            let len = match *regs.add(s) {
                Val::Array(a) => a.0.borrow().len(),
                Val::IntArray(a) | Val::FloatArray(a) => a.0.borrow().len(),
                Val::Dict(d) => d.0.borrow().len(),
                Val::Str(s) => s.as_str().chars().count(),
                Val::Int(i) => i as usize,
                other => {
                    *out = Err(RtErr::Custom(format!(
                        "len: {:?} has no length",
                        other.capture()
                    )));
                    return 1;
                }
            };
            *v = len as i64;
            0
        }
    }

    /// `Op::IsRaised` — `regs[d] = matches!(regs[s], Val::Raised)`.
    pub unsafe extern "C" fn is_raised<'gc>(regs: *mut Val<'gc>, d: usize, s: usize) {
        unsafe {
            let v = *regs.add(s);
            *regs.add(d) = Val::Bool(matches!(v, Val::Raised(_)));
        }
    }

    /// `Op::UnwrapRaised` — `regs[d] = err` for `Val::Raised(err)`.
    pub unsafe extern "C" fn unwrap_raised<'gc>(regs: *mut Val<'gc>, d: usize, s: usize) {
        unsafe {
            let Val::Raised(err) = *regs.add(s) else {
                unreachable!("UnwrapRaised on non-raised value")
            };
            *regs.add(d) = Val::Str(err);
        }
    }

    /// `Op::Unwrap` — `regs[d] = regs[s]`, faulting on `Null`/`Raised`.
    pub unsafe extern "C" fn unwrap<'gc>(
        regs: *mut Val<'gc>,
        d: usize,
        s: usize,
        out: *mut RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            match *regs.add(s) {
                Val::Null => {
                    *out = Err(RtErr::UnwrappedNull);
                    1
                }
                Val::Raised(err) => {
                    *out = Err(RtErr::UnwrappedRaised(err.as_str().to_string()));
                    1
                }
                v => {
                    *regs.add(d) = v;
                    0
                }
            }
        }
    }

    /// `Op::UnwrapUnit` — `regs[d] = Null`, faulting only on `Raised`.
    pub unsafe extern "C" fn unwrap_unit<'gc>(
        regs: *mut Val<'gc>,
        d: usize,
        s: usize,
        out: *mut RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            match *regs.add(s) {
                Val::Raised(err) => {
                    *out = Err(RtErr::UnwrappedRaised(err.as_str().to_string()));
                    1
                }
                _ => {
                    *regs.add(d) = Val::Null;
                    0
                }
            }
        }
    }

    /// `Op::Raise` — `*out = Ok(Flow::Return(Val::Raised(err)))`.
    pub unsafe extern "C" fn raise<'gc>(
        regs: *const Val<'gc>,
        s: usize,
        out: *mut RtResult<Flow<'gc>>,
    ) {
        unsafe {
            let Val::Str(err) = *regs.add(s) else {
                unreachable!("raise on a non-str value")
            };
            *out = Ok(Flow::Return(Val::Raised(err)));
        }
    }

    /// `Op::Bin` — `regs[d] = bin(regs[l], ctx, regs[r], op)`.
    pub unsafe extern "C" fn bin<'gc>(
        regs: *mut Val<'gc>,
        d: usize,
        l: usize,
        op: u8,
        r: usize,
        ctx: Ctx<'gc>,
        out: *mut RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            // SAFETY: `op` was encoded from a `BinOp` (repr(u8)).
            let op: BinOp = std::mem::transmute(op);
            match crate::val::bin(*regs.add(l), ctx, *regs.add(r), op) {
                Ok(v) => {
                    *regs.add(d) = v;
                    0
                }
                Err(e) => {
                    *out = Err(e);
                    1
                }
            }
        }
    }

    /// `Op::Unary` — `regs[d] = unary(regs[s], ctx, op)`.
    pub unsafe extern "C" fn unary<'gc>(
        regs: *mut Val<'gc>,
        d: usize,
        op: u8,
        s: usize,
        ctx: Ctx<'gc>,
        out: *mut RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            let op: UnaryOp = std::mem::transmute(op);
            match crate::val::unary(*regs.add(s), ctx, op) {
                Ok(v) => {
                    *regs.add(d) = v;
                    0
                }
                Err(e) => {
                    *out = Err(e);
                    1
                }
            }
        }
    }

    /// `Op::GetIndex` — `regs[d] = get_index(ctx, regs[s], regs[i], kind)`.
    pub unsafe extern "C" fn get_index<'gc>(
        regs: *mut Val<'gc>,
        d: usize,
        s: usize,
        i: usize,
        kind: u8,
        ctx: Ctx<'gc>,
        out: *mut RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            let kind = if kind == 0 {
                AccessKind::Direct
            } else {
                AccessKind::Option
            };
            match super::get_index(ctx, *regs.add(s), *regs.add(i), kind) {
                Ok(v) => {
                    *regs.add(d) = v;
                    0
                }
                Err(e) => {
                    *out = Err(e);
                    1
                }
            }
        }
    }

    /// `Op::SetIndex`.
    pub unsafe extern "C" fn set_index<'gc>(
        regs: *mut Val<'gc>,
        s: usize,
        i: usize,
        v: usize,
        ctx: Ctx<'gc>,
        out: *mut RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            match super::set_index(ctx, *regs.add(s), *regs.add(i), *regs.add(v)) {
                Ok(()) => 0,
                Err(e) => {
                    *out = Err(e);
                    1
                }
            }
        }
    }

    /// `Op::GetField`.
    pub unsafe extern "C" fn get_field<'gc>(
        regs: *mut Val<'gc>,
        d: usize,
        src: usize,
        slot: usize,
        kind: u8,
        out: *mut RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            let kind = if kind == 0 {
                AccessKind::Direct
            } else {
                AccessKind::Option
            };
            let receiver = *regs.add(src);
            if kind == AccessKind::Option && receiver == Val::Null {
                *regs.add(d) = Val::Null;
                return 0;
            }
            let v = match receiver {
                Val::Instance(i) => i.0.borrow().fields[slot],
                Val::Array(a) => a.0.borrow()[slot],
                Val::IntArray(a) | Val::FloatArray(a) => a.0.borrow().at(slot),
                Val::Null => {
                    *out = Err(RtErr::UnwrappedNull);
                    return 1;
                }
                other => {
                    *out = Err(RtErr::Custom(format!("no fields on {:?}", other.capture())));
                    return 1;
                }
            };
            *regs.add(d) = v;
            0
        }
    }

    /// `Op::SetField`.
    pub unsafe extern "C" fn set_field<'gc>(
        regs: *mut Val<'gc>,
        receiver: usize,
        slot: usize,
        value: usize,
        ctx: Ctx<'gc>,
        out: *mut RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            let receiver = *regs.add(receiver);
            let value = *regs.add(value);
            match receiver {
                Val::Instance(i) => i.0.borrow_mut(&ctx).fields[slot] = value,
                Val::Array(a) => a.0.borrow_mut(&ctx)[slot] = value,
                Val::IntArray(a) | Val::FloatArray(a) => a.0.borrow_mut(&ctx).set(ctx, slot, value),
                Val::Null => {
                    *out = Err(RtErr::UnwrappedNull);
                    return 1;
                }
                other => {
                    *out = Err(RtErr::Custom(format!("no fields on {:?}", other.capture())));
                    return 1;
                }
            }
            0
        }
    }

    /// `Op::Push` — `seq_push(ctx, regs[array], regs[value])`, the shared body
    /// (typed arrays push through `ArrayStore`; `Val::Array` is just the
    /// `Vec<Val>` push it always was).
    pub unsafe extern "C" fn push<'gc>(
        regs: *mut Val<'gc>,
        array: usize,
        value: usize,
        ctx: Ctx<'gc>,
    ) {
        unsafe {
            seq_push(ctx, *regs.add(array), *regs.add(value));
        }
    }

    /// `Op::Insert` — `regs[dict][strs[key]] = regs[value]`.
    pub unsafe extern "C" fn insert<'gc>(
        regs: *mut Val<'gc>,
        dict: usize,
        key: u32,
        value: usize,
        ctx: Ctx<'gc>,
        strs: *const StrInterner,
    ) {
        unsafe {
            let dict = (*regs.add(dict)).as_dict().unwrap();
            let key = ctx.intern((&*strs).get(StrId::from(key)));
            dict.0.borrow_mut(&ctx).insert(key, *regs.add(value));
        }
    }

    /// `Op::In` — `regs[d] = contains(regs[n], regs[h], cond)`.
    pub unsafe extern "C" fn contains_op<'gc>(
        regs: *mut Val<'gc>,
        d: usize,
        needle: usize,
        haystack: usize,
        condition: u8,
    ) {
        unsafe {
            let v = contains(*regs.add(needle), *regs.add(haystack), condition != 0);
            *regs.add(d) = v;
        }
    }

    /// `Op::StrEq`/`Op::StrNe` fast path — when both operands are `Str`, write
    /// `regs[d] = Bool(a == b)` (or `!=` when `eq == 0`) and return 0. Otherwise
    /// return 1 and the caller delegates the whole op to `step`, whose arm
    /// reaches `bin_cold` for the general pair — identical semantics.
    pub unsafe extern "C" fn bin_str<'gc>(
        regs: *mut Val<'gc>,
        d: usize,
        l: usize,
        r: usize,
        eq: u8,
    ) -> u8 {
        unsafe {
            match (*regs.add(l), *regs.add(r)) {
                (Val::Str(a), Val::Str(b)) => {
                    *regs.add(d) = Val::Bool(if eq != 0 { a == b } else { a != b });
                    0
                }
                _ => 1,
            }
        }
    }

    /// `Op::LoadConst` for `Constant::Str` — `regs[d] = intern(strs[id])`.
    /// (Array constants stay on `step`: a `Vec<Constant>` isn't C-shaped.)
    pub unsafe extern "C" fn load_const_str<'gc>(
        regs: *mut Val<'gc>,
        d: usize,
        id: u32,
        ctx: Ctx<'gc>,
        strs: *const StrInterner,
    ) {
        unsafe {
            *regs.add(d) = Val::Str(ctx.intern((&*strs).get(StrId::from(id))));
        }
    }

    /// `Op::IsInstance`.
    pub unsafe extern "C" fn is_instance<'gc>(regs: *mut Val<'gc>, d: usize, s: usize, adt: u32) {
        unsafe {
            let m = matches!(*regs.add(s), Val::Instance(i) if i.0.borrow().struct_id == adt);
            *regs.add(d) = Val::Bool(m);
        }
    }

    /// `Op::NewArray` — `regs[d] = []` (the pending-typed `new_seq` — same
    /// shape `step_one`'s cold arm produces).
    pub unsafe extern "C" fn new_array<'gc>(regs: *mut Val<'gc>, d: usize, ctx: Ctx<'gc>) {
        unsafe {
            *regs.add(d) = ctx.new_seq();
        }
    }

    /// `Op::NewDict` — `regs[d] = ~{}`.
    pub unsafe extern "C" fn new_dict<'gc>(regs: *mut Val<'gc>, d: usize, ctx: Ctx<'gc>) {
        unsafe {
            *regs.add(d) = Val::Dict(ctx.new_dict(DictMap::new()));
        }
    }

    /// `Op::NewInstance` — `fields_idx`/`nfields` index the window's regs.
    pub unsafe extern "C" fn new_instance<'gc>(
        regs: *mut Val<'gc>,
        d: usize,
        adt: u32,
        fields_idx: *const u32,
        nfields: usize,
        ctx: Ctx<'gc>,
    ) {
        unsafe {
            let fields = if nfields <= INLINE_FIELDS {
                let mut data = [Val::Null; INLINE_FIELDS];
                for (i, slot) in data.iter_mut().enumerate().take(nfields) {
                    *slot = *regs.add(*fields_idx.add(i) as usize);
                }
                Fields::Inline {
                    len: nfields as u8,
                    data,
                }
            } else {
                Fields::Spilled(
                    (0..nfields)
                        .map(|i| *regs.add(*fields_idx.add(i) as usize))
                        .collect(),
                )
            };
            *regs.add(d) = Val::Instance(ctx.new_instance(adt, fields));
        }
    }

    /// `Op::NewClosure`.
    pub unsafe extern "C" fn new_closure<'gc>(
        regs: *mut Val<'gc>,
        d: usize,
        body: u32,
        caps_idx: *const u32,
        ncaps: usize,
        ctx: Ctx<'gc>,
    ) {
        unsafe {
            let captures: Vec<Val> = (0..ncaps)
                .map(|i| *regs.add(*caps_idx.add(i) as usize))
                .collect();
            *regs.add(d) = Val::Closure(ctx.new_closure(BodyId::from(body), captures));
        }
    }

    /// `Op::CallNative` — resolves the native, refreshes the DebugInfo stack
    /// mirror (top ip = `code.ip`, which the body has already advanced past
    /// the op), calls, and writes `regs[d]` or the error into `out`.
    /// `ctx` arrives as two raw words (`mc`/`st`), not a by-value `Ctx`: this
    /// shim has enough leading params that a 16-byte composite at that slot
    /// lands wholly on the stack under AAPCS64, while Cranelift's flat
    /// signature would split it — `Ctx::from_parts` reassembles it.
    pub unsafe extern "C" fn call_native<'gc>(
        thread: *mut ThreadState<'gc>,
        regs: *mut Val<'gc>,
        d: usize,
        id: u32,
        args_idx: *const u32,
        nargs: usize,
        code: *mut Decoder,
        mc: *const Mutation<'gc>,
        st: *const State<'gc>,
        out: *mut RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            let ctx = Ctx::from_parts(mc, st);
            let t = &mut *thread;
            let mut args = SmallVec::<[Val; 8]>::new();
            for i in 0..nargs {
                args.push(*regs.add(*args_idx.add(i) as usize));
            }
            let native = {
                let table = ctx.state().natives.borrow();
                *table
                    .get(id as usize)
                    .and_then(|o| o.as_ref())
                    .expect("native id has no installed entry")
            };
            {
                let mut stack = ctx.fixture::<DebugInfo>().stack.borrow_mut();
                stack.clear();
                stack.extend(t.frames.iter().map(|f| (f.chunk, f.ip as u32)));
                if let Some(top) = stack.last_mut() {
                    top.1 = (*code).ip as u32;
                }
            }
            match native.call(ctx, &args) {
                Ok(v) => {
                    *regs.add(d) = v;
                    0
                }
                Err(e) => {
                    *out = Err(e);
                    1
                }
            }
        }
    }

    // ---- calls ----

    /// Resolve a `Call` op's callee register to `(body, captures)`: `Val::Fn`
    /// gets the `CallTarget::Value` signature check (a missing signature is a
    /// `not_callable` error, written to `out`), `Val::Closure` unpacks its
    /// `ClosureData`. Returns 0 on resolve, 1 on error-in-`out`.
    pub unsafe extern "C" fn call_target<'gc>(
        regs: *const Val<'gc>,
        callee: usize,
        signatures: *const IdVec<BodyId, Option<Function>>,
        body: *mut u32,
        caps: *mut *const Val<'gc>,
        ncaps: *mut usize,
        out: *mut RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            match *regs.add(callee) {
                Val::Fn(b) => {
                    if (&*signatures).get(b).and_then(|o| o.as_ref()).is_none() {
                        *out = Err(not_callable(Val::Fn(b)));
                        return 1;
                    }
                    *body = b.index() as u32;
                    *caps = std::ptr::null();
                    *ncaps = 0;
                    0
                }
                Val::Closure(c) => {
                    let d = Gc::as_ref(c.0);
                    *body = d.function.index() as u32;
                    *caps = d.captures.as_ptr();
                    *ncaps = d.captures.len();
                    0
                }
                other => {
                    *out = Err(not_callable(other));
                    1
                }
            }
        }
    }

    /// `enter_call`, FFI-shaped: args are gathered out of the caller window by
    /// register index (so no `Val`s cross the boundary), then the frame push
    /// runs verbatim. Caller must have flushed shadows and set `code.ip` to
    /// the resume offset first — `enter_call` saves it into the caller frame.
    pub unsafe extern "C" fn enter<'gc>(
        thread: *mut ThreadState<'gc>,
        code: *mut Decoder,
        chunks: *const IdVec<BodyId, Chunk>,
        body: u32,
        dst: usize,
        regs: *const Val<'gc>,
        args_idx: *const u32,
        nargs: usize,
        caps: *const Val<'gc>,
        ncaps: usize,
        out: *mut RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            let t = &mut *thread;
            let mut args = SmallVec::<[Val; 8]>::new();
            for i in 0..nargs {
                args.push(*regs.add(*args_idx.add(i) as usize));
            }
            let captures = if caps.is_null() {
                &[][..]
            } else {
                std::slice::from_raw_parts(caps, ncaps)
            };
            match enter_call(
                t,
                &mut *code,
                &*chunks,
                BodyId::from(body),
                Reg::from(dst as u32),
                &args,
                captures,
            ) {
                Ok(()) => 0,
                Err(e) => {
                    *out = Err(e);
                    1
                }
            }
        }
    }

    /// The driver's `Flow::Return` handling after an inlined callee body
    /// returns: pop the callee frame, truncate `regs`, restore the caller's
    /// saved ip, write the return value into `return_reg`. Returns the
    /// `out_kind` tag — 2 means the pop ran and the caller may resume.
    pub unsafe extern "C" fn pop_return<'gc>(
        thread: *mut ThreadState<'gc>,
        code: *mut Decoder,
        out: *const RtResult<Flow<'gc>>,
    ) -> u8 {
        unsafe {
            let t = &mut *thread;
            let Ok(Flow::Return(v)) = &*out else {
                return out_kind(out);
            };
            let popped = t.frames.pop().unwrap();
            t.regs.truncate(popped.base);
            let caller = t.frames.last().unwrap();
            (*code).ip = caller.ip;
            let base = caller.base;
            t.regs[base + popped.return_reg as usize] = *v;
            2
        }
    }

    // ---- inline calls: resolve + enter + run + pop in one FFI hop ----

    /// Shared tail of `call_body`/`call_dyn`: the `enter_call_regs` frame
    /// push — args copied straight out of the caller window by register index
    /// — then invoke the callee's `BodyFn`, and on `Flow::Return` run the
    /// driver's pop/truncate/ip-restore/`dst`-write. Returns the caller's
    /// rebuilt window base so JIT code can re-pin it, or null to propagate
    /// `out` verbatim (paused/`Flow::Call`/`Err`, or a failed arity check).
    ///
    /// `code.ip` must already hold the caller's resume offset — it becomes
    /// the caller frame's saved ip exactly like `enter_call`.
    unsafe fn enter_run_pop<'gc>(
        env: *const BodyEnv<'gc>,
        body: BodyId,
        dst: u32,
        args_idx: *const u32,
        nargs: usize,
        captures: &[Val<'gc>],
        f: BodyFn,
    ) -> *mut Val<'gc> {
        unsafe {
            let e = &*env;
            let t = &mut *e.thread;
            let chunk = &(&*e.chunks)[body];
            let out = e.out;
            // same arity check `enter_call` would run on the driver's
            // `Flow::Call` path — `*op_ip` already sits at this op
            if nargs != chunk.args as usize {
                *out = Err(RtErr::WrongArity {
                    wanted: chunk.args as usize,
                    got: nargs,
                });
                return std::ptr::null_mut();
            }
            debug_assert_eq!(captures.len(), chunk.captures.len());
            let caller_base = t.frames.last().unwrap().base;
            let new_base = t.regs.len();
            t.regs.resize(new_base + chunk.regs as usize, Val::Null);
            // caller slots stay live below `new_base` across the grow — the
            // `enter_call_regs` idiom
            for (param_reg, i) in chunk.params.iter().zip(0..nargs) {
                let ai = *args_idx.add(i) as usize;
                t.regs[new_base + param_reg.index()] = t.regs[caller_base + ai];
            }
            for (cap_reg, &cap) in chunk.captures.iter().zip(captures) {
                t.regs[new_base + cap_reg.index()] = cap;
            }
            t.frames.last_mut().unwrap().ip = (*e.code).ip;
            t.frames.push(Frame {
                chunk: body,
                ip: chunk.offset,
                return_reg: dst,
                base: new_base,
            });
            (*e.code).ip = chunk.offset;
            f(env);
            let Ok(Flow::Return(v)) = &*out else {
                return std::ptr::null_mut();
            };
            let v = *v;
            let popped = t.frames.pop().unwrap();
            t.regs.truncate(popped.base);
            let caller = t.frames.last().unwrap();
            (*e.code).ip = caller.ip;
            let caller_base = caller.base;
            t.regs[caller_base + popped.return_reg as usize] = v;
            t.regs.as_mut_ptr().add(caller_base)
        }
    }

    /// `Op::CallDirect`'s whole call path in one FFI hop: at/over
    /// `INLINE_CALL_DEPTH` produces `Flow::Call{CallTarget::Fn}` for the
    /// driver (its `enter_call` then runs the arity check, so none here);
    /// below the cap, enters the callee frame, calls its body fn looked up
    /// through `tbl` (the JIT module's body-pointer table), and pops on
    /// `Flow::Return`. Returns the rebuilt caller-window base pointer, or
    /// null to propagate `out` to the driver.
    pub unsafe extern "C" fn call_body<'gc>(
        env: *const BodyEnv<'gc>,
        tbl: *const usize,
        body: u32,
        dst: u32,
        args_idx: *const u32,
        nargs: usize,
    ) -> *mut Val<'gc> {
        unsafe {
            let e = &*env;
            let t = &mut *e.thread;
            if t.frames.len() >= INLINE_CALL_DEPTH {
                let caller_base = t.frames.last().unwrap().base;
                let mut args = SmallVec::<[Val; 8]>::new();
                for i in 0..nargs {
                    args.push(t.regs[caller_base + *args_idx.add(i) as usize]);
                }
                *e.out = Ok(Flow::Call {
                    target: CallTarget::Fn(BodyId::from(body)),
                    dst: Reg::from(dst),
                    args,
                });
                return std::ptr::null_mut();
            }
            // SAFETY: `tbl` is the JIT module's `BodyFn` table, indexed by body.
            let f: BodyFn = std::mem::transmute(*tbl.add(body as usize));
            enter_run_pop(
                env,
                BodyId::from(body),
                dst,
                args_idx,
                nargs,
                &[],
                f,
            )
        }
    }

    /// `Op::Call`'s whole call path in one FFI hop: resolve `regs[callee]`
    /// (`Val::Fn` gets the `CallTarget::Value` signature check, `Val::Closure`
    /// unpacks its `ClosureData`, anything else is `not_callable` into `out`),
    /// then the same cap/enter/run/pop as `call_body`. `regs` is the caller's
    /// (flushed) window — read before any `thread.regs` resize.
    pub unsafe extern "C" fn call_dyn<'gc>(
        env: *const BodyEnv<'gc>,
        tbl: *const usize,
        regs: *const Val<'gc>,
        callee: usize,
        dst: u32,
        args_idx: *const u32,
        nargs: usize,
    ) -> *mut Val<'gc> {
        unsafe {
            let e = &*env;
            let t = &mut *e.thread;
            let cv = *regs.add(callee);
            let (body, captures): (BodyId, &[Val<'gc>]) = match cv {
                Val::Fn(b) => {
                    if (&*e.signatures).get(b).and_then(|o| o.as_ref()).is_none() {
                        *e.out = Err(not_callable(cv));
                        return std::ptr::null_mut();
                    }
                    (b, &[][..])
                }
                Val::Closure(c) => {
                    let d = Gc::as_ref(c.0);
                    (d.function, d.captures.as_slice())
                }
                other => {
                    *e.out = Err(not_callable(other));
                    return std::ptr::null_mut();
                }
            };
            if t.frames.len() >= INLINE_CALL_DEPTH {
                let target = match cv {
                    Val::Fn(b) => CallTarget::Value(b),
                    Val::Closure(c) => CallTarget::Closure(c),
                    _ => unreachable!(),
                };
                let mut args = SmallVec::<[Val; 8]>::new();
                for i in 0..nargs {
                    args.push(*regs.add(*args_idx.add(i) as usize));
                }
                *e.out = Ok(Flow::Call {
                    target,
                    dst: Reg::from(dst),
                    args,
                });
                return std::ptr::null_mut();
            }
            // SAFETY: `tbl` is the JIT module's `BodyFn` table, indexed by body.
            let f: BodyFn = std::mem::transmute(*tbl.add(body.index()));
            enter_run_pop(env, body, dst, args_idx, nargs, captures, f)
        }
    }
}

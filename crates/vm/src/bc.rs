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

/// One bcgen-specialized body — a chunk-local `step_one`, called per op from
/// `run_dispatch` with `code.ip` at the op's first byte.
///
/// A body may run exactly one op and return, or loop internally over several:
/// `code.ip` is left at the *next unexecuted* op either way, so the driver can
/// always re-enter. Bodies that loop must replicate the dispatch loop's per-op
/// bookkeeping, in the same order — `ctx.state().paused` (when set, return
/// `Ok(Flow::Next)` immediately; the driver syncs the frame's save slot), `fuel`
/// (at 0, return `Ok(Flow::Next)` so the host regains control; else decrement),
/// then `ops_left` (at 0 return `Err(RtErr::OutOfFuel)` with `*op_ip` at the
/// current op; else decrement). `Flow::Call`/`Flow::Return` end the body's turn
/// regardless.
///
/// `op_ip` is the body's output slot for the faulting op's byte offset — set it
/// to the current op before any `Err` or `Flow::Call` return so the driver
/// locates errors exactly as it does for `step_one`.
pub type BodyFn = for<'gc> fn(
    regs: &mut [Val<'gc>],
    code: &mut Decoder,
    ctx: Ctx<'gc>,
    strs: &StrInterner,
    frames: &[Frame],
    ops_left: &mut u64,
    fuel: &mut usize,
    op_ip: &mut usize,
) -> RtResult<Flow<'gc>>;

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
    RtErr, RtResult, Val, bin, bin_cold, bin_cold_imm_float, bin_cold_imm_int, branch_cold,
    branch_cold_imm_float, branch_cold_imm_int, constant_to_val, contains, get_index, not_callable,
    set_index, unary,
};
pub use api::NativeId;
pub use compile::{
    AccessKind, BinOp, BlockTarget, BodyId, Constant, Decoder, Op, OpCode, OpFormatPart, Reg,
    UnaryOp,
};
pub use shared::{StrId, StrInterner};
pub use smallvec::SmallVec;

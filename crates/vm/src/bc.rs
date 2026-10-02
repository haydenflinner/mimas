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
pub type BodyFn = for<'gc> unsafe extern "C" fn(
    thread: *mut ThreadState<'gc>,
    code: *mut Decoder,
    ctx: Ctx<'gc>,
    strs: *const StrInterner,
    chunks: *const IdVec<BodyId, Chunk>,
    signatures: *const IdVec<BodyId, Option<Function>>,
    fuel: *mut usize,
    op_ip: *mut usize,
    out: *mut RtResult<Flow<'gc>>,
);

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
pub type InnerBodyFn = for<'gc> fn(
    thread: &mut ThreadState<'gc>,
    code: &mut Decoder,
    ctx: Ctx<'gc>,
    strs: &StrInterner,
    chunks: &IdVec<BodyId, Chunk>,
    signatures: &IdVec<BodyId, Option<Function>>,
    fuel: &mut usize,
    op_ip: &mut usize,
    gout: &mut std::mem::MaybeUninit<RtResult<Flow<'gc>>>,
    inl: bool,
) -> u8;

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
    enter_call_regs, get_index, not_callable, set_index, unary,
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
    }

    /// Probe this build's layouts — see [`Layout`]. Called once per
    /// `mimas_jit::compile`, not from emitted code.
    pub fn layout() -> Layout {
        use std::mem::{offset_of, size_of};

        const N: usize = size_of::<Val<'static>>();
        let vsz = N;
        assert_eq!(vsz % 8, 0, "Val size not a multiple of 8");
        assert!(vsz <= 64, "Val unexpectedly large: {vsz}");
        // Raw bytes of a live `Val`. Uninit padding may hold garbage, so
        // nothing is *assumed* from these bytes — every derived offset is
        // verified by a `bake` round-trip through rustc's own decode below.
        let bytes = |v: &Val<'static>| -> [u8; N] {
            let mut b = [0u8; N];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    v as *const Val<'static> as *const u8,
                    b.as_mut_ptr(),
                    N,
                );
            }
            b
        };
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

    /// `Op::Push` — `regs[array].as_array().push(regs[value])`.
    pub unsafe extern "C" fn push<'gc>(
        regs: *mut Val<'gc>,
        array: usize,
        value: usize,
        ctx: Ctx<'gc>,
    ) {
        unsafe {
            let arr = (*regs.add(array)).as_array().unwrap();
            arr.0.borrow_mut(&ctx).push(*regs.add(value));
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

    /// `Op::NewArray` — `regs[d] = []`.
    pub unsafe extern "C" fn new_array<'gc>(regs: *mut Val<'gc>, d: usize, ctx: Ctx<'gc>) {
        unsafe {
            *regs.add(d) = Val::Array(ctx.new_array(Vec::new()));
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
    #[allow(clippy::too_many_arguments)]
    unsafe fn enter_run_pop<'gc>(
        t: &mut ThreadState<'gc>,
        code: *mut Decoder,
        chunks: *const IdVec<BodyId, Chunk>,
        body: BodyId,
        dst: u32,
        args_idx: *const u32,
        nargs: usize,
        captures: &[Val<'gc>],
        f: BodyFn,
        ctx: Ctx<'gc>,
        strs: *const StrInterner,
        sigs: *const IdVec<BodyId, Option<Function>>,
        fuel: *mut usize,
        op_ip: *mut usize,
        out: *mut RtResult<Flow<'gc>>,
    ) -> *mut Val<'gc> {
        unsafe {
            let chunk = &(&*chunks)[body];
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
            t.frames.last_mut().unwrap().ip = (*code).ip;
            t.frames.push(Frame {
                chunk: body,
                ip: chunk.offset,
                return_reg: dst,
                base: new_base,
            });
            (*code).ip = chunk.offset;
            f(t, code, ctx, strs, chunks, sigs, fuel, op_ip, out);
            let Ok(Flow::Return(v)) = &*out else {
                return std::ptr::null_mut();
            };
            let v = *v;
            let popped = t.frames.pop().unwrap();
            t.regs.truncate(popped.base);
            let caller = t.frames.last().unwrap();
            (*code).ip = caller.ip;
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
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn call_body<'gc>(
        thread: *mut ThreadState<'gc>,
        code: *mut Decoder,
        chunks: *const IdVec<BodyId, Chunk>,
        tbl: *const usize,
        body: u32,
        dst: u32,
        args_idx: *const u32,
        nargs: usize,
        mc: *const Mutation<'gc>,
        st: *const State<'gc>,
        strs: *const StrInterner,
        sigs: *const IdVec<BodyId, Option<Function>>,
        fuel: *mut usize,
        op_ip: *mut usize,
        out: *mut RtResult<Flow<'gc>>,
    ) -> *mut Val<'gc> {
        unsafe {
            let t = &mut *thread;
            if t.frames.len() >= INLINE_CALL_DEPTH {
                let caller_base = t.frames.last().unwrap().base;
                let mut args = SmallVec::<[Val; 8]>::new();
                for i in 0..nargs {
                    args.push(t.regs[caller_base + *args_idx.add(i) as usize]);
                }
                *out = Ok(Flow::Call {
                    target: CallTarget::Fn(BodyId::from(body)),
                    dst: Reg::from(dst),
                    args,
                });
                return std::ptr::null_mut();
            }
            // SAFETY: `tbl` is the JIT module's `BodyFn` table, indexed by body.
            let f: BodyFn = std::mem::transmute(*tbl.add(body as usize));
            enter_run_pop(
                t,
                code,
                chunks,
                BodyId::from(body),
                dst,
                args_idx,
                nargs,
                &[],
                f,
                Ctx::from_parts(mc, st),
                strs,
                sigs,
                fuel,
                op_ip,
                out,
            )
        }
    }

    /// `Op::Call`'s whole call path in one FFI hop: resolve `regs[callee]`
    /// (`Val::Fn` gets the `CallTarget::Value` signature check, `Val::Closure`
    /// unpacks its `ClosureData`, anything else is `not_callable` into `out`),
    /// then the same cap/enter/run/pop as `call_body`. `regs` is the caller's
    /// (flushed) window — read before any `thread.regs` resize.
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn call_dyn<'gc>(
        thread: *mut ThreadState<'gc>,
        code: *mut Decoder,
        chunks: *const IdVec<BodyId, Chunk>,
        sigs: *const IdVec<BodyId, Option<Function>>,
        regs: *const Val<'gc>,
        callee: usize,
        dst: u32,
        args_idx: *const u32,
        nargs: usize,
        tbl: *const usize,
        mc: *const Mutation<'gc>,
        st: *const State<'gc>,
        strs: *const StrInterner,
        fuel: *mut usize,
        op_ip: *mut usize,
        out: *mut RtResult<Flow<'gc>>,
    ) -> *mut Val<'gc> {
        unsafe {
            let t = &mut *thread;
            let cv = *regs.add(callee);
            let (body, captures): (BodyId, &[Val<'gc>]) = match cv {
                Val::Fn(b) => {
                    if (&*sigs).get(b).and_then(|o| o.as_ref()).is_none() {
                        *out = Err(not_callable(cv));
                        return std::ptr::null_mut();
                    }
                    (b, &[][..])
                }
                Val::Closure(c) => {
                    let d = Gc::as_ref(c.0);
                    (d.function, d.captures.as_slice())
                }
                other => {
                    *out = Err(not_callable(other));
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
                *out = Ok(Flow::Call {
                    target,
                    dst: Reg::from(dst),
                    args,
                });
                return std::ptr::null_mut();
            }
            // SAFETY: `tbl` is the JIT module's `BodyFn` table, indexed by body.
            let f: BodyFn = std::mem::transmute(*tbl.add(body.index()));
            enter_run_pop(
                t,
                code,
                chunks,
                body,
                dst,
                args_idx,
                nargs,
                captures,
                f,
                Ctx::from_parts(mc, st),
                strs,
                sigs,
                fuel,
                op_ip,
                out,
            )
        }
    }
}

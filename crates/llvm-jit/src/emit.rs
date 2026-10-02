//! Per-chunk LLVM-IR emission — the JIT twin of `mimas-jit`'s `emit.rs` and
//! `bcgen`'s `emit_op`, same block layout:
//!
//! ```text
//! entry:   hoist env pointers, init scalar shadows from the reg window
//! dispatch: code.ip - chunk.offset → `switch` over op byte offsets
//! op_i:    [paused/fuel/ops_left preamble] → op semantics → jump successor
//! writeback: every observable boundary inlines the shadow writeback
//!          (`flush_seq`) so the window is authoritative before helpers,
//!          calls, and exits
//! exits:   enext (pause/fuel) / eoof (ops budget) / estep (interpreter
//!          fallback) / eerr (RtErr) / eret (propagate `out`)
//! ```
//!
//! Differences from the Cranelift emitter: mutable state (`regs`, `cur_ip`,
//! `ekind`, `bcn`/`bcn0`, the `(sv, ok)` shadow pairs) lives in entry-block
//! allocas and `default<O2>`'s mem2reg promotes them to SSA — no manual
//! `def_var`/`use_var`. The dispatcher is a sparse `switch` on
//! `code.ip - chunk.offset` instead of an `ip2idx` blob + `br_table`.
//! Helper binding is by absolute address through
//! `ExecutionEngine::add_global_mapping`. Reg-index lists (call args,
//! captures, fields) are private constant globals, not stack spills.
//! Container fast paths and the `CallDirect` inline-frame path are *not*
//! ported — those ops go straight to their `mj_*` megashim, which still
//! inlines the call through the bodies table.

use std::collections::{HashMap, HashSet};

use compile::{BlockTarget, Constant, Op, Program, Reg};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::types::{BasicMetadataTypeEnum, BasicTypeEnum, FunctionType, IntType};
use inkwell::values::{BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue, FloatValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel};
use vm::bc::{BodyFn, jit, jit::Layout};

use crate::{Error, H, Jit, Pt, Hr, SPECS};

/// `RtErr` kind codes matching `jit::out_err`'s table.
const ERR_PANIC: i64 = 0; // MatchPanicReached
const ERR_MOD0: i64 = 2; // ModByZero
const ERR_OVFW: i64 = 7; // IntegerOverflow
const ERR_OOF: i64 = 10; // OutOfFuel

/// Which scalar kind every writer of a register provably produces — bcgen's
/// `W`, verbatim.
#[derive(Clone, Copy, PartialEq)]
enum W {
    Int,
    Float,
    Copy(u32),
    /// `GetIndex`/`GetField` writes a dynamically-typed value, but the
    /// `helper_op_read` path refreshes the shadow `(sv, ok)` pair from the
    /// written slot — so the write is compatible with either scalar shadow
    /// *if* the reg is actually read as that scalar.
    DynCheck,
    Dyn,
}

/// bcgen's `analyze`, verbatim: the set of registers eligible for an int/float
/// SSA shadow — every writer is that scalar kind or a `Copy` of a member.
struct Sh {
    int: HashSet<u32>,
    float: HashSet<u32>,
}

fn analyze(ops: &[(usize, Op)], nregs: u32) -> Sh {
    let mut writes: HashMap<u32, Vec<W>> = HashMap::new();
    let mut put = |r: Reg, w: W| writes.entry(r.index() as u32).or_default().push(w);
    let mut int_reads: HashSet<u32> = HashSet::new();
    let mut float_reads: HashSet<u32> = HashSet::new();
    let mut iread = |r: Reg| {
        int_reads.insert(r.index() as u32);
    };
    let mut fread = |r: Reg| {
        float_reads.insert(r.index() as u32);
    };
    for (_, op) in ops {
        match op {
            Op::Move { dst, src } => put(*dst, W::Copy(src.index() as u32)),
            Op::LoadConst { dst, constant } => put(
                *dst,
                match constant {
                    Constant::Int(_) => W::Int,
                    Constant::Float(_) => W::Float,
                    _ => W::Dyn,
                },
            ),
            Op::AddInt { dst, left, right }
            | Op::SubInt { dst, left, right }
            | Op::MultInt { dst, left, right }
            | Op::ModInt { dst, left, right }
            | Op::IntLt { dst, left, right }
            | Op::IntLe { dst, left, right }
            | Op::IntGt { dst, left, right }
            | Op::IntGe { dst, left, right }
            | Op::IntEq { dst, left, right }
            | Op::IntNe { dst, left, right } => {
                iread(*left);
                iread(*right);
                put(
                    *dst,
                    if matches!(
                        op,
                        Op::AddInt { .. }
                            | Op::SubInt { .. }
                            | Op::MultInt { .. }
                            | Op::ModInt { .. }
                    ) {
                        W::Int
                    } else {
                        W::Dyn
                    },
                );
            }
            Op::AddIntImm { dst, left, .. }
            | Op::SubIntImm { dst, left, .. }
            | Op::MultIntImm { dst, left, .. }
            | Op::ModIntImm { dst, left, .. } => {
                iread(*left);
                put(*dst, W::Int);
            }
            Op::IntLtImm { dst, left, .. }
            | Op::IntLeImm { dst, left, .. }
            | Op::IntGtImm { dst, left, .. }
            | Op::IntGeImm { dst, left, .. }
            | Op::IntEqImm { dst, left, .. }
            | Op::IntNeImm { dst, left, .. } => {
                iread(*left);
                put(*dst, W::Dyn);
            }
            Op::AddFloat { dst, left, right }
            | Op::SubFloat { dst, left, right }
            | Op::MultFloat { dst, left, right }
            | Op::DivFloat { dst, left, right } => {
                fread(*left);
                fread(*right);
                put(*dst, W::Float);
            }
            Op::FloatLt { dst, left, right }
            | Op::FloatLe { dst, left, right }
            | Op::FloatGt { dst, left, right }
            | Op::FloatGe { dst, left, right }
            | Op::FloatEq { dst, left, right }
            | Op::FloatNe { dst, left, right } => {
                fread(*left);
                fread(*right);
                put(*dst, W::Dyn);
            }
            Op::AddFloatImm { dst, left, .. }
            | Op::SubFloatImm { dst, left, .. }
            | Op::MultFloatImm { dst, left, .. }
            | Op::ModFloatImm { dst, left, .. } => {
                fread(*left);
                put(*dst, W::Float);
            }
            Op::FloatLtImm { dst, left, .. }
            | Op::FloatLeImm { dst, left, .. }
            | Op::FloatGtImm { dst, left, .. }
            | Op::FloatGeImm { dst, left, .. }
            | Op::FloatEqImm { dst, left, .. }
            | Op::FloatNeImm { dst, left, .. } => {
                fread(*left);
                put(*dst, W::Dyn);
            }
            Op::Len { dst, .. } => put(*dst, W::Int),
            Op::ToFloat { dst, src } => {
                iread(*src);
                put(*dst, W::Float);
            }
            Op::Sqrt { dst, src } => {
                fread(*src);
                put(*dst, W::Float);
            }
            Op::ForNext { idx, bound, .. } => {
                iread(*idx);
                iread(*bound);
                put(*idx, W::Int);
            }
            Op::BIntLt { left, right, .. }
            | Op::BIntLe { left, right, .. }
            | Op::BIntGt { left, right, .. }
            | Op::BIntGe { left, right, .. }
            | Op::BIntEq { left, right, .. }
            | Op::BIntNe { left, right, .. } => {
                iread(*left);
                iread(*right);
            }
            Op::BIntLtImm { left, .. }
            | Op::BIntLeImm { left, .. }
            | Op::BIntGtImm { left, .. }
            | Op::BIntGeImm { left, .. }
            | Op::BIntEqImm { left, .. }
            | Op::BIntNeImm { left, .. } => iread(*left),
            Op::BFloatLt { left, right, .. }
            | Op::BFloatLe { left, right, .. }
            | Op::BFloatGt { left, right, .. }
            | Op::BFloatGe { left, right, .. }
            | Op::BFloatEq { left, right, .. }
            | Op::BFloatNe { left, right, .. } => {
                fread(*left);
                fread(*right);
            }
            Op::BFloatLtImm { left, .. }
            | Op::BFloatLeImm { left, .. }
            | Op::BFloatGtImm { left, .. }
            | Op::BFloatGeImm { left, .. }
            | Op::BFloatEqImm { left, .. }
            | Op::BFloatNeImm { left, .. } => fread(*left),
            // container reads produce a dynamically-typed value, but the
            // helper-op read path refreshes the scalar shadow (`DynCheck`)
            Op::GetField { dst, .. } | Op::GetIndex { dst, .. } => {
                put(*dst, W::DynCheck);
            }
            // every other op that carries a dst writes a dynamically-typed value
            Op::LoadBody { dst, .. }
            | Op::LoadEntry { dst, .. }
            | Op::BoolEq { dst, .. }
            | Op::BoolNe { dst, .. }
            | Op::StrEq { dst, .. }
            | Op::StrNe { dst, .. }
            | Op::Bin { dst, .. }
            | Op::Unary { dst, .. }
            | Op::In { dst, .. }
            | Op::IsInstance { dst, .. }
            | Op::IsRaised { dst, .. }
            | Op::UnwrapRaised { dst, .. }
            | Op::Unwrap { dst, .. }
            | Op::UnwrapUnit { dst, .. }
            | Op::NewArray { dst }
            | Op::NewDict { dst }
            | Op::NewInstance { dst, .. }
            | Op::NewClosure { dst, .. }
            | Op::Format { dst, .. }
            | Op::Call { dst, .. }
            | Op::CallDirect { dst, .. }
            | Op::CallNative { dst, .. } => put(*dst, W::Dyn),
            Op::Jump { .. }
            | Op::JumpIf { .. }
            | Op::Switch { .. }
            | Op::Push { .. }
            | Op::Insert { .. }
            | Op::SetIndex { .. }
            | Op::SetField { .. }
            | Op::StoreEntry { .. }
            | Op::Return { .. }
            | Op::Panic {}
            | Op::Raise { .. } => {}
        }
    }

    let mut int: HashSet<u32> = HashSet::new();
    let mut float: HashSet<u32> = HashSet::new();
    for r in 0..nregs {
        if writes.contains_key(&r) {
            continue;
        }
        match (int_reads.contains(&r), float_reads.contains(&r)) {
            (true, false) => {
                int.insert(r);
            }
            (false, true) => {
                float.insert(r);
            }
            _ => {}
        }
    }
    loop {
        let mut changed = false;
        for (&r, ws) in &writes {
            // a DynCheck write only carries a shadow if `r` is genuinely read
            // as that scalar — otherwise `(sv, ok)` would be pure overhead
            let has_chk = ws.iter().any(|w| matches!(w, W::DynCheck));
            let ok_i = ws.iter().all(|w| match w {
                W::Int | W::DynCheck => true,
                W::Copy(src) => int.contains(src),
                _ => false,
            }) && (!has_chk || int_reads.contains(&r));
            let ok_f = ws.iter().all(|w| match w {
                W::Float | W::DynCheck => true,
                W::Copy(src) => float.contains(src),
                _ => false,
            }) && (!has_chk || float_reads.contains(&r));
            if ok_i != int.contains(&r) {
                if ok_i {
                    int.insert(r);
                } else {
                    int.remove(&r);
                }
                changed = true;
            }
            if ok_f != float.contains(&r) {
                if ok_f {
                    float.insert(r);
                } else {
                    float.remove(&r);
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Sh { int, float }
}

/// bcgen's `may_pause`, verbatim: the `paused` flag can only change inside an
/// op that runs foreign code — `bin`/`unary` can reach registered instance-op
/// impls, and `Call*`/`CallNative` run natives and callee bodies (whose own
/// gates propagate a pause as `Flow::Next`). An op gates on `paused` iff its
/// static fallthrough predecessor is one of these.
fn may_pause(op: &Op) -> bool {
    matches!(
        op,
        Op::Bin { .. }
            | Op::Unary { .. }
            | Op::Call { .. }
            | Op::CallDirect { .. }
            | Op::CallNative { .. }
    )
}

/// Entry-hoisted values: the `BodyFn` params plus pointers computed once
/// (the cells live in `State`/`ThreadState`, stable for the call).
struct Env<'c> {
    thread: PointerValue<'c>,
    code: PointerValue<'c>,
    ctx0: PointerValue<'c>,
    ctx1: PointerValue<'c>,
    strs: PointerValue<'c>,
    chunks: PointerValue<'c>,
    sigs: PointerValue<'c>,
    fuel_p: PointerValue<'c>,
    opip_p: PointerValue<'c>,
    out: PointerValue<'c>,
    paused_p: PointerValue<'c>,
    opsleft_p: PointerValue<'c>,
    ip_p: PointerValue<'c>,
    base: IntValue<'c>,
    nregs: IntValue<'c>,
    /// Rust-side `*const usize` table of body fn pointers for the
    /// `mj_call_body`/`mj_call_dyn` megashims (inttoptr'd constant).
    bodies_tbl: PointerValue<'c>,
}

/// The body's mutable state as entry-block allocas — mem2reg builds the SSA.
struct Vs<'c> {
    /// Window base pointer (`thread.regs + base`), refreshed after resizes.
    regs: PointerValue<'c>,
    /// Byte offset of the op currently executing (for `code.ip`/`*op_ip`).
    cur_ip: PointerValue<'c>,
    /// `RtErr` code for the `eerr` exit.
    ekind: PointerValue<'c>,
    /// Batched op quota — `min(*fuel, thread.ops_left)` at the last re-arm;
    /// decremented per op, `settle`d back into the real counters at every
    /// observable boundary. bcgen's `bcn`/`bcn0`, verbatim.
    bcn: PointerValue<'c>,
    bcn0: PointerValue<'c>,
    /// Scalar shadows: `(sv alloca, ok alloca)` per register — `i64`/`i1` and
    /// `f64`/`i1`.
    int: HashMap<u32, (PointerValue<'c>, PointerValue<'c>)>,
    float: HashMap<u32, (PointerValue<'c>, PointerValue<'c>)>,
}

/// Shared exit blocks (`dispatch`/`edef`/`estep_g` are emitted once in
/// `emit_body` and don't need struct fields).
struct Ex<'c> {
    enext: BasicBlock<'c>,
    eoof: BasicBlock<'c>,
    estep: BasicBlock<'c>,
    eerr: BasicBlock<'c>,
    /// Propagate `out` verbatim (helper errors, `Flow::*`).
    eret: BasicBlock<'c>,
    /// Fallthrough past the last op: `code.ip = usize::MAX` → `step`.
    eend: BasicBlock<'c>,
}

/// LLVM intrinsics used by the emitted code.
struct Intrinsics<'c> {
    sadd_ovf: FunctionValue<'c>,
    ssub_ovf: FunctionValue<'c>,
    smul_ovf: FunctionValue<'c>,
    sqrt: FunctionValue<'c>,
}

struct Em<'c, 'm> {
    cx: &'c Context,
    m: &'m Module<'c>,
    b: Builder<'c>,
    f: FunctionValue<'c>,
    helpers: &'m [FunctionValue<'c>],
    intr: &'m Intrinsics<'c>,
    env: Env<'c>,
    v: Vs<'c>,
    ex: Ex<'c>,
    sh: &'m Sh,
    ops: &'m [(usize, Op)],
    /// Probed VM layouts — `Val` tag/payload offsets, `ThreadState`/`Frame`/
    /// `Decoder` field offsets, `Vec` header order. Emitted code reads and
    /// writes these inline instead of FFI-ing per access.
    lyt: Layout,
    blocks: Vec<BasicBlock<'c>>,
    off2idx: HashMap<usize, usize>,
    /// Scratch `i64` cell for the `Len` helper's out-param.
    slot_i: PointerValue<'c>,
}

impl<'c> Em<'c, '_> {
    // ---- small IR helpers ----

    fn pt(&self) -> inkwell::types::PointerType<'c> {
        self.cx.ptr_type(AddressSpace::default())
    }

    fn i64c(&self, v: i64) -> IntValue<'c> {
        self.cx.i64_type().const_int(v as u64, false)
    }

    fn i32c(&self, v: i64) -> IntValue<'c> {
        self.cx.i32_type().const_int(v as u64, false)
    }

    fn i8c(&self, v: i64) -> IntValue<'c> {
        self.cx.i8_type().const_int(v as u64, false)
    }

    fn f64c(&self, bits: i64) -> FloatValue<'c> {
        self.cx.f64_type().const_float(f64::from_bits(bits as u64))
    }

    /// `p + off` (byte offset) — opaque-pointer `gep i8`.
    fn gep(&self, p: PointerValue<'c>, off: i64) -> PointerValue<'c> {
        unsafe {
            self.b
                .build_gep(self.cx.i8_type(), p, &[self.i64c(off)], "")
                .unwrap()
        }
    }

    /// `p + off` with a runtime byte offset.
    fn gep_dyn(&self, p: PointerValue<'c>, off: IntValue<'c>) -> PointerValue<'c> {
        unsafe { self.b.build_gep(self.cx.i8_type(), p, &[off], "").unwrap() }
    }

    fn ld(&self, ty: BasicTypeEnum<'c>, p: PointerValue<'c>) -> BasicValueEnum<'c> {
        self.b.build_load(ty, p, "").unwrap()
    }

    fn ldi(&self, ty: IntType<'c>, p: PointerValue<'c>) -> IntValue<'c> {
        self.ld(ty.into(), p).into_int_value()
    }

    fn ldf(&self, p: PointerValue<'c>) -> FloatValue<'c> {
        self.ld(self.cx.f64_type().into(), p).into_float_value()
    }

    fn ldp(&self, p: PointerValue<'c>) -> PointerValue<'c> {
        self.ld(self.pt().into(), p).into_pointer_value()
    }

    fn st(&self, p: PointerValue<'c>, v: impl BasicValue<'c>) {
        self.b.build_store(p, v).unwrap();
    }

    fn add(&self, a: IntValue<'c>, b: IntValue<'c>) -> IntValue<'c> {
        self.b.build_int_add(a, b, "").unwrap()
    }

    fn sub(&self, a: IntValue<'c>, b: IntValue<'c>) -> IntValue<'c> {
        self.b.build_int_sub(a, b, "").unwrap()
    }

    fn mul(&self, a: IntValue<'c>, b: IntValue<'c>) -> IntValue<'c> {
        self.b.build_int_mul(a, b, "").unwrap()
    }

    fn icmp(&self, cc: IntPredicate, a: IntValue<'c>, b: IntValue<'c>) -> IntValue<'c> {
        self.b.build_int_compare(cc, a, b, "").unwrap()
    }

    /// `a == 0` as an `i1`.
    fn isz(&self, a: IntValue<'c>) -> IntValue<'c> {
        let z = a.get_type().const_zero();
        self.icmp(IntPredicate::EQ, a, z)
    }

    /// `i1 → i8` for `Val::Bool` payloads.
    fn i1_to_i8(&self, c: IntValue<'c>) -> IntValue<'c> {
        self.b
            .build_int_z_extend(c, self.cx.i8_type(), "")
            .unwrap()
    }

    fn sel(&self, c: IntValue<'c>, t: BasicValueEnum<'c>, f: BasicValueEnum<'c>) -> BasicValueEnum<'c> {
        self.b.build_select(c, t, f, "").unwrap()
    }

    fn br(&self, t: BasicBlock<'c>) {
        self.b.build_unconditional_branch(t).unwrap();
    }

    fn cbr(&self, c: IntValue<'c>, t: BasicBlock<'c>, f: BasicBlock<'c>) {
        self.b.build_conditional_branch(c, t, f).unwrap();
    }

    fn i2p(&self, v: IntValue<'c>) -> PointerValue<'c> {
        self.b.build_int_to_ptr(v, self.pt(), "").unwrap()
    }

    /// Call a `vm::bc::jit` helper; returns the raw result (`u8` status or
    /// pointer) if the signature has one. `Pt::P` params are declared `ptr`,
    /// so `i64` word args (reg indices, counts, `usize`s) are `inttoptr`'d —
    /// identical in the C ABI, and the helper's real signature is bypassed by
    /// `add_global_mapping` anyway.
    fn hcall(
        &self,
        h: H,
        args: &[BasicMetadataValueEnum<'c>],
    ) -> Option<BasicValueEnum<'c>> {
        let coerced: Vec<BasicMetadataValueEnum> = args
            .iter()
            .map(|&a| match a {
                BasicMetadataValueEnum::IntValue(iv)
                    if iv.get_type().get_bit_width() == 64 =>
                {
                    self.i2p(iv).into()
                }
                _ => a,
            })
            .collect();
        let call = self
            .b
            .build_call(self.helpers[h as usize], &coerced, "")
            .unwrap();
        call.try_as_basic_value().basic()
    }

    // ---- alloca-variable plumbing (`def_var`/`use_var` analogues) ----

    fn regs(&self) -> PointerValue<'c> {
        self.ldp(self.v.regs)
    }

    fn set_regs(&self, v: PointerValue<'c>) {
        self.st(self.v.regs, v);
    }

    fn cur_ip(&self) -> IntValue<'c> {
        self.ldi(self.cx.i64_type(), self.v.cur_ip)
    }

    fn set_cur_ip(&self, v: IntValue<'c>) {
        self.st(self.v.cur_ip, v);
    }

    fn bcn(&self) -> IntValue<'c> {
        self.ldi(self.cx.i64_type(), self.v.bcn)
    }

    fn int_sv(&self, r: u32) -> IntValue<'c> {
        self.ldi(self.cx.i64_type(), self.v.int[&r].0)
    }

    fn int_ok(&self, r: u32) -> IntValue<'c> {
        self.ldi(self.cx.bool_type(), self.v.int[&r].1)
    }

    fn float_sv(&self, r: u32) -> FloatValue<'c> {
        self.ldf(self.v.float[&r].0)
    }

    fn float_ok(&self, r: u32) -> IntValue<'c> {
        self.ldi(self.cx.bool_type(), self.v.float[&r].1)
    }

    // ---- inline `Val` access over the probed layout (`lyt`) ----

    /// `thread.frames.len()` — one load through the probed Vec header.
    fn frames_len(&self) -> IntValue<'c> {
        self.ldi(
            self.cx.i64_type(),
            self.gep(self.env.thread, (self.lyt.frames_off + self.lyt.vec_len) as i64),
        )
    }

    /// `regs + r*val_size` — `&regs[r]` for a window base `regs`.
    fn vaddr(&self, regs: PointerValue<'c>, r: u32) -> PointerValue<'c> {
        self.gep(regs, r as i64 * self.lyt.val_size as i64)
    }

    /// The discriminant's IR type (its probed byte width).
    fn tag_ty(&self) -> IntType<'c> {
        match self.lyt.tag_size {
            1 => self.cx.i8_type(),
            2 => self.cx.i16_type(),
            4 => self.cx.i32_type(),
            8 => self.cx.i64_type(),
            d => unreachable!("bad tag width {d}"),
        }
    }

    /// A discriminant constant at the probed width.
    fn tconst(&self, t: u64) -> IntValue<'c> {
        self.tag_ty().const_int(t, false)
    }

    /// `regs[r]`'s discriminant.
    fn ld_tag(&self, a: PointerValue<'c>) -> IntValue<'c> {
        self.ldi(self.tag_ty(), self.gep(a, self.lyt.val_tag as i64))
    }

    /// `regs[r]` as `Val::Int`: inline tag probe → payload load; a miss routes
    /// to `estep` exactly like the `ri` helper's `0` return did.
    fn ld_int(&self, regs: PointerValue<'c>, r: u32) -> IntValue<'c> {
        let good = self.cx.append_basic_block(self.f, "ldint.ok");
        let a = self.vaddr(regs, r);
        let t = self.ld_tag(a);
        let hit = self.icmp(IntPredicate::EQ, t, self.tconst(self.lyt.t_int));
        self.cbr(hit, good, self.ex.estep);
        self.b.position_at_end(good);
        self.ldi(self.cx.i64_type(), self.gep(a, self.lyt.val_pay as i64))
    }

    /// `regs[r]` as `Val::Float`.
    fn ld_float(&self, regs: PointerValue<'c>, r: u32) -> FloatValue<'c> {
        let good = self.cx.append_basic_block(self.f, "ldflt.ok");
        let a = self.vaddr(regs, r);
        let t = self.ld_tag(a);
        let hit = self.icmp(IntPredicate::EQ, t, self.tconst(self.lyt.t_float));
        self.cbr(hit, good, self.ex.estep);
        self.b.position_at_end(good);
        self.ldf(self.gep(a, self.lyt.val_pay as i64))
    }

    /// `*a = Val::Int(v)` — tag byte plus the 8-byte union slot.
    fn st_int(&self, a: PointerValue<'c>, v: IntValue<'c>) {
        self.st(self.gep(a, self.lyt.val_tag as i64), self.tconst(self.lyt.t_int));
        self.st(self.gep(a, self.lyt.val_pay as i64), v);
    }

    /// `*a = Val::Float(v)` (ditto, `f64` store).
    fn st_float(&self, a: PointerValue<'c>, v: FloatValue<'c>) {
        self.st(self.gep(a, self.lyt.val_tag as i64), self.tconst(self.lyt.t_float));
        self.st(self.gep(a, self.lyt.val_pay as i64), v);
    }

    /// `*a = Val::Bool(v8)` — tag plus the `u8` union slot.
    fn st_bool(&self, a: PointerValue<'c>, v8: IntValue<'c>) {
        self.st(self.gep(a, self.lyt.val_tag as i64), self.tconst(self.lyt.t_bool));
        self.st(self.gep(a, self.lyt.bool_pay as i64), v8);
    }

    /// `*a = Val::Null` — only the tag byte is read for payload-less variants.
    fn st_null(&self, a: PointerValue<'c>) {
        self.st(self.gep(a, self.lyt.val_tag as i64), self.tconst(self.lyt.t_null));
    }

    /// `*a = Val::Fn(body)` — tag plus the `u32` index in the union slot.
    fn st_fn(&self, a: PointerValue<'c>, body32: IntValue<'c>) {
        self.st(self.gep(a, self.lyt.val_tag as i64), self.tconst(self.lyt.t_fn));
        self.st(self.gep(a, self.lyt.fn_pay as i64), body32);
    }

    /// `val_size`-byte `Val` copy `*d = *s` (eight-byte chunks; the probe
    /// asserts the stride is a multiple of 8).
    fn cpy_val(&self, d: PointerValue<'c>, s: PointerValue<'c>) {
        for k in 0..(self.lyt.val_size / 8) as i64 {
            let w = self.ldi(self.cx.i64_type(), self.gep(s, k * 8));
            self.st(self.gep(d, k * 8), w);
        }
    }

    /// The block implementing the op at byte offset `t` — jump targets always
    /// land on op starts, so the dense map is exact.
    fn tgt_blk(&self, t: &BlockTarget) -> BasicBlock<'c> {
        let BlockTarget::ByteOffset(o) = t else {
            panic!("unresolved BlockTarget in compiled program")
        };
        self.blocks[self.off2idx[o]]
    }

    /// Successor block for fallthrough (`next` op, or `eend` off the end).
    fn next_blk(&self, i: usize) -> BasicBlock<'c> {
        if i + 1 < self.ops.len() {
            self.blocks[i + 1]
        } else {
            self.ex.eend
        }
    }

    /// "Flush shadows, then jump to `cont`" — the inline writeback, used when
    /// control must rejoin a *shared* block rather than continue in the
    /// current chain.
    fn flush_jump(&self, cont: BasicBlock<'c>) {
        self.flush_seq();
        self.br(cont);
    }

    /// The inline shadow-writeback sequence — used inside shared exits (which
    /// are already merge points). Branch-free: a dead shadow (`ok == 0`) means
    /// the window is already authoritative, so the tag/payload stores rewrite
    /// the bytes just loaded — a `select` per field instead of a branch per
    /// reg.
    fn flush_seq(&self) {
        let mut ints: Vec<u32> = self.sh.int.iter().copied().collect();
        ints.sort_unstable();
        let mut floats: Vec<u32> = self.sh.float.iter().copied().collect();
        floats.sort_unstable();
        if ints.is_empty() && floats.is_empty() {
            return;
        }
        let regs = self.regs();
        let tint = self.tconst(self.lyt.t_int);
        for r in ints {
            let (sv, ok) = self.v.int[&r];
            let okv = self.ldi(self.cx.bool_type(), ok);
            let v = self.ldi(self.cx.i64_type(), sv);
            let a = self.vaddr(regs, r);
            let old = self.ld_tag(a);
            let nt = self.sel(okv, tint.into(), old.into());
            self.st(self.gep(a, self.lyt.val_tag as i64), nt);
            let oldp = self.ldi(self.cx.i64_type(), self.gep(a, self.lyt.val_pay as i64));
            let np = self.sel(okv, v.into(), oldp.into());
            self.st(self.gep(a, self.lyt.val_pay as i64), np);
        }
        let tflt = self.tconst(self.lyt.t_float);
        for r in floats {
            let (sv, ok) = self.v.float[&r];
            let okv = self.ldi(self.cx.bool_type(), ok);
            let v = self.ldf(sv);
            let a = self.vaddr(regs, r);
            let old = self.ld_tag(a);
            let nt = self.sel(okv, tflt.into(), old.into());
            self.st(self.gep(a, self.lyt.val_tag as i64), nt);
            let oldp = self.ldf(self.gep(a, self.lyt.val_pay as i64));
            let np = self.sel(okv, v.into(), oldp.into());
            self.st(self.gep(a, self.lyt.val_pay as i64), np);
        }
    }

    /// `code.ip = v` through the hoisted cell pointer.
    fn store_ip(&self, v: IntValue<'c>) {
        self.st(self.env.ip_p, v);
    }

    /// `*op_ip = v` through the hoisted cell pointer.
    fn store_opip(&self, v: IntValue<'c>) {
        self.st(self.env.opip_p, v);
    }

    /// `code.ip = next; *op_ip = off` — bcgen's pre-helper/pre-call idiom.
    fn mark_op(&self, off: usize, next: usize) {
        self.store_ip(self.i64c(next as i64));
        self.store_opip(self.i64c(off as i64));
    }

    /// Route an `RtErr` kind to the `eerr` exit through a tiny trampoline that
    /// stores `ekind` — `cbr` targets can't carry the constant themselves.
    /// The block is created empty and *returned* for use as a `cbr` target;
    /// call [`Em::fill_err`] after the current path is terminated.
    fn err_tramp(&self) -> BasicBlock<'c> {
        self.cx.append_basic_block(self.f, "err")
    }

    /// Fill a trampoline created by [`Em::err_tramp`]: `ekind = kind; →eerr`.
    /// Positions at `t` (must be called when the current block is terminated).
    fn fill_err(&self, t: BasicBlock<'c>, kind: i64) {
        self.b.position_at_end(t);
        self.st(self.v.ekind, self.i8c(kind));
        self.br(self.ex.eerr);
    }

    /// Read `regs[r]` as an i64 — shadow when eligible, else the inline tag
    /// probe. A miss routes to `estep`: `step` runs the op verbatim — the same
    /// outcome bcgen's `bin_cold`/`branch_cold` tail-calls produce.
    fn int_opnd(&self, r: u32) -> IntValue<'c> {
        if self.v.int.contains_key(&r) {
            let good = self.cx.append_basic_block(self.f, "iop.ok");
            let okv = self.int_ok(r);
            self.cbr(okv, good, self.ex.estep);
            self.b.position_at_end(good);
            self.int_sv(r)
        } else {
            let regs = self.regs();
            self.ld_int(regs, r)
        }
    }

    fn float_opnd(&self, r: u32) -> FloatValue<'c> {
        if self.v.float.contains_key(&r) {
            let good = self.cx.append_basic_block(self.f, "fop.ok");
            let okv = self.float_ok(r);
            self.cbr(okv, good, self.ex.estep);
            self.b.position_at_end(good);
            self.float_sv(r)
        } else {
            let regs = self.regs();
            self.ld_float(regs, r)
        }
    }

    /// Scalar write into `dst`: shadow-slot store when shadowed, else an
    /// inline `Val` store.
    fn wr_int_dst(&self, d: u32, v: IntValue<'c>) {
        if let Some(&(sv, ok)) = self.v.int.get(&d) {
            self.st(sv, v);
            self.st(ok, self.cx.bool_type().const_int(1, false));
        } else {
            let regs = self.regs();
            let a = self.vaddr(regs, d);
            self.st_int(a, v);
        }
    }

    fn wr_float_dst(&self, d: u32, v: FloatValue<'c>) {
        if let Some(&(sv, ok)) = self.v.float.get(&d) {
            self.st(sv, v);
            self.st(ok, self.cx.bool_type().const_int(1, false));
        } else {
            let regs = self.regs();
            let a = self.vaddr(regs, d);
            self.st_float(a, v);
        }
    }

    /// Bool write into `dst` — eval dsts are always `W::Dyn` (unshadowed).
    fn wr_bool_dst(&self, d: u32, v8: IntValue<'c>) {
        let regs = self.regs();
        let a = self.vaddr(regs, d);
        self.st_bool(a, v8);
    }

    /// A `*const Val` for reg `s`, materializing a live shadow first — after
    /// this `regs[s]` is authoritative for the read.
    fn val_ptr(&self, s: u32) -> PointerValue<'c> {
        if self.v.int.contains_key(&s) {
            let c = self.cx.append_basic_block(self.f, "vp.c");
            let m = self.cx.append_basic_block(self.f, "vp.m");
            let okv = self.int_ok(s);
            self.cbr(okv, m, c);
            self.b.position_at_end(m);
            let regs = self.regs();
            let a = self.vaddr(regs, s);
            let v = self.int_sv(s);
            self.st_int(a, v);
            self.br(c);
            self.b.position_at_end(c);
        } else if self.v.float.contains_key(&s) {
            let c = self.cx.append_basic_block(self.f, "vp.c");
            let m = self.cx.append_basic_block(self.f, "vp.m");
            let okv = self.float_ok(s);
            self.cbr(okv, m, c);
            self.b.position_at_end(m);
            let regs = self.regs();
            let a = self.vaddr(regs, s);
            let v = self.float_sv(s);
            self.st_float(a, v);
            self.br(c);
            self.b.position_at_end(c);
        }
        let regs = self.regs();
        self.vaddr(regs, s)
    }

    /// Helper-op shape: inline shadow writeback, then stores `code.ip`/
    /// `*op_ip`, calls the `u8` helper, then `1`→`eret` / `0`→fallthrough.
    /// `args` are built by the caller before this runs.
    fn helper_op(&self, i: usize, off: usize, next: usize, h: H, args: &[BasicMetadataValueEnum<'c>]) {
        self.flush_seq();
        self.mark_op(off, next);
        let k = self.hcall(h, args).unwrap().into_int_value();
        let nb = self.next_blk(i);
        self.cbr(self.isz(k), nb, self.ex.eret);
    }

    /// Same but the helper returns void — always continues.
    fn helper_op_v(&self, i: usize, off: usize, next: usize, h: H, args: &[BasicMetadataValueEnum<'c>]) {
        self.flush_seq();
        self.mark_op(off, next);
        self.hcall(h, args);
        let nb = self.next_blk(i);
        self.br(nb);
    }

    /// After a helper wrote `regs[r]` behind the shadow's back, re-derive the
    /// `DynCheck` `(sv, ok)` pair from the authoritative slot.
    fn refresh_shadow(&self, r: u32) {
        if !self.v.int.contains_key(&r) && !self.v.float.contains_key(&r) {
            return;
        }
        let regs = self.regs();
        let a = self.vaddr(regs, r);
        let t = self.ld_tag(a);
        if let Some(&(sv, ok)) = self.v.int.get(&r) {
            let k = self.icmp(IntPredicate::EQ, t, self.tconst(self.lyt.t_int));
            let pv = self.ldi(self.cx.i64_type(), self.gep(a, self.lyt.val_pay as i64));
            self.st(sv, pv);
            self.st(ok, k);
        }
        if let Some(&(sv, ok)) = self.v.float.get(&r) {
            let k = self.icmp(IntPredicate::EQ, t, self.tconst(self.lyt.t_float));
            let pv = self.ldf(self.gep(a, self.lyt.val_pay as i64));
            self.st(sv, pv);
            self.st(ok, k);
        }
    }

    /// `helper_op` + a post-call [`Em::refresh_shadow`] on `dst` — every
    /// `GetIndex`/`GetField` helper invocation must go through this so a
    /// `DynCheck`-shadowed dst stays coherent.
    fn helper_op_read(&self, i: usize, off: usize, next: usize, h: H, args: &[BasicMetadataValueEnum<'c>], dst: Reg) {
        self.flush_seq();
        self.mark_op(off, next);
        let k = self.hcall(h, args).unwrap().into_int_value();
        let post = self.cx.append_basic_block(self.f, "hop.post");
        self.cbr(self.isz(k), post, self.ex.eret);
        self.b.position_at_end(post);
        self.refresh_shadow(dst.index() as u32);
        let nb = self.next_blk(i);
        self.br(nb);
    }

    /// A `u32` register-index list (call args / field regs / captures) as a
    /// private constant global; returns the array pointer for the helper.
    fn reg_list_ptr(&self, regs_idx: &[Reg]) -> PointerValue<'c> {
        let n = regs_idx.len().max(1);
        let i32t = self.cx.i32_type();
        let vals: Vec<IntValue<'c>> = if regs_idx.is_empty() {
            vec![i32t.const_int(0, false)]
        } else {
            regs_idx
                .iter()
                .map(|r| i32t.const_int(r.index() as u64, false))
                .collect()
        };
        let arr = i32t.const_array(&vals);
        let gv = self
            .m
            .add_global(i32t.array_type(n as u32), None, "ml_regs");
        gv.set_constant(true);
        gv.set_initializer(&arr);
        gv.as_pointer_value()
    }

    /// `if frames.len() == 1 { flush }` before producing `Flow::Return` —
    /// the host reads the root frame's regs after `run()`. Continues in
    /// `cont` either way.
    fn root_flush(&self, cont: BasicBlock<'c>) {
        let flen = self.frames_len();
        let is_root = self.icmp(IntPredicate::EQ, flen, self.i64c(1));
        let ft = self.cx.append_basic_block(self.f, "rootf");
        self.cbr(is_root, ft, cont);
        self.b.position_at_end(ft);
        self.flush_jump(cont);
    }

    /// `step_at`-backed whole-op fallback: nothing is stored here — `estep`
    /// itself writes `code.ip`/`*op_ip` from `cur_ip` and runs `step`.
    fn estep(&self) {
        self.br(self.ex.estep);
    }

    // ---- batched quota (`bcn`) plumbing — bcgen's settle!/gexit!/gateq! ----

    /// `spent = bcn0 - bcn; bcn0 = bcn; *fuel -= spent; ops_left -= spent`.
    /// Idempotent (a second settle charges 0), so exits can run it
    /// unconditionally. Returns the post-settle `(fuel, ops_left)` values.
    fn settle_seq(&self) -> (IntValue<'c>, IntValue<'c>) {
        let b0 = self.ldi(self.cx.i64_type(), self.v.bcn0);
        let b = self.bcn();
        let spent = self.sub(b0, b);
        self.st(self.v.bcn0, b);
        let f = self.ldi(self.cx.i64_type(), self.env.fuel_p);
        let f2 = self.sub(f, spent);
        self.st(self.env.fuel_p, f2);
        let ol = self.ldi(self.cx.i64_type(), self.env.opsleft_p);
        let ol2 = self.sub(ol, spent);
        self.st(self.env.opsleft_p, ol2);
        (f2, ol2)
    }

    /// `bcn = min(*fuel, ops_left); bcn0 = bcn` — armed at entry and re-armed
    /// wherever quota was spent outside our count.
    fn rearm_seq(&self) {
        let f = self.ldi(self.cx.i64_type(), self.env.fuel_p);
        let ol = self.ldi(self.cx.i64_type(), self.env.opsleft_p);
        let lt = self.icmp(IntPredicate::ULT, f, ol);
        let m = self.sel(lt, f.into(), ol.into());
        self.st(self.v.bcn, m);
        self.st(self.v.bcn0, m);
    }

    /// One op's quota gate — bcgen's `gateq!()`/`gatep!()`: `paused` is only
    /// loaded when the static fallthrough predecessor can run foreign code
    /// (`may_pause`); a `bcn == 0` trip (or the pause flag) routes to a
    /// per-op cold trampoline running `gexit` — settle, then the driver's
    /// ordered exit reasons, then re-arm and resume the op. Returns the
    /// continuation block the op body emits into.
    fn gate(&self, i: usize, off: usize) -> (BasicBlock<'c>, BasicBlock<'c>) {
        self.set_cur_ip(self.i64c(off as i64));
        let cont = self.cx.append_basic_block(self.f, "g.cont");
        let tramp = self.cx.append_basic_block(self.f, "g.tramp");
        if i > 0 && may_pause(&self.ops[i - 1].1) {
            let p = self.ldi(self.cx.i8_type(), self.env.paused_p);
            let g = self.cx.append_basic_block(self.f, "g.np");
            let pn = self.icmp(IntPredicate::NE, p, self.i8c(0));
            self.cbr(pn, tramp, g);
            self.b.position_at_end(g);
        }
        let bz = self.isz(self.bcn());
        self.cbr(bz, tramp, cont);
        self.b.position_at_end(cont);
        let cur = self.bcn();
        let b1 = self.sub(cur, self.i64c(1));
        self.st(self.v.bcn, b1);
        (cont, tramp)
    }

    /// Fill a gate trampoline: settle, then `paused || fuel == 0` → `enext`,
    /// `ops_left == 0` → `eoof`, else re-arm and resume at `cont`. Call when
    /// the current block is terminated.
    fn fill_gate_tramp(&self, tramp: BasicBlock<'c>, cont: BasicBlock<'c>) {
        self.b.position_at_end(tramp);
        let (f, ol) = self.settle_seq();
        let p = self.ldi(self.cx.i8_type(), self.env.paused_p);
        let pn = self.icmp(IntPredicate::NE, p, self.i8c(0));
        let fz = self.isz(f);
        let nx = self.b.build_or(pn, fz, "").unwrap();
        let t2 = self.cx.append_basic_block(self.f, "gt.nf");
        self.cbr(nx, self.ex.enext, t2);
        self.b.position_at_end(t2);
        let oz = self.isz(ol);
        let t3 = self.cx.append_basic_block(self.f, "gt.no");
        self.cbr(oz, self.ex.eoof, t3);
        self.b.position_at_end(t3);
        self.rearm_seq();
        self.br(cont);
    }
}

/// One helper's C-ABI signature as an LLVM function type — `Ctx` flattens to
/// two pointer params exactly like the Cranelift emitter's `Pt::Ctx`.
fn spec_fn_ty<'c>(cx: &'c Context, s: &crate::Spec) -> FunctionType<'c> {
    let ptr = cx.ptr_type(AddressSpace::default());
    let mut params: Vec<BasicMetadataTypeEnum<'c>> = Vec::new();
    for p in s.params {
        match p {
            Pt::P => params.push(ptr.into()),
            Pt::I8 => params.push(cx.i8_type().into()),
            Pt::I32 => params.push(cx.i32_type().into()),
            Pt::F64 => params.push(cx.f64_type().into()),
            Pt::Ctx => {
                params.push(ptr.into());
                params.push(ptr.into());
            }
        }
    }
    match s.ret {
        Hr::Void => cx.void_type().fn_type(&params, false),
        Hr::U8 => cx.i8_type().fn_type(&params, false),
        Hr::P => ptr.fn_type(&params, false),
    }
}

/// Compile `program` — every chunk to one `mlb_N` function at the `BodyFn`
/// ABI, `default<O2>` over the module, MCJIT, helper binding by
/// `add_global_mapping`.
pub(crate) fn emit_program(program: &Program) -> Result<Jit, Error> {
    Target::initialize_native(&InitializationConfig::default()).map_err(Error)?;
    let cx: &'static Context = Box::leak(Box::new(Context::create()));
    let module = cx.create_module("mimas_llvm");
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| Error(e.to_string()))?;
    let tm = target
        .create_target_machine(
            &triple,
            "generic",
            "",
            OptimizationLevel::Aggressive,
            RelocMode::Default,
            CodeModel::JITDefault,
        )
        .ok_or_else(|| Error("no target machine for host triple".into()))?;
    module.set_triple(&triple);
    module.set_data_layout(&tm.get_target_data().get_data_layout());

    let ptr = cx.ptr_type(AddressSpace::default());
    let i64t = cx.i64_type();
    let i1t = cx.bool_type();

    // Helper decls — resolved by absolute address through
    // `ExecutionEngine::add_global_mapping` below.
    let helper_fns: Vec<FunctionValue> = SPECS
        .iter()
        .map(|s| module.add_function(s.name, spec_fn_ty(cx, s), None))
        .collect();

    // `{i64, i1}` overflow intrinsics + `llvm.sqrt.f64`.
    let ovf_ty = cx
        .struct_type(&[i64t.into(), i1t.into()], false)
        .fn_type(&[i64t.into(), i64t.into()], false);
    let intr = Intrinsics {
        sadd_ovf: module.add_function("llvm.sadd.with.overflow.i64", ovf_ty, None),
        ssub_ovf: module.add_function("llvm.ssub.with.overflow.i64", ovf_ty, None),
        smul_ovf: module.add_function("llvm.smul.with.overflow.i64", ovf_ty, None),
        sqrt: module.add_function(
            "llvm.sqrt.f64",
            cx.f64_type().fn_type(&[cx.f64_type().into()], false),
            None,
        ),
    };

    // The BodyFn C ABI: (thread, code, Ctx-by-value as two ptrs, strs, chunks,
    // signatures, fuel, op_ip, out) — ten pointer params, no return.
    let body_ty = cx.void_type().fn_type(&[ptr.into(); 10], false);
    let nbodies = program.chunks.len();
    let body_fns: Vec<FunctionValue> = (0..nbodies)
        .map(|i| module.add_function(&format!("mlb_{i}"), body_ty, None))
        .collect();

    // Rust-side body-pointer table for `mj_call_body`/`mj_call_dyn` — filled
    // post-JIT with the finalized addresses; its address is baked into calls
    // as an `inttoptr` constant.
    let mut tbl = vec![0usize; nbodies.max(1)].into_boxed_slice();
    let tbl_addr = tbl.as_ptr() as usize;

    let lyt = jit::layout();
    for body in 0..nbodies {
        emit_body(
            cx,
            &module,
            program,
            body,
            &helper_fns,
            body_fns[body],
            &intr,
            &lyt,
            tbl_addr,
        )?;
    }

    module.verify().map_err(|e| Error(e.to_string()))?;
    module
        .run_passes("default<O2>", &tm, PassBuilderOptions::create())
        .map_err(|e| Error(e.to_string()))?;

    // `default<O2>`'s globaldce drops unused helper declarations — re-resolve
    // each name post-pass and map only the survivors (a removed decl was
    // unused by definition).
    let live: Vec<(FunctionValue<'static>, usize)> = SPECS
        .iter()
        .filter_map(|s| module.get_function(s.name).map(|fv| (fv, s.addr)))
        .collect();
    let engine = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .map_err(|e| Error(e.to_string()))?;
    for (fv, addr) in live {
        engine.add_global_mapping(&fv, addr);
    }

    let mut bodies = Vec::with_capacity(nbodies);
    for (i, fv) in body_fns.iter().enumerate() {
        let addr = engine
            .get_function_address(&format!("mlb_{i}"))
            .map_err(|e| Error(e.to_string()))?;
        debug_assert_ne!(addr, 0);
        let _ = fv;
        tbl[i] = addr;
        // SAFETY: the emitted function implements exactly the extern "C"
        // BodyFn signature (`body_ty` above); the engine outlives it via `Jit`.
        bodies.push(Some(unsafe { std::mem::transmute::<usize, BodyFn>(addr) }));
    }
    Ok(Jit {
        _engine: engine,
        bodies,
        _tbl: tbl,
    })
}

/// Emit `chunks[body]`'s op stream into `f` (`mlb_{body}`).
#[allow(clippy::too_many_arguments)]
fn emit_body<'c>(
    cx: &'c Context,
    module: &Module<'c>,
    program: &Program,
    body: usize,
    helper_fns: &[FunctionValue<'c>],
    f: FunctionValue<'c>,
    intr: &Intrinsics<'c>,
    lyt: &Layout,
    tbl_addr: usize,
) -> Result<(), Error> {
    let body_id = compile::BodyId::from(body as u32);
    let ops = program.ops(body_id);
    let chunk = &program.chunks[body_id];
    let chunk_off = chunk.offset;
    let sh = analyze(&ops, chunk.regs as u32);
    let i64t = cx.i64_type();
    let i8t = cx.i8_type();
    let i1t = cx.bool_type();
    let builder = cx.create_builder();

    // ---- blocks ----
    let entry = cx.append_basic_block(f, "entry");
    let dispatch = cx.append_basic_block(f, "dispatch");
    let edef = cx.append_basic_block(f, "edef");
    let enext = cx.append_basic_block(f, "enext");
    let eoof = cx.append_basic_block(f, "eoof");
    let estep = cx.append_basic_block(f, "estep");
    let estep_g = cx.append_basic_block(f, "estep_g");
    let eerr = cx.append_basic_block(f, "eerr");
    let eret = cx.append_basic_block(f, "eret");
    let eend = cx.append_basic_block(f, "eend");
    let blocks: Vec<BasicBlock> = ops
        .iter()
        .enumerate()
        .map(|(i, _)| cx.append_basic_block(f, &format!("op_{i}")))
        .collect();
    let off2idx: HashMap<usize, usize> =
        ops.iter().enumerate().map(|(i, (o, _))| (*o, i)).collect();

    // ---- entry: allocas first (mem2reg promotes them all) ----
    builder.position_at_end(entry);
    let v = Vs {
        regs: builder.build_alloca(cx.ptr_type(AddressSpace::default()), "regs").unwrap(),
        cur_ip: builder.build_alloca(i64t, "cur_ip").unwrap(),
        ekind: builder.build_alloca(i8t, "ekind").unwrap(),
        bcn: builder.build_alloca(i64t, "bcn").unwrap(),
        bcn0: builder.build_alloca(i64t, "bcn0").unwrap(),
        int: sh
            .int
            .iter()
            .map(|&r| {
                (
                    r,
                    (
                        builder.build_alloca(i64t, &format!("isv_{r}")).unwrap(),
                        builder.build_alloca(i1t, &format!("iok_{r}")).unwrap(),
                    ),
                )
            })
            .collect(),
        float: sh
            .float
            .iter()
            .map(|&r| {
                (
                    r,
                    (
                        builder.build_alloca(cx.f64_type(), &format!("fsv_{r}")).unwrap(),
                        builder.build_alloca(i1t, &format!("fok_{r}")).unwrap(),
                    ),
                )
            })
            .collect(),
    };
    let slot_i = builder.build_alloca(i64t, "slot_i").unwrap();

    let p: Vec<PointerValue> = f
        .get_params()
        .iter()
        .map(|v| v.into_pointer_value())
        .collect();
    let env = Env {
        thread: p[0],
        code: p[1],
        ctx0: p[2],
        ctx1: p[3],
        strs: p[4],
        chunks: p[5],
        sigs: p[6],
        fuel_p: p[7],
        opip_p: p[8],
        out: p[9],
        paused_p: ptr_ph(cx),
        opsleft_p: ptr_ph(cx),
        ip_p: ptr_ph(cx),
        base: i64t.const_int(0, false),
        nregs: i64t.const_int(0, false),
        bodies_tbl: ptr_ph(cx),
    };
    let mut em = Em {
        cx,
        m: module,
        b: builder,
        f,
        helpers: helper_fns,
        intr,
        env,
        v,
        ex: Ex {
            enext,
            eoof,
            estep,
            eerr,
            eret,
            eend,
        },
        sh: &sh,
        ops: &ops,
        lyt: *lyt,
        blocks,
        off2idx,
        slot_i,
    };
    // `builder` was only borrowed for allocas; Em owns its own Builder and is
    // already positioned at `entry`.
    em.b.position_at_end(entry);

    // hoist environment pointers — `&state.paused` is one add off `Ctx`'s
    // second word now that `State`'s layout is probed; everything else —
    // `&code.ip`, `&thread.ops_left`, the top frame's `base`, the `regs`
    // buffer — is a handful of loads over the probed layout. `nregs` is the
    // chunk's own `regs` field: a compile-time constant.
    em.env.paused_p = em.gep(em.env.ctx1, lyt.state_paused as i64);
    em.env.opsleft_p = em.gep(em.env.thread, lyt.ops_left_off as i64);
    em.env.ip_p = em.gep(em.env.code, lyt.code_ip as i64);
    // `thread.frames.last().unwrap().base`
    let fptr = em.ldp(em.gep(em.env.thread, (lyt.frames_off + lyt.vec_ptr) as i64));
    let flen = em.ldi(i64t, em.gep(em.env.thread, (lyt.frames_off + lyt.vec_len) as i64));
    let fm1 = em.sub(flen, em.i64c(1));
    let foff = em.mul(fm1, em.i64c(lyt.frame_size as i64));
    let faddr = em.gep_dyn(fptr, foff);
    let base = em.ldi(i64t, em.gep(faddr, lyt.frame_base as i64));
    // `thread.regs.as_mut_ptr()`
    let rp = em.ldp(em.gep(em.env.thread, (lyt.regs_off + lyt.vec_ptr) as i64));
    let boff = em.mul(base, em.i64c(lyt.val_size as i64));
    let regs0 = em.gep_dyn(rp, boff);
    em.env.base = base;
    em.env.nregs = em.i64c(chunk.regs as i64);
    em.env.bodies_tbl = em.i2p(em.i64c(tbl_addr as i64));
    em.set_regs(regs0);
    em.set_cur_ip(em.i64c(0));
    em.st(em.v.ekind, em.i8c(0));
    // arm the batched quota: `bcn = min(*fuel, ops_left)` (bcgen's entry arm).
    // The driver already checked paused/fuel/ops_left before invoking the
    // body, so bcn >= 1 here.
    em.rearm_seq();

    // shadow init: `(v, ok) = match regs[r] { Int(v) => (v, true), _ => (0,false) }`
    let mut int_keys: Vec<u32> = em.v.int.keys().copied().collect();
    int_keys.sort_unstable();
    for r in int_keys {
        let (sv, ok) = em.v.int[&r];
        let a = em.vaddr(regs0, r);
        let t = em.ld_tag(a);
        let k = em.icmp(IntPredicate::EQ, t, em.tconst(lyt.t_int));
        let vv = em.ldi(i64t, em.gep(a, lyt.val_pay as i64));
        let sv0 = em.sel(k, vv.into(), i64t.const_int(0, false).into());
        em.st(sv, sv0);
        em.st(ok, k);
    }
    let mut float_keys: Vec<u32> = em.v.float.keys().copied().collect();
    float_keys.sort_unstable();
    for r in float_keys {
        let (sv, ok) = em.v.float[&r];
        let a = em.vaddr(regs0, r);
        let t = em.ld_tag(a);
        let k = em.icmp(IntPredicate::EQ, t, em.tconst(lyt.t_float));
        let vv = em.ldf(em.gep(a, lyt.val_pay as i64));
        let zf = cx.f64_type().const_float(0.0);
        let sv0 = em.sel(k, vv.into(), zf.into());
        em.st(sv, sv0);
        em.st(ok, k);
    }
    em.br(dispatch);

    // ---- dispatch: code.ip - chunk_off → switch over op byte offsets ----
    em.b.position_at_end(dispatch);
    if ops.is_empty() {
        em.br(edef);
        em.b.position_at_end(edef);
        em.set_cur_ip(em.i64c(0));
        em.br(estep_g);
    } else {
        let ip = em.ldi(i64t, em.env.ip_p);
        let rel = em.sub(ip, em.i64c(chunk_off as i64));
        let cases: Vec<(IntValue, BasicBlock)> = ops
            .iter()
            .enumerate()
            .map(|(i, (o, _))| (em.i64c(*o as i64 - chunk_off as i64), em.blocks[i]))
            .collect();
        em.b.build_switch(rel, edef, &cases).unwrap();
        // `edef` needs the raw ip for the step fallback
        em.b.position_at_end(edef);
        em.set_cur_ip(ip);
        em.br(estep_g);
    }

    // ---- shared exits ----
    // `estep_g` — the quota gate for `edef`/`eend`: `run_dispatch` runs its
    // paused/fuel/ops_left checks and charges one op even for a garbage
    // decode, so the fallback entry gates `bcn` exactly like an op would.
    em.b.position_at_end(estep_g);
    let g_cont = cx.append_basic_block(f, "sg.cont");
    let g_tramp = cx.append_basic_block(f, "sg.tramp");
    let pv = em.ldi(i8t, em.env.paused_p);
    let pvn = em.icmp(IntPredicate::NE, pv, em.i8c(0));
    em.cbr(pvn, g_tramp, g_cont);
    em.b.position_at_end(g_cont);
    let g2 = cx.append_basic_block(f, "sg.g2");
    let bz = em.isz(em.bcn());
    em.cbr(bz, g_tramp, g2);
    em.b.position_at_end(g2);
    // merged `bcn` (the gate-tramp re-arm also lands here)
    let cur = em.bcn();
    let b1 = em.sub(cur, em.i64c(1));
    em.st(em.v.bcn, b1);
    em.br(estep);
    em.fill_gate_tramp(g_tramp, g2);

    em.b.position_at_end(enext);
    em.settle_seq();
    em.flush_seq();
    let ip = em.cur_ip();
    em.store_ip(ip);
    em.hcall(H::OutNext, &[em.env.out.into()]);
    em.b.build_return(None).unwrap();

    em.b.position_at_end(eoof);
    em.settle_seq();
    em.flush_seq();
    let ip = em.cur_ip();
    em.store_opip(ip);
    em.hcall(H::OutErr, &[em.env.out.into(), em.i8c(ERR_OOF).into()]);
    em.b.build_return(None).unwrap();

    em.b.position_at_end(estep);
    em.settle_seq();
    em.flush_seq();
    let ip = em.cur_ip();
    em.store_ip(ip);
    em.store_opip(ip);
    let regs = em.regs();
    let k = em
        .hcall(
            H::StepAt,
            &[
                em.env.thread.into(),
                regs.into(),
                em.env.nregs.into(),
                em.env.code.into(),
                em.env.ctx0.into(),
                em.env.ctx1.into(),
                em.env.strs.into(),
                em.env.out.into(),
            ],
        )
        .unwrap()
        .into_int_value();
    em.cbr(em.isz(k), dispatch, eret);

    em.b.position_at_end(eerr);
    em.settle_seq();
    em.flush_seq();
    let ip = em.cur_ip();
    em.store_opip(ip);
    let kk = em.ldi(i8t, em.v.ekind);
    em.hcall(H::OutErr, &[em.env.out.into(), kk.into()]);
    em.b.build_return(None).unwrap();

    em.b.position_at_end(eret);
    em.settle_seq();
    em.b.build_return(None).unwrap();

    em.b.position_at_end(eend);
    let max = em.i64c(-1); // usize::MAX — garbage decode, same as bcgen's `_ =>`
    em.set_cur_ip(max);
    em.br(estep_g);

    // ---- op blocks ----
    for (i, (off, op)) in ops.iter().enumerate() {
        let next = ops.get(i + 1).map(|(o, _)| *o).unwrap_or(usize::MAX);
        em.emit_op(i, *off, next, op);
    }

    f.verify(false);
    Ok(())
}

/// Placeholder pointer for `Env` fields filled during entry emission.
fn ptr_ph<'c>(cx: &'c Context) -> PointerValue<'c> {
    cx.ptr_type(AddressSpace::default()).const_null()
}

impl<'c> Em<'c, '_> {
    /// One op block: the driver's per-op bookkeeping as a batched-quota gate
    /// (bcgen's `gateq!`/`gatep!`), then semantics.
    fn emit_op(&mut self, i: usize, off: usize, next: usize, op: &Op) {
        self.b.position_at_end(self.blocks[i]);
        let (cont, tramp) = self.gate(i, off);
        self.semantics(i, off, next, op);
        // the gate's trampoline — `fill_gate_tramp` needs the current block
        // terminated, which `semantics` guarantees (every arm ends in a
        // branch or return).
        self.fill_gate_tramp(tramp, cont);
    }

    fn semantics(&mut self, i: usize, off: usize, next: usize, op: &Op) {
        match op {
            Op::Move { dst, src } => {
                self.emit_move(*dst, *src);
                let nb = self.next_blk(i);
                self.br(nb);
            }
            Op::Jump { target } => {
                let t = self.tgt_blk(target);
                self.br(t);
            }
            Op::JumpIf {
                cond,
                target,
                is_true,
            } => {
                // `regs[cond] == Val::Bool(is_true)` — tag + payload byte.
                let regs = self.regs();
                let a = self.vaddr(regs, cond.index() as u32);
                let t = self.ld_tag(a);
                let tb2 = self.icmp(IntPredicate::EQ, t, self.tconst(self.lyt.t_bool));
                let pb = self.ldi(self.cx.i8_type(), self.gep(a, self.lyt.bool_pay as i64));
                let peq = self.icmp(IntPredicate::EQ, pb, self.i8c(*is_true as i64));
                let k = self.b.build_and(tb2, peq, "").unwrap();
                let hit = self.mask_by_shadow(*cond, k);
                let (tb, fb) = (self.tgt_blk(target), self.next_blk(i));
                self.cbr(hit, tb, fb);
            }
            Op::ForNext { idx, bound, target } => {
                let iv = self.int_opnd(idx.index() as u32);
                let bv = self.int_opnd(bound.index() as u32);
                let i2 = self.add(iv, self.i64c(1));
                self.wr_int_dst(idx.index() as u32, i2);
                let hit = self.icmp(IntPredicate::SLT, i2, bv);
                let (tb, fb) = (self.tgt_blk(target), self.next_blk(i));
                self.cbr(hit, tb, fb);
            }
            Op::Switch {
                scrut,
                base,
                default,
                table,
            } => {
                // Int scrutinee only: `idx = v - base`, `switch` over the table
                // (a miss, including negative/wrapped indices, → `default`).
                // `Instance`/other scrutinees → `estep` runs the arm verbatim.
                let v = self.int_opnd(scrut.index() as u32);
                let idx = self.sub(v, self.i64c(*base as i64));
                let dblk = self.tgt_blk(default);
                let cases: Vec<(IntValue, BasicBlock)> = table
                    .iter()
                    .enumerate()
                    .map(|(j, t)| (self.i64c(j as i64), self.tgt_blk(t)))
                    .collect();
                self.b.build_switch(idx, dblk, &cases).unwrap();
            }
            Op::Format { .. } => self.estep(),
            Op::Return { val } => {
                let vp = self.val_ptr(val.index() as u32);
                let do_ret = self.cx.append_basic_block(self.f, "ret");
                self.root_flush(do_ret);
                self.b.position_at_end(do_ret);
                self.settle_seq();
                self.hcall(H::OutReturn, &[self.env.out.into(), vp.into()]);
                self.b.build_return(None).unwrap();
            }
            Op::Panic {} => {
                self.st(self.v.ekind, self.i8c(ERR_PANIC));
                self.br(self.ex.eerr);
            }
            Op::Raise { val } => {
                self.flush_seq();
                self.mark_op(off, next);
                let (regs, s) = (self.regs(), self.i64c(val.index() as i64));
                self.hcall(H::Raise, &[regs.into(), s.into(), self.env.out.into()]);
                self.root_flush(self.ex.eret);
            }
            Op::LoadConst { dst, constant } => match constant {
                Constant::Int(v) => {
                    let v = self.i64c(*v);
                    self.wr_int_dst(dst.index() as u32, v);
                    let nb = self.next_blk(i);
                    self.br(nb);
                }
                Constant::Float(v) => {
                    let v = self.cx.f64_type().const_float(*v);
                    self.wr_float_dst(dst.index() as u32, v);
                    let nb = self.next_blk(i);
                    self.br(nb);
                }
                Constant::Bool(bv) => {
                    let v = self.i8c(*bv as i64);
                    let regs = self.regs();
                    let a = self.vaddr(regs, dst.index() as u32);
                    self.st_bool(a, v);
                    let nb = self.next_blk(i);
                    self.br(nb);
                }
                Constant::Null => {
                    let regs = self.regs();
                    let a = self.vaddr(regs, dst.index() as u32);
                    self.st_null(a);
                    let nb = self.next_blk(i);
                    self.br(nb);
                }
                Constant::Str(id) => {
                    let (regs, d, id) = (
                        self.regs(),
                        self.i64c(dst.index() as i64),
                        self.i32c(id.index() as i64),
                    );
                    self.helper_op_v(
                        i,
                        off,
                        next,
                        H::LoadConstStr,
                        &[
                            regs.into(),
                            d.into(),
                            id.into(),
                            self.env.ctx0.into(),
                            self.env.ctx1.into(),
                            self.env.strs.into(),
                        ],
                    );
                }
                Constant::Array(_) => self.estep(),
            },
            Op::LoadBody { dst, body } => {
                let regs = self.regs();
                let a = self.vaddr(regs, dst.index() as u32);
                let b32 = self.i32c(body.index() as i64);
                self.st_fn(a, b32);
                let nb = self.next_blk(i);
                self.br(nb);
            }
            Op::LoadEntry { dst, slot } => {
                // entry-frame absolute slot: tb = regs - base*VS; v = tb[slot]
                let (regs, base) = (self.regs(), self.env.base);
                let boff = self.mul(base, self.i64c(self.lyt.val_size as i64));
                let tb = self.gep_dyn(regs, self.b.build_int_neg(boff, "").unwrap());
                let vp = self.gep(tb, slot.index() as i64 * self.lyt.val_size as i64);
                let d = self.vaddr(regs, dst.index() as u32);
                self.cpy_val(d, vp);
                let nb = self.next_blk(i);
                self.br(nb);
            }
            Op::StoreEntry { slot, src } => {
                let vp = self.val_ptr(src.index() as u32);
                let (regs, base) = (self.regs(), self.env.base);
                let boff = self.mul(base, self.i64c(self.lyt.val_size as i64));
                let tb = self.gep_dyn(regs, self.b.build_int_neg(boff, "").unwrap());
                let d = self.gep(tb, slot.index() as i64 * self.lyt.val_size as i64);
                self.cpy_val(d, vp);
                let nb = self.next_blk(i);
                self.br(nb);
            }
            Op::BoolEq { dst, left, right } => self.emit_bool(i, *dst, *left, *right, true),
            Op::BoolNe { dst, left, right } => self.emit_bool(i, *dst, *left, *right, false),
            Op::AddInt { dst, left, right } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.int_opnd(right.index() as u32);
                self.emit_checked(i, *dst, a, b, self.intr.sadd_ovf);
            }
            Op::SubInt { dst, left, right } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.int_opnd(right.index() as u32);
                self.emit_checked(i, *dst, a, b, self.intr.ssub_ovf);
            }
            Op::MultInt { dst, left, right } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.int_opnd(right.index() as u32);
                self.emit_checked(i, *dst, a, b, self.intr.smul_ovf);
            }
            Op::ModInt { dst, left, right } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.int_opnd(right.index() as u32);
                self.emit_mod_int(i, *dst, a, b);
            }
            Op::IntLt { dst, left, right } => {
                self.emit_eval_i(i, *dst, *left, *right, IntPredicate::SLT)
            }
            Op::IntLe { dst, left, right } => {
                self.emit_eval_i(i, *dst, *left, *right, IntPredicate::SLE)
            }
            Op::IntGt { dst, left, right } => {
                self.emit_eval_i(i, *dst, *left, *right, IntPredicate::SGT)
            }
            Op::IntGe { dst, left, right } => {
                self.emit_eval_i(i, *dst, *left, *right, IntPredicate::SGE)
            }
            Op::IntEq { dst, left, right } => {
                self.emit_eval_i(i, *dst, *left, *right, IntPredicate::EQ)
            }
            Op::IntNe { dst, left, right } => {
                self.emit_eval_i(i, *dst, *left, *right, IntPredicate::NE)
            }
            Op::AddIntImm { dst, left, val } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.i64c(*val);
                self.emit_checked(i, *dst, a, b, self.intr.sadd_ovf);
            }
            Op::SubIntImm { dst, left, val } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.i64c(*val);
                self.emit_checked(i, *dst, a, b, self.intr.ssub_ovf);
            }
            Op::MultIntImm { dst, left, val } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.i64c(*val);
                self.emit_checked(i, *dst, a, b, self.intr.smul_ovf);
            }
            Op::ModIntImm { dst, left, val } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.i64c(*val);
                self.emit_mod_int(i, *dst, a, b);
            }
            Op::IntLtImm { dst, left, val } => {
                self.emit_eval_imm_i(i, *dst, *left, *val, IntPredicate::SLT)
            }
            Op::IntLeImm { dst, left, val } => {
                self.emit_eval_imm_i(i, *dst, *left, *val, IntPredicate::SLE)
            }
            Op::IntGtImm { dst, left, val } => {
                self.emit_eval_imm_i(i, *dst, *left, *val, IntPredicate::SGT)
            }
            Op::IntGeImm { dst, left, val } => {
                self.emit_eval_imm_i(i, *dst, *left, *val, IntPredicate::SGE)
            }
            Op::IntEqImm { dst, left, val } => {
                self.emit_eval_imm_i(i, *dst, *left, *val, IntPredicate::EQ)
            }
            Op::IntNeImm { dst, left, val } => {
                self.emit_eval_imm_i(i, *dst, *left, *val, IntPredicate::NE)
            }
            Op::AddFloat { dst, left, right } => self.emit_farit(i, *dst, *left, *right, FOp::Add),
            Op::SubFloat { dst, left, right } => self.emit_farit(i, *dst, *left, *right, FOp::Sub),
            Op::MultFloat { dst, left, right } => self.emit_farit(i, *dst, *left, *right, FOp::Mul),
            Op::DivFloat { dst, left, right } => self.emit_farit(i, *dst, *left, *right, FOp::Div),
            Op::FloatLt { dst, left, right } => {
                self.emit_eval_f(i, *dst, *left, *right, FloatPredicate::OLT)
            }
            Op::FloatLe { dst, left, right } => {
                self.emit_eval_f(i, *dst, *left, *right, FloatPredicate::OLE)
            }
            Op::FloatGt { dst, left, right } => {
                self.emit_eval_f(i, *dst, *left, *right, FloatPredicate::OGT)
            }
            Op::FloatGe { dst, left, right } => {
                self.emit_eval_f(i, *dst, *left, *right, FloatPredicate::OGE)
            }
            Op::FloatEq { dst, left, right } => {
                self.emit_eval_f(i, *dst, *left, *right, FloatPredicate::OEQ)
            }
            Op::FloatNe { dst, left, right } => {
                self.emit_eval_f(i, *dst, *left, *right, FloatPredicate::ONE)
            }
            Op::AddFloatImm { dst, left, val } => {
                self.emit_farit_imm(i, *dst, *left, *val, FOp::Add)
            }
            Op::SubFloatImm { dst, left, val } => {
                self.emit_farit_imm(i, *dst, *left, *val, FOp::Sub)
            }
            Op::MultFloatImm { dst, left, val } => {
                self.emit_farit_imm(i, *dst, *left, *val, FOp::Mul)
            }
            Op::ModFloatImm { dst, left, val } => {
                self.emit_farit_imm(i, *dst, *left, *val, FOp::Mod)
            }
            Op::FloatLtImm { dst, left, val } => {
                self.emit_eval_fimm(i, *dst, *left, *val, FloatPredicate::OLT)
            }
            Op::FloatLeImm { dst, left, val } => {
                self.emit_eval_fimm(i, *dst, *left, *val, FloatPredicate::OLE)
            }
            Op::FloatGtImm { dst, left, val } => {
                self.emit_eval_fimm(i, *dst, *left, *val, FloatPredicate::OGT)
            }
            Op::FloatGeImm { dst, left, val } => {
                self.emit_eval_fimm(i, *dst, *left, *val, FloatPredicate::OGE)
            }
            Op::FloatEqImm { dst, left, val } => {
                self.emit_eval_fimm(i, *dst, *left, *val, FloatPredicate::OEQ)
            }
            Op::FloatNeImm { dst, left, val } => {
                self.emit_eval_fimm(i, *dst, *left, *val, FloatPredicate::ONE)
            }
            Op::StrEq { dst, left, right } => {
                self.emit_str_eval(i, off, next, *dst, *left, *right, true)
            }
            Op::StrNe { dst, left, right } => {
                self.emit_str_eval(i, off, next, *dst, *left, *right, false)
            }
            Op::BIntLt {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(
                i,
                target,
                *left,
                Some(*right),
                None,
                *is_true,
                IntPredicate::SLT,
            ),
            Op::BIntLe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(
                i,
                target,
                *left,
                Some(*right),
                None,
                *is_true,
                IntPredicate::SLE,
            ),
            Op::BIntGt {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(
                i,
                target,
                *left,
                Some(*right),
                None,
                *is_true,
                IntPredicate::SGT,
            ),
            Op::BIntGe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(
                i,
                target,
                *left,
                Some(*right),
                None,
                *is_true,
                IntPredicate::SGE,
            ),
            Op::BIntEq {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(i, target, *left, Some(*right), None, *is_true, IntPredicate::EQ),
            Op::BIntNe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(i, target, *left, Some(*right), None, *is_true, IntPredicate::NE),
            Op::BIntLtImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, IntPredicate::SLT),
            Op::BIntLeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, IntPredicate::SLE),
            Op::BIntGtImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, IntPredicate::SGT),
            Op::BIntGeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, IntPredicate::SGE),
            Op::BIntEqImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, IntPredicate::EQ),
            Op::BIntNeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, IntPredicate::NE),
            Op::BFloatLt {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(
                i,
                target,
                *left,
                Some(*right),
                None,
                *is_true,
                FloatPredicate::OLT,
            ),
            Op::BFloatLe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(
                i,
                target,
                *left,
                Some(*right),
                None,
                *is_true,
                FloatPredicate::OLE,
            ),
            Op::BFloatGt {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(
                i,
                target,
                *left,
                Some(*right),
                None,
                *is_true,
                FloatPredicate::OGT,
            ),
            Op::BFloatGe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(
                i,
                target,
                *left,
                Some(*right),
                None,
                *is_true,
                FloatPredicate::OGE,
            ),
            Op::BFloatEq {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(
                i,
                target,
                *left,
                Some(*right),
                None,
                *is_true,
                FloatPredicate::OEQ,
            ),
            Op::BFloatNe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(
                i,
                target,
                *left,
                Some(*right),
                None,
                *is_true,
                FloatPredicate::ONE,
            ),
            Op::BFloatLtImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, FloatPredicate::OLT),
            Op::BFloatLeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, FloatPredicate::OLE),
            Op::BFloatGtImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, FloatPredicate::OGT),
            Op::BFloatGeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, FloatPredicate::OGE),
            Op::BFloatEqImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, FloatPredicate::OEQ),
            Op::BFloatNeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, FloatPredicate::ONE),
            Op::ToFloat { dst, src } => {
                let iv = self.int_opnd(src.index() as u32);
                let f = self
                    .b
                    .build_signed_int_to_float(iv, self.cx.f64_type(), "")
                    .unwrap();
                self.wr_float_dst(dst.index() as u32, f);
                let nb = self.next_blk(i);
                self.br(nb);
            }
            Op::Sqrt { dst, src } => {
                let fv = self.float_opnd(src.index() as u32);
                let r = self
                    .b
                    .build_call(self.intr.sqrt, &[fv.into()], "")
                    .unwrap()
                    .try_as_basic_value()
                    .basic()
                    .unwrap()
                    .into_float_value();
                self.wr_float_dst(dst.index() as u32, r);
                let nb = self.next_blk(i);
                self.br(nb);
            }
            // ---- helper-backed ops ----
            Op::NewArray { dst } => {
                let (regs, d) = (self.regs(), self.i64c(dst.index() as i64));
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::NewArray,
                    &[regs.into(), d.into(), self.env.ctx0.into(), self.env.ctx1.into()],
                );
            }
            Op::NewDict { dst } => {
                let (regs, d) = (self.regs(), self.i64c(dst.index() as i64));
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::NewDict,
                    &[regs.into(), d.into(), self.env.ctx0.into(), self.env.ctx1.into()],
                );
            }
            Op::NewInstance { dst, adt, fields } => {
                let fp = self.reg_list_ptr(fields);
                let (regs, d, a, n) = (
                    self.regs(),
                    self.i64c(dst.index() as i64),
                    self.i32c(adt.index() as i64),
                    self.i64c(fields.len() as i64),
                );
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::NewInstance,
                    &[
                        regs.into(),
                        d.into(),
                        a.into(),
                        fp.into(),
                        n.into(),
                        self.env.ctx0.into(),
                        self.env.ctx1.into(),
                    ],
                );
            }
            Op::NewClosure {
                dst,
                body,
                captures,
            } => {
                let cp = self.reg_list_ptr(captures);
                let (regs, d, b32, n) = (
                    self.regs(),
                    self.i64c(dst.index() as i64),
                    self.i32c(body.index() as i64),
                    self.i64c(captures.len() as i64),
                );
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::NewClosure,
                    &[
                        regs.into(),
                        d.into(),
                        b32.into(),
                        cp.into(),
                        n.into(),
                        self.env.ctx0.into(),
                        self.env.ctx1.into(),
                    ],
                );
            }
            Op::Push { array, value } => {
                let (regs, a, v) = (
                    self.regs(),
                    self.i64c(array.index() as i64),
                    self.i64c(value.index() as i64),
                );
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::Push,
                    &[regs.into(), a.into(), v.into(), self.env.ctx0.into(), self.env.ctx1.into()],
                );
            }
            Op::Insert { dict, key, value } => {
                let (regs, d, k, v) = (
                    self.regs(),
                    self.i64c(dict.index() as i64),
                    self.i32c(key.index() as i64),
                    self.i64c(value.index() as i64),
                );
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::Insert,
                    &[
                        regs.into(),
                        d.into(),
                        k.into(),
                        v.into(),
                        self.env.ctx0.into(),
                        self.env.ctx1.into(),
                        self.env.strs.into(),
                    ],
                );
            }
            Op::SetIndex { set, index, value } => {
                let (regs, s, ii, v) = (
                    self.regs(),
                    self.i64c(set.index() as i64),
                    self.i64c(index.index() as i64),
                    self.i64c(value.index() as i64),
                );
                self.helper_op(
                    i,
                    off,
                    next,
                    H::SetIndex,
                    &[
                        regs.into(),
                        s.into(),
                        ii.into(),
                        v.into(),
                        self.env.ctx0.into(),
                        self.env.ctx1.into(),
                        self.env.out.into(),
                    ],
                );
            }
            Op::GetIndex {
                dst,
                set,
                index,
                kind,
            } => {
                let (regs, d, s, ii, kk) = (
                    self.regs(),
                    self.i64c(dst.index() as i64),
                    self.i64c(set.index() as i64),
                    self.i64c(index.index() as i64),
                    self.i8c(*kind as i64),
                );
                self.helper_op_read(
                    i,
                    off,
                    next,
                    H::GetIndex,
                    &[
                        regs.into(),
                        d.into(),
                        s.into(),
                        ii.into(),
                        kk.into(),
                        self.env.ctx0.into(),
                        self.env.ctx1.into(),
                        self.env.out.into(),
                    ],
                    *dst,
                );
            }
            Op::GetField {
                dst,
                src,
                slot,
                kind,
            } => {
                let (regs, d, s, sl, kk) = (
                    self.regs(),
                    self.i64c(dst.index() as i64),
                    self.i64c(src.index() as i64),
                    self.i64c(*slot as i64),
                    self.i8c(*kind as i64),
                );
                self.helper_op_read(
                    i,
                    off,
                    next,
                    H::GetField,
                    &[regs.into(), d.into(), s.into(), sl.into(), kk.into(), self.env.out.into()],
                    *dst,
                );
            }
            Op::SetField {
                receiver,
                slot,
                value,
            } => {
                let (regs, r, sl, v) = (
                    self.regs(),
                    self.i64c(receiver.index() as i64),
                    self.i64c(*slot as i64),
                    self.i64c(value.index() as i64),
                );
                self.helper_op(
                    i,
                    off,
                    next,
                    H::SetField,
                    &[
                        regs.into(),
                        r.into(),
                        sl.into(),
                        v.into(),
                        self.env.ctx0.into(),
                        self.env.ctx1.into(),
                        self.env.out.into(),
                    ],
                );
            }
            Op::In {
                dst,
                needle,
                haystack,
                condition,
            } => {
                let (regs, d, n, h, c) = (
                    self.regs(),
                    self.i64c(dst.index() as i64),
                    self.i64c(needle.index() as i64),
                    self.i64c(haystack.index() as i64),
                    self.i8c(*condition as i64),
                );
                self.helper_op_v(i, off, next, H::ContainsOp, &[regs.into(), d.into(), n.into(), h.into(), c.into()]);
            }
            Op::IsInstance { dst, src, adt } => {
                let (regs, d, s, a) = (
                    self.regs(),
                    self.i64c(dst.index() as i64),
                    self.i64c(src.index() as i64),
                    self.i32c(adt.index() as i64),
                );
                self.helper_op_v(i, off, next, H::IsInstance, &[regs.into(), d.into(), s.into(), a.into()]);
            }
            Op::IsRaised { dst, src } => {
                let (regs, d, s) = (
                    self.regs(),
                    self.i64c(dst.index() as i64),
                    self.i64c(src.index() as i64),
                );
                self.helper_op_v(i, off, next, H::IsRaised, &[regs.into(), d.into(), s.into()]);
            }
            Op::UnwrapRaised { dst, src } => {
                let (regs, d, s) = (
                    self.regs(),
                    self.i64c(dst.index() as i64),
                    self.i64c(src.index() as i64),
                );
                self.helper_op_v(i, off, next, H::UnwrapRaised, &[regs.into(), d.into(), s.into()]);
            }
            Op::Unwrap { dst, src } => {
                let (regs, d, s) = (
                    self.regs(),
                    self.i64c(dst.index() as i64),
                    self.i64c(src.index() as i64),
                );
                self.helper_op(i, off, next, H::Unwrap, &[regs.into(), d.into(), s.into(), self.env.out.into()]);
            }
            Op::UnwrapUnit { dst, src } => {
                let (regs, d, s) = (
                    self.regs(),
                    self.i64c(dst.index() as i64),
                    self.i64c(src.index() as i64),
                );
                self.helper_op(i, off, next, H::UnwrapUnit, &[regs.into(), d.into(), s.into(), self.env.out.into()]);
            }
            Op::Len { dst, src } => {
                self.flush_seq();
                self.mark_op(off, next);
                let (regs, s, v) = (
                    self.regs(),
                    self.i64c(src.index() as i64),
                    self.slot_i,
                );
                let k = self
                    .hcall(H::Len, &[regs.into(), s.into(), v.into(), self.env.out.into()])
                    .unwrap()
                    .into_int_value();
                let w = self.cx.append_basic_block(self.f, "len.ok");
                self.cbr(self.isz(k), w, self.ex.eret);
                self.b.position_at_end(w);
                let v = self.ldi(self.cx.i64_type(), self.slot_i);
                self.wr_int_dst(dst.index() as u32, v);
                let nb = self.next_blk(i);
                self.br(nb);
            }
            Op::Bin {
                dst,
                left,
                op,
                right,
            } => {
                let (regs, d, l, o, r) = (
                    self.regs(),
                    self.i64c(dst.index() as i64),
                    self.i64c(left.index() as i64),
                    self.i8c(*op as i64),
                    self.i64c(right.index() as i64),
                );
                self.helper_op(
                    i,
                    off,
                    next,
                    H::Bin,
                    &[
                        regs.into(),
                        d.into(),
                        l.into(),
                        o.into(),
                        r.into(),
                        self.env.ctx0.into(),
                        self.env.ctx1.into(),
                        self.env.out.into(),
                    ],
                );
            }
            Op::Unary { dst, op, src } => {
                let (regs, d, o, s) = (
                    self.regs(),
                    self.i64c(dst.index() as i64),
                    self.i8c(*op as i64),
                    self.i64c(src.index() as i64),
                );
                self.helper_op(
                    i,
                    off,
                    next,
                    H::Unary,
                    &[
                        regs.into(),
                        d.into(),
                        o.into(),
                        s.into(),
                        self.env.ctx0.into(),
                        self.env.ctx1.into(),
                        self.env.out.into(),
                    ],
                );
            }
            Op::CallNative { dst, id, args } => {
                let ap = self.reg_list_ptr(args);
                let (regs, d, nid, n) = (
                    self.regs(),
                    self.i64c(dst.index() as i64),
                    self.i32c(id.index() as i64),
                    self.i64c(args.len() as i64),
                );
                self.helper_op(
                    i,
                    off,
                    next,
                    H::CallNative,
                    &[
                        self.env.thread.into(),
                        regs.into(),
                        d.into(),
                        nid.into(),
                        ap.into(),
                        n.into(),
                        self.env.code.into(),
                        self.env.ctx0.into(),
                        self.env.ctx1.into(),
                        self.env.out.into(),
                    ],
                );
            }
            Op::CallDirect { dst, body, args } => {
                self.emit_call_direct(i, off, next, *dst, *body, args);
            }
            Op::Call { dst, callee, args } => {
                self.emit_call(i, off, next, *dst, *callee, args);
            }
        }
    }

    /// A masked `is_bool` answer: a cond reg with a live scalar shadow holds an
    /// `Int`/`Float` — definitely not `Val::Bool(is_true)` — compare-false.
    fn mask_by_shadow(&self, r: Reg, k: IntValue<'c>) -> IntValue<'c> {
        let ok = self
            .v
            .int
            .get(&(r.index() as u32))
            .map(|&(_, o)| o)
            .or_else(|| self.v.float.get(&(r.index() as u32)).map(|&(_, o)| o));
        match ok {
            None => k,
            Some(ok) => {
                let okv = self.ldi(self.cx.bool_type(), ok);
                let not = self.b.build_not(okv, "").unwrap();
                self.b.build_and(k, not, "").unwrap()
            }
        }
    }

    /// `let Some(v) = a.checked_add/sub/mul(b)` via the `with.overflow`
    /// intrinsic — on overflow, `RtErr::IntegerOverflow` with `*op_ip = off`
    /// (the `eerr` block does the store from `cur_ip`).
    fn emit_checked(&mut self, i: usize, dst: Reg, a: IntValue<'c>, b: IntValue<'c>, ovf: FunctionValue<'c>) {
        let r = self
            .b
            .build_call(ovf, &[a.into(), b.into()], "")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_struct_value();
        let v = self
            .b
            .build_extract_value(r, 0, "")
            .unwrap()
            .into_int_value();
        let of = self
            .b
            .build_extract_value(r, 1, "")
            .unwrap()
            .into_int_value();
        let okb = self.cx.append_basic_block(self.f, "ovf.ok");
        let t = self.err_tramp();
        self.cbr(of, t, okb);
        self.b.position_at_end(okb);
        self.wr_int_dst(dst.index() as u32, v);
        let nb = self.next_blk(i);
        self.br(nb);
        self.fill_err(t, ERR_OVFW);
    }

    fn emit_bool(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, eq: bool) {
        // `let Val::Bool(l) = ...` twice — a miss is the interpreter's
        // `unreachable!`; `estep` reproduces it (panic inside `step`).
        let l = self.bool_opnd(left);
        let r = self.bool_opnd(right);
        let cc = if eq { IntPredicate::EQ } else { IntPredicate::NE };
        let c = self.icmp(cc, l, r);
        self.wr_bool_dst(dst.index() as u32, self.i1_to_i8(c));
        let nb = self.next_blk(i);
        self.br(nb);
    }

    fn bool_opnd(&mut self, r: Reg) -> IntValue<'c> {
        let idx = r.index() as u32;
        if let Some(&(_, ok)) = self.v.int.get(&idx).or_else(|| self.v.float.get(&idx)) {
            // a live scalar shadow means the reg is NOT a Bool → estep
            let okv = self.ldi(self.cx.bool_type(), ok);
            let notok = self.b.build_not(okv, "").unwrap();
            let nb = self.cx.append_basic_block(self.f, "bop.nb");
            self.cbr(notok, nb, self.ex.estep);
            self.b.position_at_end(nb);
        }
        let good = self.cx.append_basic_block(self.f, "bop.ok");
        let regs = self.regs();
        let a = self.vaddr(regs, idx);
        let t = self.ld_tag(a);
        let ok = self.icmp(IntPredicate::EQ, t, self.tconst(self.lyt.t_bool));
        self.cbr(ok, good, self.ex.estep);
        self.b.position_at_end(good);
        self.ldi(self.cx.i8_type(), self.gep(a, self.lyt.bool_pay as i64))
    }

    fn emit_mod_int(&mut self, i: usize, dst: Reg, a: IntValue<'c>, b: IntValue<'c>) {
        // b == 0 → Err(ModByZero); i64::MIN % -1 panics in the interpreter —
        // route that pair to `estep` so `step` hits the identical panic.
        let bz = self.isz(b);
        let run = self.cx.append_basic_block(self.f, "mod.run");
        let tramp = self.err_tramp();
        self.cbr(bz, tramp, run);
        self.b.position_at_end(run);
        let amin = self.icmp(IntPredicate::EQ, a, self.i64c(i64::MIN));
        let bm1 = self.icmp(IntPredicate::EQ, b, self.i64c(-1));
        let bad = self.b.build_and(amin, bm1, "").unwrap();
        let run2 = self.cx.append_basic_block(self.f, "mod.run2");
        self.cbr(bad, self.ex.estep, run2);
        self.b.position_at_end(run2);
        let v = self.b.build_int_signed_rem(a, b, "").unwrap();
        self.wr_int_dst(dst.index() as u32, v);
        let nb = self.next_blk(i);
        self.br(nb);
        self.fill_err(tramp, ERR_MOD0);
    }

    fn emit_eval_i(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, cc: IntPredicate) {
        let a = self.int_opnd(left.index() as u32);
        let b = self.int_opnd(right.index() as u32);
        let c = self.icmp(cc, a, b);
        self.wr_bool_dst(dst.index() as u32, self.i1_to_i8(c));
        let nb = self.next_blk(i);
        self.br(nb);
    }

    fn emit_eval_imm_i(&mut self, i: usize, dst: Reg, left: Reg, val: i64, cc: IntPredicate) {
        let a = self.int_opnd(left.index() as u32);
        let b = self.i64c(val);
        let c = self.icmp(cc, a, b);
        self.wr_bool_dst(dst.index() as u32, self.i1_to_i8(c));
        let nb = self.next_blk(i);
        self.br(nb);
    }

    fn emit_eval_f(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, cc: FloatPredicate) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.float_opnd(right.index() as u32);
        let c = self.b.build_float_compare(cc, a, b, "").unwrap();
        self.wr_bool_dst(dst.index() as u32, self.i1_to_i8(c));
        let nb = self.next_blk(i);
        self.br(nb);
    }

    fn emit_eval_fimm(&mut self, i: usize, dst: Reg, left: Reg, val: i64, cc: FloatPredicate) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.f64c(val);
        let c = self.b.build_float_compare(cc, a, b, "").unwrap();
        self.wr_bool_dst(dst.index() as u32, self.i1_to_i8(c));
        let nb = self.next_blk(i);
        self.br(nb);
    }

    fn emit_farit(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, o: FOp) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.float_opnd(right.index() as u32);
        let v = o.emit(&self.b, a, b);
        self.wr_float_dst(dst.index() as u32, v);
        let nb = self.next_blk(i);
        self.br(nb);
    }

    fn emit_farit_imm(&mut self, i: usize, dst: Reg, left: Reg, val: i64, o: FOp) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.f64c(val);
        let v = o.emit(&self.b, a, b);
        self.wr_float_dst(dst.index() as u32, v);
        let nb = self.next_blk(i);
        self.br(nb);
    }

    /// `StrEq`/`StrNe`: the (Str, Str) fast path via `bin_str`; anything else
    /// runs the op through `step`.
    fn emit_str_eval(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        dst: Reg,
        left: Reg,
        right: Reg,
        eq: bool,
    ) {
        self.flush_seq();
        self.mark_op(off, next);
        let (regs, d, l, r, e) = (
            self.regs(),
            self.i64c(dst.index() as i64),
            self.i64c(left.index() as i64),
            self.i64c(right.index() as i64),
            self.i8c(eq as i64),
        );
        // 0 → wrote regs[dst]; 1 → both-Str fast path missed → `step`
        let k = self
            .hcall(H::BinStr, &[regs.into(), d.into(), l.into(), r.into(), e.into()])
            .unwrap()
            .into_int_value();
        let nb = self.next_blk(i);
        self.cbr(self.isz(k), nb, self.ex.estep);
    }

    /// `B*` register/imm branch ops: `hit = a cc b`, then take `target` iff
    /// `hit == is_true`.
    #[allow(clippy::too_many_arguments)]
    fn emit_brr(
        &mut self,
        i: usize,
        target: &BlockTarget,
        left: Reg,
        right: Option<Reg>,
        imm: Option<i64>,
        is_true: bool,
        cc: IntPredicate,
    ) {
        let a = self.int_opnd(left.index() as u32);
        let b = match right {
            Some(r) => self.int_opnd(r.index() as u32),
            None => self.i64c(imm.unwrap()),
        };
        let hit = self.icmp(cc, a, b);
        let (tb, nb) = (self.tgt_blk(target), self.next_blk(i));
        let (t, f) = if is_true { (tb, nb) } else { (nb, tb) };
        self.cbr(hit, t, f);
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_brf(
        &mut self,
        i: usize,
        target: &BlockTarget,
        left: Reg,
        right: Option<Reg>,
        imm: Option<i64>,
        is_true: bool,
        cc: FloatPredicate,
    ) {
        let a = self.float_opnd(left.index() as u32);
        let b = match right {
            Some(r) => self.float_opnd(r.index() as u32),
            None => self.f64c(imm.unwrap()),
        };
        let hit = self.b.build_float_compare(cc, a, b, "").unwrap();
        let (tb, nb) = (self.tgt_blk(target), self.next_blk(i));
        let (t, f) = if is_true { (tb, nb) } else { (nb, tb) };
        self.cbr(hit, t, f);
    }

    /// `Move` — bcgen's flag-propagating copy for same-kind shadow pairs, the
    /// `wr_val` shape for mixed, a plain copy otherwise.
    fn emit_move(&mut self, dst: Reg, src: Reg) {
        let d = dst.index() as u32;
        let s = src.index() as u32;
        let di = self.v.int.get(&d).copied();
        let df = self.v.float.get(&d).copied();
        let si = self.v.int.get(&s).copied();
        let sf = self.v.float.get(&s).copied();
        if di.is_some() && si.is_some() {
            // `d = s; dok = sok; if !dok { wr(d, rd(s)) }`
            let (dsv, dok) = di.unwrap();
            let (ssv, sok) = si.unwrap();
            let v = self.ldi(self.cx.i64_type(), ssv);
            let k = self.ldi(self.cx.bool_type(), sok);
            self.st(dsv, v);
            self.st(dok, k);
            let c = self.cx.append_basic_block(self.f, "mv.c");
            let w = self.cx.append_basic_block(self.f, "mv.w");
            let dokv = self.ldi(self.cx.bool_type(), dok);
            self.cbr(dokv, c, w);
            self.b.position_at_end(w);
            let regs = self.regs();
            let vp = self.vaddr(regs, s);
            let dd = self.vaddr(regs, d);
            self.cpy_val(dd, vp);
            self.br(c);
            self.b.position_at_end(c);
        } else if df.is_some() && sf.is_some() {
            let (dsv, dok) = df.unwrap();
            let (ssv, sok) = sf.unwrap();
            let v = self.ldf(ssv);
            let k = self.ldi(self.cx.bool_type(), sok);
            self.st(dsv, v);
            self.st(dok, k);
            let c = self.cx.append_basic_block(self.f, "mv.c");
            let w = self.cx.append_basic_block(self.f, "mv.w");
            let dokv = self.ldi(self.cx.bool_type(), dok);
            self.cbr(dokv, c, w);
            self.b.position_at_end(w);
            let regs = self.regs();
            let vp = self.vaddr(regs, s);
            let dd = self.vaddr(regs, d);
            self.cpy_val(dd, vp);
            self.br(c);
            self.b.position_at_end(c);
        } else if let Some(dv) = di {
            // dst int-shadowed, src not int: `match v { Int(x) => shadow, v
            // => { ok=0; wr(d,v) } }` — bcgen's `wr_val` arm.
            self.emit_move_wrv(d, s, dv, sf, true);
        } else if let Some(dv) = df {
            self.emit_move_wrv(d, s, dv, si, false);
        } else {
            // unshadowed dst: materialize a live shadow src into regs first
            let vp = self.val_ptr(s);
            let regs = self.regs();
            let dd = self.vaddr(regs, d);
            self.cpy_val(dd, vp);
        }
    }

    /// dst scalar-shadowed / src not same-kind-shadowed: probe the (possibly
    /// opposite-kind-shadowed) src for the right tag; hit → shadow write,
    /// miss → `ok = 0; wr(d, regs[s])`.
    fn emit_move_wrv(
        &mut self,
        d: u32,
        s: u32,
        dv: (PointerValue<'c>, PointerValue<'c>),
        osrc: Option<(PointerValue<'c>, PointerValue<'c>)>,
        dst_is_int: bool,
    ) {
        let (dsv, dok) = dv;
        let miss = self.cx.append_basic_block(self.f, "mvw.miss");
        let good = self.cx.append_basic_block(self.f, "mvw.ok");
        let done = self.cx.append_basic_block(self.f, "mvw.done");
        if let Some((ssv, sok)) = osrc {
            // src carries a live opposite-kind shadow → the Val is that scalar
            // → miss arm writes it through directly.
            let okv = self.ldi(self.cx.bool_type(), sok);
            let genb = self.cx.append_basic_block(self.f, "mvw.gen");
            let fmiss = self.cx.append_basic_block(self.f, "mvw.fmiss");
            self.cbr(okv, fmiss, genb);
            self.b.position_at_end(fmiss);
            let regs = self.regs();
            let dd = self.vaddr(regs, d);
            if dst_is_int {
                let v = self.ldf(ssv);
                self.st_float(dd, v);
            } else {
                let v = self.ldi(self.cx.i64_type(), ssv);
                self.st_int(dd, v);
            }
            self.st(dok, self.cx.bool_type().const_int(0, false));
            self.br(done);
            self.b.position_at_end(genb);
        }
        // generic: probe the authoritative slot for the wanted tag
        let want = if dst_is_int {
            self.lyt.t_int
        } else {
            self.lyt.t_float
        };
        let regs = self.regs();
        let sa = self.vaddr(regs, s);
        let t = self.ld_tag(sa);
        let k = self.icmp(IntPredicate::EQ, t, self.tconst(want));
        self.cbr(k, good, miss);
        self.b.position_at_end(good);
        let v: BasicValueEnum = if dst_is_int {
            self.ldi(self.cx.i64_type(), self.gep(sa, self.lyt.val_pay as i64))
                .into()
        } else {
            self.ldf(self.gep(sa, self.lyt.val_pay as i64)).into()
        };
        self.st(dsv, v);
        self.st(dok, self.cx.bool_type().const_int(1, false));
        self.br(done);
        self.b.position_at_end(miss);
        let regs = self.regs();
        let vp = self.vaddr(regs, s);
        let dd = self.vaddr(regs, d);
        self.cpy_val(dd, vp);
        self.st(dok, self.cx.bool_type().const_int(0, false));
        self.br(done);
        self.b.position_at_end(done);
    }

    /// `CallDirect`/`Call` — both go through the `mj_call_body`/`mj_call_dyn`
    /// megashims (enter + run + pop in one FFI hop; the bodies table is the
    /// Rust-side `tbl` passed as an `inttoptr` constant). The Cranelift
    /// emitter's inline frame push is *not* ported — the shim still enters
    /// the callee's JIT body directly, so this is one FFI call per call.
    fn emit_call_direct(&mut self, i: usize, off: usize, next: usize, dst: Reg, body: compile::BodyId, args: &[Reg]) {
        let ap = self.reg_list_ptr(args);
        let nargs = args.len();
        self.flush_seq();
        // `code.ip = next; *op_ip = off` before the call (the shim's frame
        // push saves `code.ip` as the caller resume slot; errors locate via
        // op_ip).
        self.mark_op(off, next);
        // the callee draws fuel/ops_left through its own `bcn` — charge our
        // spent ops first so its reads are exact
        self.settle_seq();
        let (b, d, n) = (
            self.i64c(body.index() as i64),
            self.i64c(dst.index() as i64),
            self.i64c(nargs as i64),
        );
        let rp = self
            .hcall(
                H::CallBody,
                &[
                    self.env.thread.into(),
                    self.env.code.into(),
                    self.env.chunks.into(),
                    self.env.bodies_tbl.into(),
                    b.into(),
                    d.into(),
                    ap.into(),
                    n.into(),
                    self.env.ctx0.into(),
                    self.env.ctx1.into(),
                    self.env.strs.into(),
                    self.env.sigs.into(),
                    self.env.fuel_p.into(),
                    self.env.opip_p.into(),
                    self.env.out.into(),
                ],
            )
            .unwrap()
            .into_pointer_value();
        self.post_call(i, dst, rp);
    }

    /// `Call` — one `mj_call_dyn` hop: callee resolution (incl. the
    /// `CallTarget::Value` signature check), depth cap, enter/run/pop.
    fn emit_call(&mut self, i: usize, off: usize, next: usize, dst: Reg, callee: Reg, args: &[Reg]) {
        // Flush first — the shim reads the callee Val and arg slots out of
        // the window. `code.ip = next; *op_ip = off` before the call.
        self.flush_seq();
        self.mark_op(off, next);
        self.settle_seq();
        let ap = self.reg_list_ptr(args);
        let nargs = args.len();
        let (regs, c, d, n) = (
            self.regs(),
            self.i64c(callee.index() as i64),
            self.i64c(dst.index() as i64),
            self.i64c(nargs as i64),
        );
        let rp = self
            .hcall(
                H::CallDyn,
                &[
                    self.env.thread.into(),
                    self.env.code.into(),
                    self.env.chunks.into(),
                    self.env.sigs.into(),
                    regs.into(),
                    c.into(),
                    d.into(),
                    ap.into(),
                    n.into(),
                    self.env.bodies_tbl.into(),
                    self.env.ctx0.into(),
                    self.env.ctx1.into(),
                    self.env.strs.into(),
                    self.env.fuel_p.into(),
                    self.env.opip_p.into(),
                    self.env.out.into(),
                ],
            )
            .unwrap()
            .into_pointer_value();
        self.post_call(i, dst, rp);
    }

    /// After a megashim call returns: null means propagate `out` verbatim
    /// (`eret`); otherwise the returned pointer is the caller's rebuilt
    /// window — re-pin it, re-arm the quota (the callee consumed
    /// fuel/ops_left through its own `bcn`), refresh dst's shadow, and resume
    /// at the next op.
    fn post_call(&mut self, i: usize, dst: Reg, rp: PointerValue<'c>) {
        let rpi = self
            .b
            .build_ptr_to_int(rp, self.cx.i64_type(), "")
            .unwrap();
        let isz = self.isz(rpi);
        let resumed = self.cx.append_basic_block(self.f, "call.resume");
        self.cbr(isz, self.ex.eret, resumed);
        self.b.position_at_end(resumed);
        self.rearm_seq();
        self.set_regs(rp);
        self.dst_refresh(i, dst, rp);
    }

    /// Post-call tail: `regs[dst]` now holds the callee's return value —
    /// refresh its scalar shadow (if any) and resume at the next op.
    fn dst_refresh(&mut self, i: usize, dst: Reg, regs2: PointerValue<'c>) {
        let d = dst.index() as u32;
        if let Some(&(sv, ok)) = self.v.int.get(&d) {
            let a = self.vaddr(regs2, d);
            let t = self.ld_tag(a);
            let k = self.icmp(IntPredicate::EQ, t, self.tconst(self.lyt.t_int));
            let v = self.ldi(self.cx.i64_type(), self.gep(a, self.lyt.val_pay as i64));
            let z = self.i64c(0);
            let sv0 = self.sel(k, v.into(), z.into());
            self.st(sv, sv0);
            self.st(ok, k);
        } else if let Some(&(sv, ok)) = self.v.float.get(&d) {
            let a = self.vaddr(regs2, d);
            let t = self.ld_tag(a);
            let k = self.icmp(IntPredicate::EQ, t, self.tconst(self.lyt.t_float));
            let v = self.ldf(self.gep(a, self.lyt.val_pay as i64));
            let z = self.cx.f64_type().const_float(0.0);
            let sv0 = self.sel(k, v.into(), z.into());
            self.st(sv, sv0);
            self.st(ok, k);
        }
        let nb = self.next_blk(i);
        self.br(nb);
    }
}

#[derive(Clone, Copy)]
enum FOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

impl FOp {
    fn emit<'c>(self, b: &Builder<'c>, a: FloatValue<'c>, bv: FloatValue<'c>) -> FloatValue<'c> {
        match self {
            FOp::Add => b.build_float_add(a, bv, "").unwrap(),
            FOp::Sub => b.build_float_sub(a, bv, "").unwrap(),
            FOp::Mul => b.build_float_mul(a, bv, "").unwrap(),
            FOp::Div => b.build_float_div(a, bv, "").unwrap(),
            // Rust `%` on f64 → fmod semantics
            FOp::Mod => b.build_float_rem(a, bv, "").unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn empty_program_compiles() {
        let program = compile::Program {
            entry: compile::BodyId::from(0u32),
            chunks: Default::default(),
            signatures: Default::default(),
            strs: Default::default(),
            bytes: vec![],
            root: Default::default(),
            struct_names: Default::default(),
            methods: Default::default(),
            field_names: Default::default(),
            tests: vec![],
        };
        let jit = crate::compile(&program).unwrap();
        assert!(jit.bodies().is_empty());
    }


}

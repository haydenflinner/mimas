//! Per-chunk CLIF emission — the JIT twin of `bcgen`'s `emit_op`.
//!
//! Each body compiles to one `extern "C"` fn at the `BodyFn` ABI:
//!
//! ```text
//! entry:   hoist env pointers, init scalar shadows from the reg window
//! dispatch: code.ip → dense op index (ip2idx data table) → br_table
//! op_i:    [paused/fuel/ops_left preamble] → op semantics → jump successor
//! writeback: every observable boundary inlines the shadow writeback
//!          (`flush_seq`) so the window is authoritative before helpers,
//!          calls, and exits
//! exits:   enext (pause/fuel) / eoof (ops budget) / estep (interpreter
//!          fallback) / eerr (RtErr) / eret (propagate `out`)
//! ```
//!
//! `code.ip` stays a *byte offset* — the dispatcher maps it through a u32
//! lookup blob (`mj_ip2idx`) to the dense op index `br_table` wants. In-op flow
//! is direct block-to-block jumps; the dispatcher is only re-entered after a
//! `step` fallback (which may jump anywhere) and at body entry.
//!
//! Scalar shadows are `Variable`s (`def_var`/`use_var`, the frontend does SSA
//! construction); every boundary that can observe `thread.regs` — exits,
//! helper calls, inlined calls — routes through FLUSH so the window is
//! authoritative exactly when the interpreter contract needs it.

use std::collections::{HashMap, HashSet};

use compile::{BlockTarget, Constant, Op, Program, Reg};
use cranelift_codegen::Context;
use cranelift_codegen::ir::{
    self, AbiParam, Block, FuncRef, InstBuilder, JumpTableData, MemFlagsData, Signature, StackSlot,
    StackSlotData, StackSlotKind, Value, types,
};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::JITModule;
use cranelift_module::{DataDescription, DataId, FuncId, Module, ModuleResult};
use vm::bc::INLINE_CALL_DEPTH;

use crate::H;

const I64: ir::Type = types::I64;
const I32: ir::Type = types::I32;
const I8: ir::Type = types::I8;
const F64: ir::Type = types::F64;

fn tf() -> MemFlagsData {
    MemFlagsData::trusted()
}

/// `size_of::<Val>()` — the window stride.
fn val_size() -> i64 {
    std::mem::size_of::<vm::bc::Val<'static>>() as i64
}

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
            // every other op that carries a dst writes a dynamically-typed value
            Op::GetField { dst, .. }
            | Op::GetIndex { dst, .. }
            | Op::LoadBody { dst, .. }
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
            let ok_i = ws.iter().all(|w| match w {
                W::Int => true,
                W::Copy(src) => int.contains(src),
                _ => false,
            });
            let ok_f = ws.iter().all(|w| match w {
                W::Float => true,
                W::Copy(src) => float.contains(src),
                _ => false,
            });
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

/// Immutable body-wide values: the `BodyFn` params plus pointers hoisted once
/// in `entry` (the cells live in `State`/`ThreadState`, stable for the call).
#[derive(Clone, Copy)]
struct Env {
    thread: Value,
    code: Value,
    ctx0: Value,
    ctx1: Value,
    strs: Value,
    chunks: Value,
    sigs: Value,
    fuel_p: Value,
    opip_p: Value,
    out: Value,
    paused_p: Value,
    opsleft_p: Value,
    ip_p: Value,
    base: Value,
    nregs: Value,
    /// `mj_bodies` data-object address — `body_id`-indexed fn-ptr table.
    bodies_tbl: Value,
}

/// The body's mutable state as frontend `Variable`s — `def_var`/`use_var` and
/// the frontend's SSA construction handle the joins.
struct Vs {
    /// Window base pointer (`thread.regs + base`), refreshed after resizes.
    regs: Variable,
    /// Byte offset of the op currently executing (for `code.ip`/`*op_ip`).
    cur_ip: Variable,
    /// `RtErr` code for the `eerr` exit.
    ekind: Variable,
    /// Scalar shadows: `(value, ok)` vars per register.
    int: HashMap<u32, (Variable, Variable)>,
    float: HashMap<u32, (Variable, Variable)>,
}

/// Shared exit/trampoline blocks.
struct Ex {
    #[allow(dead_code)]
    dispatch: Block,
    /// `code.ip` isn't a known op start of this chunk → hand it to `step`.
    #[allow(dead_code)]
    edef: Block,
    enext: Block,
    eoof: Block,
    estep: Block,
    eerr: Block,
    /// Propagate `out` verbatim (inlined-callee results, helper errors).
    eret: Block,
    /// Fallthrough past the last op: `code.ip = usize::MAX` → `step` (the
    /// interpreter decodes garbage exactly as bcgen's `_ =>` arm does).
    eend: Block,
}

struct Em<'a> {
    fb: FunctionBuilder<'a>,
    hrefs: Vec<FuncRef>,
    brefs: Vec<FuncRef>,
    env: Env,
    v: Vs,
    ex: Ex,
    sh: &'a Sh,
    ops: &'a [(usize, Op)],
    blocks: Vec<Block>,
    off2idx: HashMap<usize, usize>,
    slot_i: StackSlot,
    slot_f: StackSlot,
    slot_b: StackSlot,
    slot_c: StackSlot,
}

impl Em<'_> {
    // ---- small IR helpers ----

    fn iconst(&mut self, v: i64) -> Value {
        self.fb.ins().iconst(I64, v)
    }

    fn iconst8(&mut self, v: i64) -> Value {
        self.fb.ins().iconst(I8, v)
    }

    fn iconst32(&mut self, v: i64) -> Value {
        self.fb.ins().iconst(I32, v)
    }

    fn regs(&mut self) -> Value {
        self.fb.use_var(self.v.regs)
    }

    /// Call a `vm::bc::jit` helper; returns the `u8` status if the signature
    /// has one.
    fn hcall(&mut self, h: H, args: &[Value]) -> Option<Value> {
        let inst = self.fb.ins().call(self.hrefs[h as usize], args);
        self.fb.inst_results(inst).first().copied()
    }

    /// The block implementing the op at byte offset `t` — jump targets always
    /// land on op starts, so the dense map is exact.
    fn tgt_blk(&self, t: &BlockTarget) -> Block {
        let BlockTarget::ByteOffset(o) = t else {
            panic!("unresolved BlockTarget in compiled program")
        };
        self.blocks[self.off2idx[o]]
    }

    /// Successor block for fallthrough (`next` op, or `eend` off the end).
    fn next_blk(&self, i: usize) -> Block {
        if i + 1 < self.ops.len() {
            self.blocks[i + 1]
        } else {
            self.ex.eend
        }
    }

    /// "Flush shadows, then jump to `cont`" — the inline writeback, used when
    /// control must rejoin a *shared* block (e.g. `eret`) rather than continue
    /// in the current chain.
    fn flush_jump(&mut self, cont: Block) {
        self.flush_seq();
        self.fb.ins().jump(cont, &[]);
    }

    /// The inline shadow-writeback sequence — used inside shared exits (which
    /// are already merge points, so no `fret` dispatch needed there).
    fn flush_seq(&mut self) {
        let mut ints: Vec<u32> = self.sh.int.iter().copied().collect();
        ints.sort_unstable();
        for r in ints {
            let (sv, ok) = self.v.int[&r];
            let okv = self.fb.use_var(ok);
            let (w, c) = (self.fb.create_block(), self.fb.create_block());
            self.fb.ins().brif(okv, w, &[], c, &[]);
            self.fb.switch_to_block(w);
            let (regs, idx, v) = (self.regs(), self.iconst(r as i64), self.fb.use_var(sv));
            self.hcall(H::WrI, &[regs, idx, v]);
            self.fb.ins().jump(c, &[]);
            self.fb.switch_to_block(c);
        }
        let mut floats: Vec<u32> = self.sh.float.iter().copied().collect();
        floats.sort_unstable();
        for r in floats {
            let (sv, ok) = self.v.float[&r];
            let okv = self.fb.use_var(ok);
            let (w, c) = (self.fb.create_block(), self.fb.create_block());
            self.fb.ins().brif(okv, w, &[], c, &[]);
            self.fb.switch_to_block(w);
            let (regs, idx, v) = (self.regs(), self.iconst(r as i64), self.fb.use_var(sv));
            self.hcall(H::WrF, &[regs, idx, v]);
            self.fb.ins().jump(c, &[]);
            self.fb.switch_to_block(c);
        }
    }

    /// `code.ip = v` through the hoisted cell pointer.
    fn store_ip(&mut self, v: Value) {
        self.fb.ins().store(tf(), v, self.env.ip_p, 0);
    }

    /// `*op_ip = v` through the hoisted cell pointer.
    fn store_opip(&mut self, v: Value) {
        self.fb.ins().store(tf(), v, self.env.opip_p, 0);
    }

    /// `code.ip = next; *op_ip = off` — bcgen's pre-helper/pre-call idiom.
    fn mark_op(&mut self, off: usize, next: usize) {
        let n = self.iconst(next as i64);
        let o = self.iconst(off as i64);
        self.store_ip(n);
        self.store_opip(o);
    }

    /// Route an `RtErr` kind to the `eerr` exit through a tiny trampoline that
    /// just sets `ekind` — `brif` targets can't carry the constant themselves.
    /// The block is created empty and *returned* for use as a `brif` target;
    /// call [`Em::fill_err`] after the current path is terminated (switching
    /// mid-emission from a partially-filled block is illegal).
    fn err_tramp(&mut self) -> Block {
        self.fb.create_block()
    }

    /// Fill a trampoline created by [`Em::err_tramp`]: `ekind = kind; →eerr`.
    /// Switches to `t` (must be called when the current block is terminated).
    fn fill_err(&mut self, t: Block, kind: i64) {
        self.fb.switch_to_block(t);
        let k = self.iconst8(kind);
        self.fb.def_var(self.v.ekind, k);
        self.fb.ins().jump(self.ex.eerr, &[]);
    }

    /// Read `regs[r]` as an i64 — shadow when eligible, else the `ri` helper.
    /// A miss routes to `estep`: `step` runs the op verbatim — the same
    /// outcome bcgen's `bin_cold`/`branch_cold` tail-calls produce.
    fn int_opnd(&mut self, r: u32) -> Value {
        let good = self.fb.create_block();
        if let Some(&(sv, ok)) = self.v.int.get(&r) {
            let okv = self.fb.use_var(ok);
            self.fb.ins().brif(okv, good, &[], self.ex.estep, &[]);
            self.fb.switch_to_block(good);
            self.fb.use_var(sv)
        } else {
            let (regs, idx, addr) = (
                self.regs(),
                self.iconst(r as i64),
                self.fb.ins().stack_addr(I64, self.slot_i, 0),
            );
            let ok = self.hcall(H::Ri, &[regs, idx, addr]).unwrap();
            self.fb.ins().brif(ok, good, &[], self.ex.estep, &[]);
            self.fb.switch_to_block(good);
            self.fb.ins().stack_load(I64, I64, self.slot_i, 0)
        }
    }

    fn float_opnd(&mut self, r: u32) -> Value {
        let good = self.fb.create_block();
        if let Some(&(sv, ok)) = self.v.float.get(&r) {
            let okv = self.fb.use_var(ok);
            self.fb.ins().brif(okv, good, &[], self.ex.estep, &[]);
            self.fb.switch_to_block(good);
            self.fb.use_var(sv)
        } else {
            let (regs, idx, addr) = (
                self.regs(),
                self.iconst(r as i64),
                self.fb.ins().stack_addr(I64, self.slot_f, 0),
            );
            let ok = self.hcall(H::Rf, &[regs, idx, addr]).unwrap();
            self.fb.ins().brif(ok, good, &[], self.ex.estep, &[]);
            self.fb.switch_to_block(good);
            self.fb.ins().stack_load(I64, F64, self.slot_f, 0)
        }
    }

    /// Scalar write into `dst`: shadow-var update when shadowed, else `wr_*`.
    fn wr_int_dst(&mut self, d: u32, v: Value) {
        if let Some(&(sv, ok)) = self.v.int.get(&d) {
            self.fb.def_var(sv, v);
            let one = self.iconst8(1);
            self.fb.def_var(ok, one);
        } else {
            let (regs, idx) = (self.regs(), self.iconst(d as i64));
            self.hcall(H::WrI, &[regs, idx, v]);
        }
    }

    fn wr_float_dst(&mut self, d: u32, v: Value) {
        if let Some(&(sv, ok)) = self.v.float.get(&d) {
            self.fb.def_var(sv, v);
            let one = self.iconst8(1);
            self.fb.def_var(ok, one);
        } else {
            let (regs, idx) = (self.regs(), self.iconst(d as i64));
            self.hcall(H::WrF, &[regs, idx, v]);
        }
    }

    /// Bool write into `dst` — eval dsts are always `W::Dyn` (unshadowed).
    fn wr_bool_dst(&mut self, d: u32, v8: Value) {
        let (regs, idx) = (self.regs(), self.iconst(d as i64));
        self.hcall(H::WrB, &[regs, idx, v8]);
    }

    /// A `*const Val` for reg `s`, materializing a live shadow first — after
    /// this `regs[s]` is authoritative for the read.
    fn val_ptr(&mut self, s: u32) -> Value {
        if let Some(&(sv, ok)) = self.v.int.get(&s) {
            let c = self.fb.create_block();
            let m = self.fb.create_block();
            let okv = self.fb.use_var(ok);
            self.fb.ins().brif(okv, m, &[], c, &[]);
            self.fb.switch_to_block(m);
            let (regs, idx, v) = (self.regs(), self.iconst(s as i64), self.fb.use_var(sv));
            self.hcall(H::WrI, &[regs, idx, v]);
            self.fb.ins().jump(c, &[]);
            self.fb.switch_to_block(c);
        } else if let Some(&(sv, ok)) = self.v.float.get(&s) {
            let c = self.fb.create_block();
            let m = self.fb.create_block();
            let okv = self.fb.use_var(ok);
            self.fb.ins().brif(okv, m, &[], c, &[]);
            self.fb.switch_to_block(m);
            let (regs, idx, v) = (self.regs(), self.iconst(s as i64), self.fb.use_var(sv));
            self.hcall(H::WrF, &[regs, idx, v]);
            self.fb.ins().jump(c, &[]);
            self.fb.switch_to_block(c);
        }
        let (regs, idx) = (self.regs(), self.iconst(s as i64));
        self.hcall(H::Rval, &[regs, idx]).unwrap()
    }

    /// Helper-op shape: inline shadow writeback, then stores `code.ip`/
    /// `*op_ip`, calls the `u8` helper, then `1`→`eret` / `0`→fallthrough.
    /// `args` are built by the caller before this runs — the writeback blocks
    /// are all dominated by the current block, so they stay usable.
    fn helper_op(&mut self, i: usize, off: usize, next: usize, h: H, args: &[Value]) {
        self.flush_seq();
        self.mark_op(off, next);
        let k = self.hcall(h, args).unwrap();
        let nb = self.next_blk(i);
        self.fb.ins().brif(k, self.ex.eret, &[], nb, &[]);
    }

    /// Same but the helper returns void — always continues.
    fn helper_op_v(&mut self, i: usize, off: usize, next: usize, h: H, args: &[Value]) {
        self.flush_seq();
        self.mark_op(off, next);
        self.hcall(h, args);
        { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
    }

    /// A `u32` register-index list (call args / field regs / captures) spilled
    /// to a stack slot; returns the slot address for the helper.
    fn reg_list_slot(&mut self, regs_idx: &[Reg]) -> Value {
        let size = (regs_idx.len().max(1) * 4) as u32;
        let slot = self
            .fb
            .create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size, 3));
        for (j, r) in regs_idx.iter().enumerate() {
            let v = self.iconst32(r.index() as i64);
            self.fb.ins().stack_store(I64, v, slot, (j * 4) as i32);
        }
        self.fb.ins().stack_addr(I64, slot, 0)
    }

    /// `if frames.len() == 1 { flush }` before producing `Flow::Return` —
    /// the host reads the root frame's regs after `run()`. Continues in
    /// `cont` either way.
    fn root_flush(&mut self, cont: Block) {
        let flen = self.hcall(H::FramesLen, &[self.env.thread]).unwrap();
        let one = self.iconst(1);
        let is_root = self.fb.ins().icmp(IntCC::Equal, flen, one);
        let ft = self.fb.create_block();
        self.fb.ins().brif(is_root, ft, &[], cont, &[]);
        self.fb.switch_to_block(ft);
        self.flush_jump(cont);
    }

    /// `StepAt`-backed whole-op fallback: nothing is stored here — `estep`
    /// itself writes `code.ip`/`*op_ip` from `cur_ip` and runs `step`.
    fn estep(&mut self) {
        self.fb.ins().jump(self.ex.estep, &[]);
    }
}

#[allow(dead_code)]
fn tgt(t: &BlockTarget) -> usize {
    let BlockTarget::ByteOffset(o) = t else {
        panic!("unresolved BlockTarget in compiled program")
    };
    *o
}

/// Compile `chunks[body]` into `ctx.func` and define it as `body_ids[body]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_body(
    module: &mut JITModule,
    program: &Program,
    body: usize,
    body_sig: &Signature,
    helper_ids: &[FuncId],
    body_ids: &[FuncId],
    bodies_data: DataId,
    fbc: &mut FunctionBuilderContext,
    ctx: &mut Context,
) -> ModuleResult<()> {
    let body_id = compile::BodyId::from(body as u32);
    let ops = program.ops(body_id);
    let chunk = &program.chunks[body_id];
    let chunk_off = chunk.offset;
    // this chunk's byte span ends where the next chunk's stream begins
    let end = program
        .chunks
        .iter()
        .map(|(_, c)| c.offset)
        .filter(|o| *o > chunk_off)
        .min()
        .unwrap_or(program.bytes.len());
    let span = end - chunk_off;
    let sh = analyze(&ops, chunk.regs as u32);

    // ip2idx blob: u32 dense index per byte of this chunk's span, MAX elsewhere
    let map_data = {
        let mut map = vec![u32::MAX; span.max(1)];
        for (i, (off, _)) in ops.iter().enumerate() {
            map[off - chunk_off] = i as u32;
        }
        let bytes: Vec<u8> = map.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let id = module.declare_anonymous_data(false, false)?;
        let mut dd = DataDescription::new();
        dd.define(bytes.into());
        module.define_data(id, &dd)?;
        id
    };

    module.clear_context(ctx);
    ctx.func.signature = body_sig.clone();

    let hrefs: Vec<FuncRef> = helper_ids
        .iter()
        .map(|id| module.declare_func_in_func(*id, &mut ctx.func))
        .collect();
    let brefs: Vec<FuncRef> = body_ids
        .iter()
        .map(|id| module.declare_func_in_func(*id, &mut ctx.func))
        .collect();
    let bodies_gv = module.declare_data_in_func(bodies_data, &mut ctx.func);
    let map_gv = module.declare_data_in_func(map_data, &mut ctx.func);

    let mut fb = FunctionBuilder::new(&mut ctx.func, fbc);

    // ---- vars ----
    let v_regs = fb.declare_var(I64);
    let v_cur_ip = fb.declare_var(I64);
    let v_ekind = fb.declare_var(I8);
    let mut int_vars = HashMap::new();
    let mut float_vars = HashMap::new();
    for &r in &sh.int {
        int_vars.insert(r, (fb.declare_var(I64), fb.declare_var(I8)));
    }
    for &r in &sh.float {
        float_vars.insert(r, (fb.declare_var(F64), fb.declare_var(I8)));
    }

    // ---- blocks ----
    let entry = fb.create_block();
    let dispatch = fb.create_block();
    let edef = fb.create_block();
    let enext = fb.create_block();
    let eoof = fb.create_block();
    let estep = fb.create_block();
    let eerr = fb.create_block();
    let eret = fb.create_block();
    let eend = fb.create_block();
    fb.set_cold_block(edef);
    fb.set_cold_block(estep);
    fb.set_cold_block(eerr);
    fb.set_cold_block(eoof);
    let blocks: Vec<Block> = ops.iter().map(|_| fb.create_block()).collect();

    let off2idx: HashMap<usize, usize> = ops
        .iter()
        .enumerate()
        .map(|(i, (o, _))| (*o, i))
        .collect();

    // ---- entry ----
    fb.append_block_params_for_function_params(entry);
    fb.switch_to_block(entry);
    let p: Vec<Value> = fb.block_params(entry).to_vec();
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
        paused_p: Value::from_u32(0),
        opsleft_p: Value::from_u32(0),
        ip_p: Value::from_u32(0),
        base: Value::from_u32(0),
        nregs: Value::from_u32(0),
        bodies_tbl: Value::from_u32(0),
    };
    let mut em = Em {
        fb,
        hrefs,
        brefs,
        env,
        v: Vs {
            regs: v_regs,
            cur_ip: v_cur_ip,
            ekind: v_ekind,
            int: int_vars,
            float: float_vars,
        },
        ex: Ex {
            dispatch,
            edef,
            enext,
            eoof,
            estep,
            eerr,
            eret,
            eend,
        },
        sh: &sh,
        ops: &ops,
        blocks,
        off2idx,
        slot_i: StackSlot::from_u32(0),
        slot_f: StackSlot::from_u32(0),
        slot_b: StackSlot::from_u32(0),
        slot_c: StackSlot::from_u32(0),
    };
    let mk_slot = |fb: &mut FunctionBuilder, size: u32| {
        fb.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size, 3))
    };
    em.slot_i = mk_slot(&mut em.fb, 8);
    em.slot_f = mk_slot(&mut em.fb, 8);
    em.slot_b = mk_slot(&mut em.fb, 8);
    em.slot_c = mk_slot(&mut em.fb, 24);

    // hoist environment pointers
    let paused_p = em
        .hcall(H::PausedPtr, &[em.env.ctx0, em.env.ctx1])
        .unwrap();
    let opsleft_p = em.hcall(H::OpsLeftPtr, &[em.env.thread]).unwrap();
    let ip_p = em.hcall(H::IpPtr, &[em.env.code]).unwrap();
    let base = em.hcall(H::FrameBase, &[em.env.thread]).unwrap();
    let nregs = em
        .hcall(H::FrameNregs, &[em.env.thread, em.env.chunks])
        .unwrap();
    let rp = em.hcall(H::RegsPtr, &[em.env.thread]).unwrap();
    let boff = em.fb.ins().imul_imm_s(base, val_size());
    let regs0 = em.fb.ins().iadd(rp, boff);
    let tbl = em.fb.ins().symbol_value(I64, bodies_gv);
    let map_addr = em.fb.ins().symbol_value(I64, map_gv);
    em.env.paused_p = paused_p;
    em.env.opsleft_p = opsleft_p;
    em.env.ip_p = ip_p;
    em.env.base = base;
    em.env.nregs = nregs;
    em.env.bodies_tbl = tbl;
    em.fb.def_var(v_regs, regs0);
    let z64 = em.iconst(0);
    em.fb.def_var(v_cur_ip, z64);
    let z8 = em.iconst8(0);
    em.fb.def_var(v_ekind, z8);

    // shadow init: `(v, ok) = match regs[r] { Int(v) => (v, true), _ => (0,false) }`
    let int_keys: Vec<u32> = {
        let mut k: Vec<u32> = em.v.int.keys().copied().collect();
        k.sort_unstable();
        k
    };
    for r in int_keys {
        let (sv, ok) = em.v.int[&r];
        let idx = em.iconst(r as i64);
        let addr = em.fb.ins().stack_addr(I64, em.slot_i, 0);
        let k = em.hcall(H::Ri, &[regs0, idx, addr]).unwrap();
        let v = em.fb.ins().stack_load(I64, I64, em.slot_i, 0);
        let sv0 = em.fb.ins().select(k, v, z64);
        em.fb.def_var(sv, sv0);
        em.fb.def_var(ok, k);
    }
    let float_keys: Vec<u32> = {
        let mut k: Vec<u32> = em.v.float.keys().copied().collect();
        k.sort_unstable();
        k
    };
    for r in float_keys {
        let (sv, ok) = em.v.float[&r];
        let idx = em.iconst(r as i64);
        let addr = em.fb.ins().stack_addr(I64, em.slot_f, 0);
        let k = em.hcall(H::Rf, &[regs0, idx, addr]).unwrap();
        let v = em.fb.ins().stack_load(I64, F64, em.slot_f, 0);
        let zf = em.fb.ins().f64const(0.0);
        let sv0 = em.fb.ins().select(k, v, zf);
        em.fb.def_var(sv, sv0);
        em.fb.def_var(ok, k);
    }
    em.fb.ins().jump(dispatch, &[]);

    // ---- dispatch: code.ip → dense index → op block ----
    em.fb.switch_to_block(dispatch);
    if ops.is_empty() {
        em.fb.ins().jump(edef, &[]);
        em.fb.switch_to_block(edef);
        em.fb.def_var(v_cur_ip, z64);
        em.fb.ins().jump(estep, &[]);
    } else {
        let ip = em.fb.ins().load(I64, tf(), ip_p, 0);
        let rel = em.fb.ins().iadd_imm_s(ip, -(chunk_off as i64));
        let spanc = em.iconst(span as i64);
        let inb = em.fb.ins().icmp(IntCC::UnsignedLessThan, rel, spanc);
        let l_ok = em.fb.create_block();
        em.fb.ins().brif(inb, l_ok, &[], edef, &[]);
        em.fb.switch_to_block(l_ok);
        let sh2 = em.fb.ins().ishl_imm_s(rel, 2);
        let adr = em.fb.ins().iadd(map_addr, sh2);
        let idx = em.fb.ins().load(I32, tf().with_readonly(), adr, 0);
        let neg1 = em.iconst32(-1);
        let bad = em.fb.ins().icmp(IntCC::Equal, idx, neg1);
        let tbl_blk = em.fb.create_block();
        em.fb.ins().brif(bad, edef, &[], tbl_blk, &[]);
        em.fb.switch_to_block(tbl_blk);
        let def_bc = em.fb.func.dfg.block_call(edef, &[]);
        let calls: Vec<_> = em
            .blocks
            .iter()
            .map(|b| em.fb.func.dfg.block_call(*b, &[]))
            .collect();
        let table = em
            .fb
            .create_jump_table(JumpTableData::new(def_bc, &calls));
        em.fb.ins().br_table(idx, table);
        // `edef` needs the raw ip for the step fallback
        em.fb.switch_to_block(edef);
        em.fb.def_var(v_cur_ip, ip);
        em.fb.ins().jump(estep, &[]);
    }

    // ---- shared exits ----
    em.fb.switch_to_block(enext);
    em.flush_seq();
    let ip = em.fb.use_var(v_cur_ip);
    em.store_ip(ip);
    em.hcall(H::OutNext, &[em.env.out]);
    em.fb.ins().return_(&[]);

    em.fb.switch_to_block(eoof);
    em.flush_seq();
    let ip = em.fb.use_var(v_cur_ip);
    em.store_opip(ip);
    let k = em.iconst8(ERR_OOF);
    em.hcall(H::OutErr, &[em.env.out, k]);
    em.fb.ins().return_(&[]);

    em.fb.switch_to_block(estep);
    em.flush_seq();
    let ip = em.fb.use_var(v_cur_ip);
    em.store_ip(ip);
    em.store_opip(ip);
    let regs = em.regs();
    let k = em
        .hcall(
            H::StepAt,
            &[
                em.env.thread,
                regs,
                em.env.nregs,
                em.env.code,
                em.env.ctx0,
                em.env.ctx1,
                em.env.strs,
                em.env.out,
            ],
        )
        .unwrap();
    em.fb.ins().brif(k, eret, &[], dispatch, &[]);

    em.fb.switch_to_block(eerr);
    em.flush_seq();
    let ip = em.fb.use_var(v_cur_ip);
    em.store_opip(ip);
    let kk = em.fb.use_var(v_ekind);
    em.hcall(H::OutErr, &[em.env.out, kk]);
    em.fb.ins().return_(&[]);

    em.fb.switch_to_block(eret);
    em.fb.ins().return_(&[]);

    em.fb.switch_to_block(eend);
    let max = em.iconst(-1); // usize::MAX — garbage decode, same as bcgen's `_ =>`
    em.fb.def_var(v_cur_ip, max);
    em.fb.ins().jump(estep, &[]);

    // ---- op blocks ----
    for (i, (off, op)) in ops.iter().enumerate() {
        let next = ops.get(i + 1).map(|(o, _)| *o).unwrap_or(usize::MAX);
        em.emit_op(i, *off, next, op);
    }

    em.fb.seal_all_blocks();
    let fe_cfg = module.isa().frontend_config();
    em.fb.finalize(fe_cfg);
    module.define_function(body_ids[body], ctx).map_err(|e| {
        // surface verifier details — `ModuleError`'s Display hides them
        if let cranelift_module::ModuleError::Compilation(
            cranelift_codegen::CodegenError::Verifier(errs),
        ) = &e
        {
            eprintln!("verifier errors in body {body}:\n{errs}\n{}", ctx.func.display());
        }
        e
    })
}

impl Em<'_> {
    /// One op block: the driver's per-op bookkeeping preamble, then semantics.
    fn emit_op(&mut self, i: usize, off: usize, next: usize, op: &Op) {
        self.fb.switch_to_block(self.blocks[i]);
        let o = self.iconst(off as i64);
        self.fb.def_var(self.v.cur_ip, o);
        // `if paused || *fuel == 0 { flush; return Flow::Next }` — same order
        // as run_dispatch / bcgen's loop head.
        let c1 = self.fb.create_block();
        let p = self.fb.ins().load(I8, tf(), self.env.paused_p, 0);
        self.fb.ins().brif(p, self.ex.enext, &[], c1, &[]);
        self.fb.switch_to_block(c1);
        let f = self.fb.ins().load(I64, tf(), self.env.fuel_p, 0);
        let z0 = self.iconst(0);
        let isz = self.fb.ins().icmp(IntCC::Equal, f, z0);
        let c2 = self.fb.create_block();
        self.fb.ins().brif(isz, self.ex.enext, &[], c2, &[]);
        self.fb.switch_to_block(c2);
        let f1 = self.fb.ins().iadd_imm_s(f, -1);
        self.fb.ins().store(tf(), f1, self.env.fuel_p, 0);
        let ol = self.fb.ins().load(I64, tf(), self.env.opsleft_p, 0);
        let isz = self.fb.ins().icmp(IntCC::Equal, ol, z0);
        let c3 = self.fb.create_block();
        self.fb.ins().brif(isz, self.ex.eoof, &[], c3, &[]);
        self.fb.switch_to_block(c3);
        let o1 = self.fb.ins().iadd_imm_s(ol, -1);
        self.fb.ins().store(tf(), o1, self.env.opsleft_p, 0);
        self.semantics(i, off, next, op);
    }

    fn semantics(&mut self, i: usize, off: usize, next: usize, op: &Op) {
        match op {
            Op::Move { dst, src } => {
                self.emit_move(*dst, *src);
                { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
            }
            Op::Jump { target } => {
                let t = self.tgt_blk(target);
                self.fb.ins().jump(t, &[]);
            }
            Op::JumpIf {
                cond,
                target,
                is_true,
            } => {
                let (regs, idx) = (self.regs(), self.iconst(cond.index() as i64));
                let t8 = self.iconst8(*is_true as i64);
                let k = self.hcall(H::IsBool, &[regs, idx, t8]).unwrap();
                let hit = self.mask_by_shadow(*cond, k);
                let (tb, fb) = (self.tgt_blk(target), self.next_blk(i));
                self.fb.ins().brif(hit, tb, &[], fb, &[]);
            }
            Op::ForNext {
                idx,
                bound,
                target,
            } => {
                let iv = self.int_opnd(idx.index() as u32);
                let bv = self.int_opnd(bound.index() as u32);
                let i2 = self.fb.ins().iadd_imm_s(iv, 1);
                self.wr_int_dst(idx.index() as u32, i2);
                let hit = self.fb.ins().icmp(IntCC::SignedLessThan, i2, bv);
                let (tb, fb) = (self.tgt_blk(target), self.next_blk(i));
                self.fb.ins().brif(hit, tb, &[], fb, &[]);
            }
            Op::Switch { .. } | Op::Format { .. } => self.estep(),
            Op::Return { val } => {
                let vp = self.val_ptr(val.index() as u32);
                let do_ret = self.fb.create_block();
                self.root_flush(do_ret);
                self.fb.switch_to_block(do_ret);
                self.hcall(H::OutReturn, &[self.env.out, vp]);
                self.fb.ins().return_(&[]);
            }
            Op::Panic {} => {
                let k = self.iconst8(ERR_PANIC);
                self.fb.def_var(self.v.ekind, k);
                self.fb.ins().jump(self.ex.eerr, &[]);
            }
            Op::Raise { val } => {
                self.flush_seq();
                self.mark_op(off, next);
                let (regs, s) = (self.regs(), self.iconst(val.index() as i64));
                self.hcall(H::Raise, &[regs, s, self.env.out]);
                self.root_flush(self.ex.eret);
            }
            Op::LoadConst { dst, constant } => match constant {
                Constant::Int(v) => {
                    let v = self.iconst(*v);
                    self.wr_int_dst(dst.index() as u32, v);
                    { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
                }
                Constant::Float(v) => {
                    let v = self.fb.ins().f64const(*v);
                    self.wr_float_dst(dst.index() as u32, v);
                    { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
                }
                Constant::Bool(b) => {
                    let v = self.iconst8(*b as i64);
                    let (regs, idx) = (self.regs(), self.iconst(dst.index() as i64));
                    self.hcall(H::WrB, &[regs, idx, v]);
                    { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
                }
                Constant::Null => {
                    let (regs, idx) = (self.regs(), self.iconst(dst.index() as i64));
                    self.hcall(H::WrNull, &[regs, idx]);
                    { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
                }
                Constant::Str(id) => {
                    let (regs, d, id) = (
                        self.regs(),
                        self.iconst(dst.index() as i64),
                        self.iconst32(id.index() as i64),
                    );
                    self.helper_op_v(
                        i,
                        off,
                        next,
                        H::LoadConstStr,
                        &[regs, d, id, self.env.ctx0, self.env.ctx1, self.env.strs],
                    );
                }
                Constant::Array(_) => self.estep(),
            },
            Op::LoadBody { dst, body } => {
                let (regs, d, b) = (
                    self.regs(),
                    self.iconst(dst.index() as i64),
                    self.iconst32(body.index() as i64),
                );
                self.hcall(H::WrFn, &[regs, d, b]);
                { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
            }
            Op::LoadEntry { dst, slot } => {
                // entry-frame absolute slot: tb = regs - base*VS; v = tb[slot]
                let (regs, base) = (self.regs(), self.env.base);
                let boff = self.fb.ins().imul_imm_s(base, val_size());
                let tb = self.fb.ins().isub(regs, boff);
                let vp = self
                    .fb
                    .ins()
                    .iadd_imm_s(tb, slot.index() as i64 * val_size());
                let d = self.iconst(dst.index() as i64);
                self.hcall(H::WrV, &[regs, d, vp]);
                { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
            }
            Op::StoreEntry { slot, src } => {
                let vp = self.val_ptr(src.index() as u32);
                let (regs, base) = (self.regs(), self.env.base);
                let boff = self.fb.ins().imul_imm_s(base, val_size());
                let tb = self.fb.ins().isub(regs, boff);
                let s = self.iconst(slot.index() as i64);
                self.hcall(H::WrV, &[tb, s, vp]);
                { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
            }
            Op::BoolEq { dst, left, right } => self.emit_bool(i, *dst, *left, *right, true),
            Op::BoolNe { dst, left, right } => self.emit_bool(i, *dst, *left, *right, false),
            Op::AddInt { dst, left, right } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.int_opnd(right.index() as u32);
                self.emit_checked(i, *dst, a, b, false);
            }
            Op::SubInt { dst, left, right } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.int_opnd(right.index() as u32);
                self.emit_checked(i, *dst, a, b, true);
            }
            Op::MultInt { dst, left, right } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.int_opnd(right.index() as u32);
                let (v, of) = self.fb.ins().smul_overflow(a, b);
                let okb = self.fb.create_block();
                let t = self.err_tramp();
                self.fb.ins().brif(of, t, &[], okb, &[]);
                self.fb.switch_to_block(okb);
                self.wr_int_dst(dst.index() as u32, v);
                { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
                self.fill_err(t, ERR_OVFW);
            }
            Op::ModInt { dst, left, right } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.int_opnd(right.index() as u32);
                self.emit_mod_int(i, *dst, a, b);
            }
            Op::IntLt { dst, left, right } => {
                self.emit_eval_i(i, *dst, *left, *right, IntCC::SignedLessThan)
            }
            Op::IntLe { dst, left, right } => {
                self.emit_eval_i(i, *dst, *left, *right, IntCC::SignedLessThanOrEqual)
            }
            Op::IntGt { dst, left, right } => {
                self.emit_eval_i(i, *dst, *left, *right, IntCC::SignedGreaterThan)
            }
            Op::IntGe { dst, left, right } => {
                self.emit_eval_i(i, *dst, *left, *right, IntCC::SignedGreaterThanOrEqual)
            }
            Op::IntEq { dst, left, right } => {
                self.emit_eval_i(i, *dst, *left, *right, IntCC::Equal)
            }
            Op::IntNe { dst, left, right } => {
                self.emit_eval_i(i, *dst, *left, *right, IntCC::NotEqual)
            }
            Op::AddIntImm { dst, left, val } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.iconst(*val);
                self.emit_checked(i, *dst, a, b, false);
            }
            Op::SubIntImm { dst, left, val } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.iconst(*val);
                self.emit_checked(i, *dst, a, b, true);
            }
            Op::MultIntImm { dst, left, val } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.iconst(*val);
                let (v, of) = self.fb.ins().smul_overflow(a, b);
                let okb = self.fb.create_block();
                let t = self.err_tramp();
                self.fb.ins().brif(of, t, &[], okb, &[]);
                self.fb.switch_to_block(okb);
                self.wr_int_dst(dst.index() as u32, v);
                { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
                self.fill_err(t, ERR_OVFW);
            }
            Op::ModIntImm { dst, left, val } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.iconst(*val);
                self.emit_mod_int(i, *dst, a, b);
            }
            Op::IntLtImm { dst, left, val } => {
                self.emit_eval_imm_i(i, *dst, *left, *val, IntCC::SignedLessThan)
            }
            Op::IntLeImm { dst, left, val } => {
                self.emit_eval_imm_i(i, *dst, *left, *val, IntCC::SignedLessThanOrEqual)
            }
            Op::IntGtImm { dst, left, val } => {
                self.emit_eval_imm_i(i, *dst, *left, *val, IntCC::SignedGreaterThan)
            }
            Op::IntGeImm { dst, left, val } => {
                self.emit_eval_imm_i(i, *dst, *left, *val, IntCC::SignedGreaterThanOrEqual)
            }
            Op::IntEqImm { dst, left, val } => {
                self.emit_eval_imm_i(i, *dst, *left, *val, IntCC::Equal)
            }
            Op::IntNeImm { dst, left, val } => {
                self.emit_eval_imm_i(i, *dst, *left, *val, IntCC::NotEqual)
            }
            Op::AddFloat { dst, left, right } => {
                self.emit_farit(i, *dst, *left, *right, FOp::Add)
            }
            Op::SubFloat { dst, left, right } => {
                self.emit_farit(i, *dst, *left, *right, FOp::Sub)
            }
            Op::MultFloat { dst, left, right } => {
                self.emit_farit(i, *dst, *left, *right, FOp::Mul)
            }
            Op::DivFloat { dst, left, right } => {
                self.emit_farit(i, *dst, *left, *right, FOp::Div)
            }
            Op::FloatLt { dst, left, right } => {
                self.emit_eval_f(i, *dst, *left, *right, FloatCC::LessThan)
            }
            Op::FloatLe { dst, left, right } => {
                self.emit_eval_f(i, *dst, *left, *right, FloatCC::LessThanOrEqual)
            }
            Op::FloatGt { dst, left, right } => {
                self.emit_eval_f(i, *dst, *left, *right, FloatCC::GreaterThan)
            }
            Op::FloatGe { dst, left, right } => {
                self.emit_eval_f(i, *dst, *left, *right, FloatCC::GreaterThanOrEqual)
            }
            Op::FloatEq { dst, left, right } => {
                self.emit_eval_f(i, *dst, *left, *right, FloatCC::Equal)
            }
            Op::FloatNe { dst, left, right } => {
                self.emit_eval_f(i, *dst, *left, *right, FloatCC::NotEqual)
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
                self.emit_eval_fimm(i, *dst, *left, *val, FloatCC::LessThan)
            }
            Op::FloatLeImm { dst, left, val } => {
                self.emit_eval_fimm(i, *dst, *left, *val, FloatCC::LessThanOrEqual)
            }
            Op::FloatGtImm { dst, left, val } => {
                self.emit_eval_fimm(i, *dst, *left, *val, FloatCC::GreaterThan)
            }
            Op::FloatGeImm { dst, left, val } => {
                self.emit_eval_fimm(i, *dst, *left, *val, FloatCC::GreaterThanOrEqual)
            }
            Op::FloatEqImm { dst, left, val } => {
                self.emit_eval_fimm(i, *dst, *left, *val, FloatCC::Equal)
            }
            Op::FloatNeImm { dst, left, val } => {
                self.emit_eval_fimm(i, *dst, *left, *val, FloatCC::NotEqual)
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
            } => self.emit_brr(i, target, *left, Some(*right), None, *is_true, IntCC::SignedLessThan),
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
                IntCC::SignedLessThanOrEqual,
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
                IntCC::SignedGreaterThan,
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
                IntCC::SignedGreaterThanOrEqual,
            ),
            Op::BIntEq {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(i, target, *left, Some(*right), None, *is_true, IntCC::Equal),
            Op::BIntNe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(i, target, *left, Some(*right), None, *is_true, IntCC::NotEqual),
            Op::BIntLtImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(
                i,
                target,
                *left,
                None,
                Some(*val),
                *is_true,
                IntCC::SignedLessThan,
            ),
            Op::BIntLeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(
                i,
                target,
                *left,
                None,
                Some(*val),
                *is_true,
                IntCC::SignedLessThanOrEqual,
            ),
            Op::BIntGtImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(
                i,
                target,
                *left,
                None,
                Some(*val),
                *is_true,
                IntCC::SignedGreaterThan,
            ),
            Op::BIntGeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(
                i,
                target,
                *left,
                None,
                Some(*val),
                *is_true,
                IntCC::SignedGreaterThanOrEqual,
            ),
            Op::BIntEqImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, IntCC::Equal),
            Op::BIntNeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, IntCC::NotEqual),
            Op::BFloatLt {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(i, target, *left, Some(*right), None, *is_true, FloatCC::LessThan),
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
                FloatCC::LessThanOrEqual,
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
                FloatCC::GreaterThan,
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
                FloatCC::GreaterThanOrEqual,
            ),
            Op::BFloatEq {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(i, target, *left, Some(*right), None, *is_true, FloatCC::Equal),
            Op::BFloatNe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(i, target, *left, Some(*right), None, *is_true, FloatCC::NotEqual),
            Op::BFloatLtImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, FloatCC::LessThan),
            Op::BFloatLeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(
                i,
                target,
                *left,
                None,
                Some(*val),
                *is_true,
                FloatCC::LessThanOrEqual,
            ),
            Op::BFloatGtImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(
                i,
                target,
                *left,
                None,
                Some(*val),
                *is_true,
                FloatCC::GreaterThan,
            ),
            Op::BFloatGeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(
                i,
                target,
                *left,
                None,
                Some(*val),
                *is_true,
                FloatCC::GreaterThanOrEqual,
            ),
            Op::BFloatEqImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, FloatCC::Equal),
            Op::BFloatNeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, FloatCC::NotEqual),
            Op::ToFloat { dst, src } => {
                let iv = self.int_opnd(src.index() as u32);
                let f = self.fb.ins().fcvt_from_sint(F64, iv);
                self.wr_float_dst(dst.index() as u32, f);
                { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
            }
            Op::Sqrt { dst, src } => {
                let f = self.float_opnd(src.index() as u32);
                let r = self.fb.ins().sqrt(f);
                self.wr_float_dst(dst.index() as u32, r);
                { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
            }
            // ---- helper-backed ops ----
            Op::NewArray { dst } => {
                let (regs, d) = (self.regs(), self.iconst(dst.index() as i64));
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::NewArray,
                    &[regs, d, self.env.ctx0, self.env.ctx1],
                );
            }
            Op::NewDict { dst } => {
                let (regs, d) = (self.regs(), self.iconst(dst.index() as i64));
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::NewDict,
                    &[regs, d, self.env.ctx0, self.env.ctx1],
                );
            }
            Op::NewInstance { dst, adt, fields } => {
                let fp = self.reg_list_slot(fields);
                let (regs, d, a, n) = (
                    self.regs(),
                    self.iconst(dst.index() as i64),
                    self.iconst32(adt.index() as i64),
                    self.iconst(fields.len() as i64),
                );
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::NewInstance,
                    &[regs, d, a, fp, n, self.env.ctx0, self.env.ctx1],
                );
            }
            Op::NewClosure {
                dst,
                body,
                captures,
            } => {
                let cp = self.reg_list_slot(captures);
                let (regs, d, b, n) = (
                    self.regs(),
                    self.iconst(dst.index() as i64),
                    self.iconst32(body.index() as i64),
                    self.iconst(captures.len() as i64),
                );
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::NewClosure,
                    &[regs, d, b, cp, n, self.env.ctx0, self.env.ctx1],
                );
            }
            Op::Push { array, value } => {
                let (regs, a, v) = (
                    self.regs(),
                    self.iconst(array.index() as i64),
                    self.iconst(value.index() as i64),
                );
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::Push,
                    &[regs, a, v, self.env.ctx0, self.env.ctx1],
                );
            }
            Op::Insert { dict, key, value } => {
                let (regs, d, k, v) = (
                    self.regs(),
                    self.iconst(dict.index() as i64),
                    self.iconst32(key.index() as i64),
                    self.iconst(value.index() as i64),
                );
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::Insert,
                    &[regs, d, k, v, self.env.ctx0, self.env.ctx1, self.env.strs],
                );
            }
            Op::SetIndex { set, index, value } => {
                let (regs, s, ii, v) = (
                    self.regs(),
                    self.iconst(set.index() as i64),
                    self.iconst(index.index() as i64),
                    self.iconst(value.index() as i64),
                );
                self.helper_op(
                    i,
                    off,
                    next,
                    H::SetIndex,
                    &[regs, s, ii, v, self.env.ctx0, self.env.ctx1, self.env.out],
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
                    self.iconst(dst.index() as i64),
                    self.iconst(set.index() as i64),
                    self.iconst(index.index() as i64),
                    self.iconst8(*kind as i64),
                );
                self.helper_op(
                    i,
                    off,
                    next,
                    H::GetIndex,
                    &[regs, d, s, ii, kk, self.env.ctx0, self.env.ctx1, self.env.out],
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
                    self.iconst(dst.index() as i64),
                    self.iconst(src.index() as i64),
                    self.iconst(*slot as i64),
                    self.iconst8(*kind as i64),
                );
                self.helper_op(i, off, next, H::GetField, &[regs, d, s, sl, kk, self.env.out]);
            }
            Op::SetField {
                receiver,
                slot,
                value,
            } => {
                let (regs, r, sl, v) = (
                    self.regs(),
                    self.iconst(receiver.index() as i64),
                    self.iconst(*slot as i64),
                    self.iconst(value.index() as i64),
                );
                self.helper_op(
                    i,
                    off,
                    next,
                    H::SetField,
                    &[regs, r, sl, v, self.env.ctx0, self.env.ctx1, self.env.out],
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
                    self.iconst(dst.index() as i64),
                    self.iconst(needle.index() as i64),
                    self.iconst(haystack.index() as i64),
                    self.iconst8(*condition as i64),
                );
                self.helper_op_v(i, off, next, H::ContainsOp, &[regs, d, n, h, c]);
            }
            Op::IsInstance { dst, src, adt } => {
                let (regs, d, s, a) = (
                    self.regs(),
                    self.iconst(dst.index() as i64),
                    self.iconst(src.index() as i64),
                    self.iconst32(adt.index() as i64),
                );
                self.helper_op_v(i, off, next, H::IsInstance, &[regs, d, s, a]);
            }
            Op::IsRaised { dst, src } => {
                let (regs, d, s) = (
                    self.regs(),
                    self.iconst(dst.index() as i64),
                    self.iconst(src.index() as i64),
                );
                self.helper_op_v(i, off, next, H::IsRaised, &[regs, d, s]);
            }
            Op::UnwrapRaised { dst, src } => {
                let (regs, d, s) = (
                    self.regs(),
                    self.iconst(dst.index() as i64),
                    self.iconst(src.index() as i64),
                );
                self.helper_op_v(i, off, next, H::UnwrapRaised, &[regs, d, s]);
            }
            Op::Unwrap { dst, src } => {
                let (regs, d, s) = (
                    self.regs(),
                    self.iconst(dst.index() as i64),
                    self.iconst(src.index() as i64),
                );
                self.helper_op(i, off, next, H::Unwrap, &[regs, d, s, self.env.out]);
            }
            Op::UnwrapUnit { dst, src } => {
                let (regs, d, s) = (
                    self.regs(),
                    self.iconst(dst.index() as i64),
                    self.iconst(src.index() as i64),
                );
                self.helper_op(i, off, next, H::UnwrapUnit, &[regs, d, s, self.env.out]);
            }
            Op::Len { dst, src } => {
                self.flush_seq();
                self.mark_op(off, next);
                let (regs, s, v) = (
                    self.regs(),
                    self.iconst(src.index() as i64),
                    self.fb.ins().stack_addr(I64, self.slot_i, 0),
                );
                let k = self.hcall(H::Len, &[regs, s, v, self.env.out]).unwrap();
                let w = self.fb.create_block();
                self.fb.ins().brif(k, self.ex.eret, &[], w, &[]);
                self.fb.switch_to_block(w);
                let v = self.fb.ins().stack_load(I64, I64, self.slot_i, 0);
                self.wr_int_dst(dst.index() as u32, v);
                { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
            }
            Op::Bin {
                dst,
                left,
                op,
                right,
            } => {
                let (regs, d, l, o, r) = (
                    self.regs(),
                    self.iconst(dst.index() as i64),
                    self.iconst(left.index() as i64),
                    self.iconst8(*op as i64),
                    self.iconst(right.index() as i64),
                );
                self.helper_op(
                    i,
                    off,
                    next,
                    H::Bin,
                    &[regs, d, l, o, r, self.env.ctx0, self.env.ctx1, self.env.out],
                );
            }
            Op::Unary { dst, op, src } => {
                let (regs, d, o, s) = (
                    self.regs(),
                    self.iconst(dst.index() as i64),
                    self.iconst8(*op as i64),
                    self.iconst(src.index() as i64),
                );
                self.helper_op(
                    i,
                    off,
                    next,
                    H::Unary,
                    &[regs, d, o, s, self.env.ctx0, self.env.ctx1, self.env.out],
                );
            }
            Op::CallNative { dst, id, args } => {
                let ap = self.reg_list_slot(args);
                let (regs, d, nid, n) = (
                    self.regs(),
                    self.iconst(dst.index() as i64),
                    self.iconst32(id.index() as i64),
                    self.iconst(args.len() as i64),
                );
                self.helper_op(
                    i,
                    off,
                    next,
                    H::CallNative,
                    &[
                        self.env.thread,
                        regs,
                        d,
                        nid,
                        ap,
                        n,
                        self.env.code,
                        self.env.ctx0,
                        self.env.ctx1,
                        self.env.out,
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
    /// `Int`/`Float` — definitely not `Val::Bool(is_true)` — bcgen's defer
    /// rewrite makes the same read produce the scalar `Val`, so compare-false.
    fn mask_by_shadow(&mut self, r: Reg, k: Value) -> Value {
        let ok = self
            .v
            .int
            .get(&(r.index() as u32))
            .map(|&(_, o)| o)
            .or_else(|| self.v.float.get(&(r.index() as u32)).map(|&(_, o)| o));
        match ok {
            None => k,
            Some(ok) => {
                let okv = self.fb.use_var(ok);
                let z8 = self.iconst8(0);
                let not = self.fb.ins().icmp(IntCC::Equal, okv, z8);
                self.fb.ins().band(k, not)
            }
        }
    }

    /// `let Some(v) = a.checked_add/sub(b)` — on overflow, `RtErr::IntegerOverflow`
    /// with `*op_ip = off` (the `eerr` block does both stores from `cur_ip`).
    fn emit_checked(&mut self, i: usize, dst: Reg, a: Value, b: Value, sub: bool) {
        let (v, of) = if sub {
            self.fb.ins().ssub_overflow(a, b)
        } else {
            self.fb.ins().sadd_overflow(a, b)
        };
        let okb = self.fb.create_block();
        let t = self.err_tramp();
        self.fb.ins().brif(of, t, &[], okb, &[]);
        self.fb.switch_to_block(okb);
        self.wr_int_dst(dst.index() as u32, v);
        { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
        self.fill_err(t, ERR_OVFW);
    }

    fn emit_bool(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, eq: bool) {
        // `let Val::Bool(l) = ...` twice — a miss is the interpreter's
        // `unreachable!`; `estep` reproduces it (panic inside `step`).
        let l = self.bool_opnd(left);
        let r = self.bool_opnd(right);
        let cc = if eq { IntCC::Equal } else { IntCC::NotEqual };
        let c = self.fb.ins().icmp(cc, l, r);
        self.wr_bool_dst(dst.index() as u32, c);
        { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
    }

    fn bool_opnd(&mut self, r: Reg) -> Value {
        let idx = r.index() as u32;
        if let Some(&(_, ok)) = self.v.int.get(&idx).or_else(|| self.v.float.get(&idx)) {
            // a live scalar shadow means the reg is NOT a Bool → estep
            let okv = self.fb.use_var(ok);
            let z8 = self.iconst8(0);
            let notok = self.fb.ins().icmp(IntCC::Equal, okv, z8);
            let nb = self.fb.create_block();
            self.fb.ins().brif(notok, nb, &[], self.ex.estep, &[]);
            self.fb.switch_to_block(nb);
        }
        let good = self.fb.create_block();
        let (regs, i64idx, addr) = (
            self.regs(),
            self.iconst(idx as i64),
            self.fb.ins().stack_addr(I64, self.slot_b, 0),
        );
        let ok = self.hcall(H::Rb, &[regs, i64idx, addr]).unwrap();
        self.fb.ins().brif(ok, good, &[], self.ex.estep, &[]);
        self.fb.switch_to_block(good);
        self.fb.ins().stack_load(I64, I8, self.slot_b, 0)
    }

    fn emit_mod_int(&mut self, i: usize, dst: Reg, a: Value, b: Value) {
        // b == 0 → Err(ModByZero); i64::MIN % -1 panics in the interpreter —
        // route that pair to `estep` so `step` hits the identical panic.
        let z = self.iconst(0);
        let bz = self.fb.ins().icmp(IntCC::Equal, b, z);
        let run = self.fb.create_block();
        let tramp = self.err_tramp();
        self.fb.ins().brif(bz, tramp, &[], run, &[]);
        self.fb.switch_to_block(run);
        let c_min = self.iconst(i64::MIN);
        let c_m1 = self.iconst(-1);
        let amin = self.fb.ins().icmp(IntCC::Equal, a, c_min);
        let bm1 = self.fb.ins().icmp(IntCC::Equal, b, c_m1);
        let bad = self.fb.ins().band(amin, bm1);
        let run2 = self.fb.create_block();
        self.fb.ins().brif(bad, self.ex.estep, &[], run2, &[]);
        self.fb.switch_to_block(run2);
        let v = self.fb.ins().srem(a, b);
        self.wr_int_dst(dst.index() as u32, v);
        { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
        self.fill_err(tramp, ERR_MOD0);
    }

    fn emit_eval_i(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, cc: IntCC) {
        let a = self.int_opnd(left.index() as u32);
        let b = self.int_opnd(right.index() as u32);
        let c = self.fb.ins().icmp(cc, a, b);
        self.wr_bool_dst(dst.index() as u32, c);
        { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
    }

    fn emit_eval_imm_i(&mut self, i: usize, dst: Reg, left: Reg, val: i64, cc: IntCC) {
        let a = self.int_opnd(left.index() as u32);
        let b = self.iconst(val);
        let c = self.fb.ins().icmp(cc, a, b);
        self.wr_bool_dst(dst.index() as u32, c);
        { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
    }

    fn emit_eval_f(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, cc: FloatCC) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.float_opnd(right.index() as u32);
        let c = self.fb.ins().fcmp(cc, a, b);
        self.wr_bool_dst(dst.index() as u32, c);
        { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
    }

    fn emit_eval_fimm(&mut self, i: usize, dst: Reg, left: Reg, val: i64, cc: FloatCC) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.fb.ins().f64const(f64::from_bits(val as u64));
        let c = self.fb.ins().fcmp(cc, a, b);
        self.wr_bool_dst(dst.index() as u32, c);
        { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
    }

    fn emit_farit(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, o: FOp) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.float_opnd(right.index() as u32);
        let v = o.emit(&mut self.fb, a, b);
        self.wr_float_dst(dst.index() as u32, v);
        { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
    }

    fn emit_farit_imm(&mut self, i: usize, dst: Reg, left: Reg, val: i64, o: FOp) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.fb.ins().f64const(f64::from_bits(val as u64));
        let v = o.emit(&mut self.fb, a, b);
        self.wr_float_dst(dst.index() as u32, v);
        { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
    }

    /// `StrEq`/`StrNe`: the (Str, Str) fast path via `bin_str`; anything else
    /// runs the op through `step` (which reaches `bin_cold` for the general
    /// pair — the interpreter's own behavior).
    fn emit_str_eval(&mut self, i: usize, off: usize, next: usize, dst: Reg, left: Reg, right: Reg, eq: bool) {
        self.flush_seq();
        self.mark_op(off, next);
        let (regs, d, l, r, e) = (
            self.regs(),
            self.iconst(dst.index() as i64),
            self.iconst(left.index() as i64),
            self.iconst(right.index() as i64),
            self.iconst8(eq as i64),
        );
        // 0 → wrote regs[dst]; 1 → both-Str fast path missed → `step`
        let k = self.hcall(H::BinStr, &[regs, d, l, r, e]).unwrap();
        let nb = self.next_blk(i);
        self.fb.ins().brif(k, self.ex.estep, &[], nb, &[]);
    }

    /// `B*` register/imm branch ops: `hit = a cc b`, then
    /// `code.ip = if hit == is_true { t } else { next }`.
    fn emit_brr(
        &mut self,
        i: usize,
        target: &BlockTarget,
        left: Reg,
        right: Option<Reg>,
        imm: Option<i64>,
        is_true: bool,
        cc: IntCC,
    ) {
        let a = self.int_opnd(left.index() as u32);
        let b = match right {
            Some(r) => self.int_opnd(r.index() as u32),
            None => self.iconst(imm.unwrap()),
        };
        let hit = self.fb.ins().icmp(cc, a, b);
        let (tb, nb) = (self.tgt_blk(target), self.next_blk(i));
        let (t, f) = if is_true { (tb, nb) } else { (nb, tb) };
        self.fb.ins().brif(hit, t, &[], f, &[]);
    }

    fn emit_brf(
        &mut self,
        i: usize,
        target: &BlockTarget,
        left: Reg,
        right: Option<Reg>,
        imm: Option<i64>,
        is_true: bool,
        cc: FloatCC,
    ) {
        let a = self.float_opnd(left.index() as u32);
        let b = match right {
            Some(r) => self.float_opnd(r.index() as u32),
            None => self.fb.ins().f64const(f64::from_bits(imm.unwrap() as u64)),
        };
        let hit = self.fb.ins().fcmp(cc, a, b);
        let (tb, nb) = (self.tgt_blk(target), self.next_blk(i));
        let (t, f) = if is_true { (tb, nb) } else { (nb, tb) };
        self.fb.ins().brif(hit, t, &[], f, &[]);
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
            let v = self.fb.use_var(ssv);
            let k = self.fb.use_var(sok);
            self.fb.def_var(dsv, v);
            self.fb.def_var(dok, k);
            let c = self.fb.create_block();
            let w = self.fb.create_block();
            let dokv = self.fb.use_var(dok);
            self.fb.ins().brif(dokv, c, &[], w, &[]);
            self.fb.switch_to_block(w);
            let regs = self.regs();
            let sidx = self.iconst(s as i64);
            let vp = self.hcall(H::Rval, &[regs, sidx]).unwrap();
            let dd = self.iconst(d as i64);
            self.hcall(H::WrV, &[regs, dd, vp]);
            self.fb.ins().jump(c, &[]);
            self.fb.switch_to_block(c);
        } else if df.is_some() && sf.is_some() {
            let (dsv, dok) = df.unwrap();
            let (ssv, sok) = sf.unwrap();
            let v = self.fb.use_var(ssv);
            let k = self.fb.use_var(sok);
            self.fb.def_var(dsv, v);
            self.fb.def_var(dok, k);
            let c = self.fb.create_block();
            let w = self.fb.create_block();
            let dokv = self.fb.use_var(dok);
            self.fb.ins().brif(dokv, c, &[], w, &[]);
            self.fb.switch_to_block(w);
            let regs = self.regs();
            let sidx = self.iconst(s as i64);
            let vp = self.hcall(H::Rval, &[regs, sidx]).unwrap();
            let dd = self.iconst(d as i64);
            self.hcall(H::WrV, &[regs, dd, vp]);
            self.fb.ins().jump(c, &[]);
            self.fb.switch_to_block(c);
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
            let dd = self.iconst(d as i64);
            self.hcall(H::WrV, &[regs, dd, vp]);
        }
    }

    /// dst scalar-shadowed / src not same-kind-shadowed: probe the (possibly
    /// opposite-kind-shadowed) src for the right tag; hit → shadow write,
    /// miss → `ok = 0; wr(d, regs[s])`.
    fn emit_move_wrv(
        &mut self,
        d: u32,
        s: u32,
        dv: (Variable, Variable),
        osrc: Option<(Variable, Variable)>,
        dst_is_int: bool,
    ) {
        let (dsv, dok) = dv;
        let miss = self.fb.create_block();
        let good = self.fb.create_block();
        let done = self.fb.create_block();
        if let Some((ssv, sok)) = osrc {
            // src carries a live opposite-kind shadow → the Val is that scalar
            // → miss arm writes it through directly.
            let okv = self.fb.use_var(sok);
            let genb = self.fb.create_block();
            let fmiss = self.fb.create_block();
            self.fb.ins().brif(okv, fmiss, &[], genb, &[]);
            self.fb.switch_to_block(fmiss);
            let v = self.fb.use_var(ssv);
            let (regs, dd) = (self.regs(), self.iconst(d as i64));
            if dst_is_int {
                self.hcall(H::WrF, &[regs, dd, v]);
            } else {
                self.hcall(H::WrI, &[regs, dd, v]);
            }
            let z = self.iconst8(0);
            self.fb.def_var(dok, z);
            self.fb.ins().jump(done, &[]);
            self.fb.switch_to_block(genb);
        }
        // generic: probe the authoritative slot for the wanted tag
        let (slot, rh, wh) = if dst_is_int {
            (self.slot_i, H::Ri, H::WrI)
        } else {
            (self.slot_f, H::Rf, H::WrF)
        };
        let _ = wh;
        let (regs, ss, addr) = (
            self.regs(),
            self.iconst(s as i64),
            self.fb.ins().stack_addr(I64, slot, 0),
        );
        let k = self.hcall(rh, &[regs, ss, addr]).unwrap();
        self.fb.ins().brif(k, good, &[], miss, &[]);
        self.fb.switch_to_block(good);
        let v = self.fb.ins().stack_load(
            I64,
            if dst_is_int { I64 } else { F64 },
            slot,
            0,
        );
        self.fb.def_var(dsv, v);
        let one = self.iconst8(1);
        self.fb.def_var(dok, one);
        self.fb.ins().jump(done, &[]);
        self.fb.switch_to_block(miss);
        let regs = self.regs();
        let vp = self.hcall(H::Rval, &[regs, ss]).unwrap();
        let dd = self.iconst(d as i64);
        self.hcall(H::WrV, &[regs, dd, vp]);
        let z = self.iconst8(0);
        self.fb.def_var(dok, z);
        self.fb.ins().jump(done, &[]);
        self.fb.switch_to_block(done);
    }

    /// `CallDirect` — the inline fast path: `enter_call` via shim, then a direct
    /// call to the callee's JIT body; `Flow::Return` pops and resumes in-place.
    /// At/over `INLINE_CALL_DEPTH`, produce `Flow::Call` for the driver.
    fn emit_call_direct(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        dst: Reg,
        body: compile::BodyId,
        args: &[Reg],
    ) {
        let ap = self.reg_list_slot(args);
        let nargs = args.len();
        // `code.ip = next; *op_ip = off` before either path (enter_call saves
        // `code.ip` as the caller resume slot; errors locate via op_ip).
        self.mark_op(off, next);
        let depth = self.hcall(H::FramesLen, &[self.env.thread]).unwrap();
        let cap = self.iconst(INLINE_CALL_DEPTH as i64);
        let inl = self.fb.ins().icmp(IntCC::UnsignedLessThan, depth, cap);
        let capt = self.fb.create_block();
        let inlt = self.fb.create_block();
        self.fb.ins().brif(inl, inlt, &[], capt, &[]);

        // depth cap → `Flow::Call` for the driver
        self.fb.switch_to_block(capt);
        self.flush_seq();
        let (regs, b, d, n) = (
            self.regs(),
            self.iconst32(body.index() as i64),
            self.iconst(dst.index() as i64),
            self.iconst(nargs as i64),
        );
        self.hcall(H::OutCallDirect, &[self.env.out, regs, b, d, ap, n]);
        self.fb.ins().return_(&[]);

        // inline: enter the callee frame, call its JIT body, pop on Return
        self.fb.switch_to_block(inlt);
        self.flush_seq();
        let (regs, b, d, n, zp) = (
            self.regs(),
            self.iconst32(body.index() as i64),
            self.iconst(dst.index() as i64),
            self.iconst(nargs as i64),
            self.iconst(0),
        );
        let k = self
            .hcall(
                H::Enter,
                &[
                    self.env.thread,
                    self.env.code,
                    self.env.chunks,
                    b,
                    d,
                    regs,
                    ap,
                    n,
                    zp,
                    zp,
                    self.env.out,
                ],
            )
            .unwrap();
        let callb = self.fb.create_block();
        self.fb.ins().brif(k, self.ex.eret, &[], callb, &[]);
        self.fb.switch_to_block(callb);
        self.call_body(self.brefs[body.index()]);
        self.emit_pop_resume(i, dst);
    }

    /// `Call` — resolve the callee register through `call_target`, then the
    /// same inline path with an indirect call through the `mj_bodies` table.
    fn emit_call(&mut self, i: usize, off: usize, next: usize, dst: Reg, callee: Reg, args: &[Reg]) {
        // Resolve first — `call_target` reads regs, so flush before it (this
        // covers the error-arms' `*op_ip` too). bcgen resolves the target
        // before storing `code.ip = next`, so the same ordering holds here.
        self.flush_seq();
        let bp = self.fb.ins().stack_addr(I64, self.slot_c, 0);
        let cp = self.fb.ins().stack_addr(I64, self.slot_c, 8);
        let np = self.fb.ins().stack_addr(I64, self.slot_c, 16);
        let (regs, c) = (self.regs(), self.iconst(callee.index() as i64));
        let k = self
            .hcall(
                H::CallTarget,
                &[regs, c, self.env.sigs, bp, cp, np, self.env.out],
            )
            .unwrap();
        let rok = self.fb.create_block();
        let rerr = self.fb.create_block();
        self.fb.ins().brif(k, rerr, &[], rok, &[]);
        self.fb.switch_to_block(rerr);
        let o = self.iconst(off as i64);
        self.store_opip(o);
        self.fb.ins().return_(&[]);

        self.fb.switch_to_block(rok);
        let ap = self.reg_list_slot(args);
        let nargs = args.len();
        self.mark_op(off, next);
        let depth = self.hcall(H::FramesLen, &[self.env.thread]).unwrap();
        let cap = self.iconst(INLINE_CALL_DEPTH as i64);
        let inl = self.fb.ins().icmp(IntCC::UnsignedLessThan, depth, cap);
        let capt = self.fb.create_block();
        let inlt = self.fb.create_block();
        self.fb.ins().brif(inl, inlt, &[], capt, &[]);

        self.fb.switch_to_block(capt);
        let (regs2, c2, d, n) = (
            self.regs(),
            self.iconst(callee.index() as i64),
            self.iconst(dst.index() as i64),
            self.iconst(nargs as i64),
        );
        self.hcall(H::OutCall, &[self.env.out, regs2, c2, d, ap, n]);
        self.fb.ins().return_(&[]);

        self.fb.switch_to_block(inlt);
        let b32 = self.fb.ins().stack_load(I64, I32, self.slot_c, 0);
        let caps = self.fb.ins().stack_load(I64, I64, self.slot_c, 8);
        let ncaps = self.fb.ins().stack_load(I64, I64, self.slot_c, 16);
        let (regs, d, n) = (
            self.regs(),
            self.iconst(dst.index() as i64),
            self.iconst(nargs as i64),
        );
        let k = self
            .hcall(
                H::Enter,
                &[
                    self.env.thread,
                    self.env.code,
                    self.env.chunks,
                    b32,
                    d,
                    regs,
                    ap,
                    n,
                    caps,
                    ncaps,
                    self.env.out,
                ],
            )
            .unwrap();
        let callb = self.fb.create_block();
        self.fb.ins().brif(k, self.ex.eret, &[], callb, &[]);
        self.fb.switch_to_block(callb);
        // `BODIES[body]` — indirect call through the fn-ptr data table
        let b64 = self.fb.ins().uextend(I64, b32);
        let boff = self.fb.ins().ishl_imm_s(b64, 3);
        let adr = self.fb.ins().iadd(self.env.bodies_tbl, boff);
        let fptr = self.fb.ins().load(I64, tf().with_readonly(), adr, 0);
        let sig_ref = self.sigref();
        let e = self.env;
        self.fb.ins().call_indirect(
            sig_ref,
            fptr,
            &[
                e.thread, e.code, e.ctx0, e.ctx1, e.strs, e.chunks, e.sigs, e.fuel_p, e.opip_p,
                e.out,
            ],
        );
        self.emit_pop_resume(i, dst);
    }

    /// Direct call to a callee JIT body — the same extern-C signature the
    /// driver uses.
    fn call_body(&mut self, f: FuncRef) {
        let e = self.env;
        self.fb.ins().call(
            f,
            &[
                e.thread, e.code, e.ctx0, e.ctx1, e.strs, e.chunks, e.sigs, e.fuel_p, e.opip_p,
                e.out,
            ],
        );
    }

    /// After an inlined callee returns: `pop_return` (the driver's Return
    /// handling), rebuild the caller window, refresh dst's shadow, resume at
    /// the next op. Any other `out` propagates verbatim via `eret`.
    fn emit_pop_resume(&mut self, i: usize, dst: Reg) {
        let k = self
            .hcall(H::PopReturn, &[self.env.thread, self.env.code, self.env.out])
            .unwrap();
        let two = self.iconst8(2);
        let popped = self.fb.ins().icmp(IntCC::Equal, k, two);
        let resumed = self.fb.create_block();
        self.fb.ins().brif(popped, resumed, &[], self.ex.eret, &[]);
        self.fb.switch_to_block(resumed);
        // callee's enter_call may have moved `thread.regs` — rebuild the window
        let rp = self.hcall(H::RegsPtr, &[self.env.thread]).unwrap();
        let boff = self.fb.ins().imul_imm_s(self.env.base, val_size());
        let regs2 = self.fb.ins().iadd(rp, boff);
        self.fb.def_var(self.v.regs, regs2);
        let d = dst.index() as u32;
        if let Some(&(sv, ok)) = self.v.int.get(&d) {
            let idx = self.iconst(d as i64);
            let addr = self.fb.ins().stack_addr(I64, self.slot_i, 0);
            let k = self.hcall(H::Ri, &[regs2, idx, addr]).unwrap();
            let v = self.fb.ins().stack_load(I64, I64, self.slot_i, 0);
            let z = self.iconst(0);
            let sv0 = self.fb.ins().select(k, v, z);
            self.fb.def_var(sv, sv0);
            self.fb.def_var(ok, k);
        } else if let Some(&(sv, ok)) = self.v.float.get(&d) {
            let idx = self.iconst(d as i64);
            let addr = self.fb.ins().stack_addr(I64, self.slot_f, 0);
            let k = self.hcall(H::Rf, &[regs2, idx, addr]).unwrap();
            let v = self.fb.ins().stack_load(I64, F64, self.slot_f, 0);
            let z = self.fb.ins().f64const(0.0);
            let sv0 = self.fb.ins().select(k, v, z);
            self.fb.def_var(sv, sv0);
            self.fb.def_var(ok, k);
        }
        { let nb = self.next_blk(i); self.fb.ins().jump(nb, &[]); }
    }

    /// `import_signature` for indirect body calls.
    fn sigref(&mut self) -> ir::SigRef {
        let mut sig = Signature::new(cranelift_codegen::isa::CallConv::SystemV);
        for _ in 0..10 {
            sig.params.push(AbiParam::new(I64));
        }
        self.fb.func.import_signature(sig)
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
    fn emit(self, fb: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        match self {
            FOp::Add => fb.ins().fadd(a, b),
            FOp::Sub => fb.ins().fsub(a, b),
            FOp::Mul => fb.ins().fmul(a, b),
            FOp::Div => fb.ins().fdiv(a, b),
            FOp::Mod => {
                // Rust `%` on f64 → `a - trunc(a/b)*b` (fmod semantics)
                let q = fb.ins().fdiv(a, b);
                let t = fb.ins().trunc(q);
                let p = fb.ins().fmul(t, b);
                fb.ins().fsub(a, p)
            }
        }
    }
}

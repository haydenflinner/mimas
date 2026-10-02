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

use crate::H;
use compile::{AccessKind, BlockTarget, Constant, Op, Program, Reg};
use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    self, Block, FuncRef, InstBuilder, JumpTableData, MemFlagsData, Signature, StackSlot,
    StackSlotData, StackSlotKind, Value, types,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::JITModule;
use cranelift_module::{DataDescription, DataId, FuncId, Module, ModuleResult};
use vm::bc::{INLINE_CALL_DEPTH, jit::Layout};

const I64: ir::Type = types::I64;
const I32: ir::Type = types::I32;
const I8: ir::Type = types::I8;
const F64: ir::Type = types::F64;

fn tf() -> MemFlagsData {
    MemFlagsData::trusted()
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
    /// `GetIndex`/`GetField` writes a dynamically-typed value, but the JIT
    /// probes the loaded element's tag and refreshes the shadow `(sv, ok)`
    /// pair — so the write is compatible with either scalar shadow *if* the
    /// reg is actually read as that scalar (the read-set gate keeps purely
    /// dynamic dsts like `balls[i]`'s Instance out of the shadow sets).
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
            // inline emitters refresh the scalar shadow themselves (`DynCheck`)
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
    /// Batched op quota — `min(*fuel, thread.ops_left)` at the last re-arm;
    /// decremented per op, `settle`d back into the real counters at every
    /// observable boundary. bcgen's `bcn`/`bcn0`, verbatim.
    bcn: Variable,
    bcn0: Variable,
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
    /// `edef`/`eend` route here first — `run_dispatch` checks the quota and
    /// charges one op even for a garbage decode.
    #[allow(dead_code)]
    estep_g: Block,
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
    env: Env,
    v: Vs,
    ex: Ex,
    sh: &'a Sh,
    ops: &'a [(usize, Op)],
    /// Probed VM layouts — `Val` tag/payload offsets, `ThreadState`/`Frame`/
    /// `Decoder` field offsets, `Vec` header order. Emitted code reads and
    /// writes these inline instead of FFI-ing per access.
    lyt: Layout,
    /// The whole program — `CallDirect` needs the callee's `Chunk` (offset,
    /// regs, param mapping) to emit the inline frame push.
    prog: &'a Program,
    /// `FuncRef`s for `CallDirect` targets, so the fast path is a direct
    /// native `call`, not an FFI hop.
    call_refs: HashMap<u32, FuncRef>,
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

    // ---- inline `Val` access over the probed layout (`lyt`) — the emit-time
    // twin of `bc::jit`'s `ri`/`wr_*`/`rval` FFI helpers ----

    /// `thread.frames.len()` — one load through the probed Vec header.
    fn frames_len(&mut self) -> Value {
        self.fb.ins().load(
            I64,
            tf(),
            self.env.thread,
            (self.lyt.frames_off + self.lyt.vec_len) as i32,
        )
    }

    /// `regs + r*val_size` — `&regs[r]` for a window base `regs`.
    fn vaddr(&mut self, regs: Value, r: u32) -> Value {
        self.fb
            .ins()
            .iadd_imm_s(regs, r as i64 * self.lyt.val_size as i64)
    }

    /// The discriminant's CLIF type (its probed byte width).
    fn tag_ty(&self) -> ir::Type {
        match self.lyt.tag_size {
            1 => I8,
            2 => types::I16,
            4 => I32,
            8 => I64,
            d => unreachable!("bad tag width {d}"),
        }
    }

    /// A discriminant constant at the probed width.
    fn tconst(&mut self, t: u64) -> Value {
        let ty = self.tag_ty();
        self.fb.ins().iconst(ty, t as i64)
    }

    /// `regs[r]`'s discriminant.
    fn ld_tag(&mut self, a: Value) -> Value {
        let ty = self.tag_ty();
        self.fb.ins().load(ty, tf(), a, self.lyt.val_tag as i32)
    }

    /// `regs[r]` as `Val::Int`: inline tag probe → payload load; a miss routes
    /// to `estep` exactly like the `ri` helper's `0` return did.
    fn ld_int(&mut self, regs: Value, r: u32) -> Value {
        let good = self.fb.create_block();
        let a = self.vaddr(regs, r);
        let t = self.ld_tag(a);
        let want = self.tconst(self.lyt.t_int);
        let hit = self.fb.ins().icmp(IntCC::Equal, t, want);
        self.fb.ins().brif(hit, good, &[], self.ex.estep, &[]);
        self.fb.switch_to_block(good);
        self.fb.ins().load(I64, tf(), a, self.lyt.val_pay as i32)
    }

    /// `regs[r]` as `Val::Float`.
    fn ld_float(&mut self, regs: Value, r: u32) -> Value {
        let good = self.fb.create_block();
        let a = self.vaddr(regs, r);
        let t = self.ld_tag(a);
        let want = self.tconst(self.lyt.t_float);
        let hit = self.fb.ins().icmp(IntCC::Equal, t, want);
        self.fb.ins().brif(hit, good, &[], self.ex.estep, &[]);
        self.fb.switch_to_block(good);
        self.fb.ins().load(F64, tf(), a, self.lyt.val_pay as i32)
    }

    /// `*a = Val::Int(v)` — tag byte plus the 8-byte union slot.
    fn st_int(&mut self, a: Value, v: Value) {
        let t = self.tconst(self.lyt.t_int);
        self.fb.ins().store(tf(), t, a, self.lyt.val_tag as i32);
        self.fb.ins().store(tf(), v, a, self.lyt.val_pay as i32);
    }

    /// `*a = Val::Float(v)` (ditto, `f64` store).
    fn st_float(&mut self, a: Value, v: Value) {
        let t = self.tconst(self.lyt.t_float);
        self.fb.ins().store(tf(), t, a, self.lyt.val_tag as i32);
        self.fb.ins().store(tf(), v, a, self.lyt.val_pay as i32);
    }

    /// `*a = Val::Bool(v8)` — tag plus the `u8` union slot.
    fn st_bool(&mut self, a: Value, v8: Value) {
        let t = self.tconst(self.lyt.t_bool);
        self.fb.ins().store(tf(), t, a, self.lyt.val_tag as i32);
        self.fb.ins().store(tf(), v8, a, self.lyt.bool_pay as i32);
    }

    /// `*a = Val::Null` — only the tag byte is read for payload-less
    /// variants, so only it is written.
    fn st_null(&mut self, a: Value) {
        let t = self.tconst(self.lyt.t_null);
        self.fb.ins().store(tf(), t, a, self.lyt.val_tag as i32);
    }

    /// `*a = Val::Fn(body)` — tag plus the `u32` index in the union slot.
    fn st_fn(&mut self, a: Value, body32: Value) {
        let t = self.tconst(self.lyt.t_fn);
        self.fb.ins().store(tf(), t, a, self.lyt.val_tag as i32);
        self.fb.ins().store(tf(), body32, a, self.lyt.fn_pay as i32);
    }

    /// `val_size`-byte `Val` copy `*d = *s` (eight-byte chunks; the probe
    /// asserts the stride is a multiple of 8).
    fn cpy_val(&mut self, d: Value, s: Value) {
        for k in 0..(self.lyt.val_size / 8) as i32 {
            let w = self.fb.ins().load(I64, tf(), s, k * 8);
            self.fb.ins().store(tf(), w, d, k * 8);
        }
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
    /// Branch-free: a dead shadow (`ok == 0`) means the window is already
    /// authoritative, so the tag/payload stores rewrite the bytes just
    /// loaded — a `select` per field instead of a branch per reg.
    fn flush_seq(&mut self) {
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
            let okv = self.fb.use_var(ok);
            let v = self.fb.use_var(sv);
            let a = self.vaddr(regs, r);
            let old = self.ld_tag(a);
            let nt = self.fb.ins().select(okv, tint, old);
            self.fb.ins().store(tf(), nt, a, self.lyt.val_tag as i32);
            let oldp = self.fb.ins().load(I64, tf(), a, self.lyt.val_pay as i32);
            let np = self.fb.ins().select(okv, v, oldp);
            self.fb.ins().store(tf(), np, a, self.lyt.val_pay as i32);
        }
        let tflt = self.tconst(self.lyt.t_float);
        for r in floats {
            let (sv, ok) = self.v.float[&r];
            let okv = self.fb.use_var(ok);
            let v = self.fb.use_var(sv);
            let a = self.vaddr(regs, r);
            let old = self.ld_tag(a);
            let nt = self.fb.ins().select(okv, tflt, old);
            self.fb.ins().store(tf(), nt, a, self.lyt.val_tag as i32);
            let oldp = self.fb.ins().load(F64, tf(), a, self.lyt.val_pay as i32);
            let np = self.fb.ins().select(okv, v, oldp);
            self.fb.ins().store(tf(), np, a, self.lyt.val_pay as i32);
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
        if let Some(&(sv, ok)) = self.v.int.get(&r) {
            let good = self.fb.create_block();
            let okv = self.fb.use_var(ok);
            self.fb.ins().brif(okv, good, &[], self.ex.estep, &[]);
            self.fb.switch_to_block(good);
            self.fb.use_var(sv)
        } else {
            let regs = self.regs();
            self.ld_int(regs, r)
        }
    }

    fn float_opnd(&mut self, r: u32) -> Value {
        if let Some(&(sv, ok)) = self.v.float.get(&r) {
            let good = self.fb.create_block();
            let okv = self.fb.use_var(ok);
            self.fb.ins().brif(okv, good, &[], self.ex.estep, &[]);
            self.fb.switch_to_block(good);
            self.fb.use_var(sv)
        } else {
            let regs = self.regs();
            self.ld_float(regs, r)
        }
    }

    /// Scalar write into `dst`: shadow-var update when shadowed, else an
    /// inline `Val` store.
    fn wr_int_dst(&mut self, d: u32, v: Value) {
        if let Some(&(sv, ok)) = self.v.int.get(&d) {
            self.fb.def_var(sv, v);
            let one = self.iconst8(1);
            self.fb.def_var(ok, one);
        } else {
            let regs = self.regs();
            let a = self.vaddr(regs, d);
            self.st_int(a, v);
        }
    }

    fn wr_float_dst(&mut self, d: u32, v: Value) {
        if let Some(&(sv, ok)) = self.v.float.get(&d) {
            self.fb.def_var(sv, v);
            let one = self.iconst8(1);
            self.fb.def_var(ok, one);
        } else {
            let regs = self.regs();
            let a = self.vaddr(regs, d);
            self.st_float(a, v);
        }
    }

    /// Bool write into `dst` — eval dsts are always `W::Dyn` (unshadowed).
    fn wr_bool_dst(&mut self, d: u32, v8: Value) {
        let regs = self.regs();
        let a = self.vaddr(regs, d);
        self.st_bool(a, v8);
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
            let regs = self.regs();
            let a = self.vaddr(regs, s);
            let v = self.fb.use_var(sv);
            self.st_int(a, v);
            self.fb.ins().jump(c, &[]);
            self.fb.switch_to_block(c);
        } else if let Some(&(sv, ok)) = self.v.float.get(&s) {
            let c = self.fb.create_block();
            let m = self.fb.create_block();
            let okv = self.fb.use_var(ok);
            self.fb.ins().brif(okv, m, &[], c, &[]);
            self.fb.switch_to_block(m);
            let regs = self.regs();
            let a = self.vaddr(regs, s);
            let v = self.fb.use_var(sv);
            self.st_float(a, v);
            self.fb.ins().jump(c, &[]);
            self.fb.switch_to_block(c);
        }
        let regs = self.regs();
        self.vaddr(regs, s)
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
        {
            let nb = self.next_blk(i);
            self.fb.ins().jump(nb, &[]);
        }
    }

    /// After a helper wrote `regs[r]` behind the shadow's back (a container
    /// read's slow path), re-derive the `DynCheck` `(sv, ok)` pair from the
    /// authoritative slot — `ok` tracks the loaded tag exactly like the
    /// inline `read_elem` refresh.
    fn refresh_shadow(&mut self, r: u32) {
        if !self.v.int.contains_key(&r) && !self.v.float.contains_key(&r) {
            return;
        }
        let regs = self.regs();
        let a = self.vaddr(regs, r);
        let t = self.ld_tag(a);
        if let Some(&(sv, ok)) = self.v.int.get(&r) {
            let want = self.tconst(self.lyt.t_int);
            let k = self.fb.ins().icmp(IntCC::Equal, t, want);
            let pv = self.fb.ins().load(I64, tf(), a, self.lyt.val_pay as i32);
            self.fb.def_var(sv, pv);
            self.fb.def_var(ok, k);
        }
        if let Some(&(sv, ok)) = self.v.float.get(&r) {
            let want = self.tconst(self.lyt.t_float);
            let k = self.fb.ins().icmp(IntCC::Equal, t, want);
            let pv = self.fb.ins().load(F64, tf(), a, self.lyt.val_pay as i32);
            self.fb.def_var(sv, pv);
            self.fb.def_var(ok, k);
        }
    }

    /// `helper_op` + a post-call [`Em::refresh_shadow`] on `dst` — every
    /// `GetIndex`/`GetField` helper invocation (slow tails AND `Option`-kind
    /// ops) must go through this so a `DynCheck`-shadowed dst stays coherent.
    fn helper_op_read(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        h: H,
        args: &[Value],
        dst: Reg,
    ) {
        self.flush_seq();
        self.mark_op(off, next);
        let k = self.hcall(h, args).unwrap();
        let post = self.fb.create_block();
        self.fb.ins().brif(k, self.ex.eret, &[], post, &[]);
        self.fb.switch_to_block(post);
        self.refresh_shadow(dst.index() as u32);
        let nb = self.next_blk(i);
        self.fb.ins().jump(nb, &[]);
    }

    /// A `u32` register-index list (call args / field regs / captures) spilled
    /// to a stack slot; returns the slot address for the helper.
    fn reg_list_slot(&mut self, regs_idx: &[Reg]) -> Value {
        let size = (regs_idx.len().max(1) * 4) as u32;
        let slot = self.fb.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            size,
            3,
        ));
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
        let flen = self.frames_len();
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

    // ---- batched quota (`bcn`) plumbing — bcgen's settle!/gexit!/gateq! ----

    /// `spent = bcn0 - bcn; bcn0 = bcn; *fuel -= spent; ops_left -= spent`.
    /// Idempotent (a second settle charges 0), so exits can run it
    /// unconditionally. Returns the post-settle `(fuel, ops_left)` values.
    fn settle_seq(&mut self) -> (Value, Value) {
        let b0 = self.fb.use_var(self.v.bcn0);
        let b = self.fb.use_var(self.v.bcn);
        let spent = self.fb.ins().isub(b0, b);
        self.fb.def_var(self.v.bcn0, b);
        let f = self.fb.ins().load(I64, tf(), self.env.fuel_p, 0);
        let f2 = self.fb.ins().isub(f, spent);
        self.fb.ins().store(tf(), f2, self.env.fuel_p, 0);
        let ol = self.fb.ins().load(I64, tf(), self.env.opsleft_p, 0);
        let ol2 = self.fb.ins().isub(ol, spent);
        self.fb.ins().store(tf(), ol2, self.env.opsleft_p, 0);
        (f2, ol2)
    }

    /// `bcn = min(*fuel, ops_left); bcn0 = bcn` — armed at entry and re-armed
    /// wherever quota was spent outside our count (an inlined callee draws
    /// from the same counters through its own `bcn`).
    fn rearm_seq(&mut self) {
        let f = self.fb.ins().load(I64, tf(), self.env.fuel_p, 0);
        let ol = self.fb.ins().load(I64, tf(), self.env.opsleft_p, 0);
        let m = self.fb.ins().umin(f, ol);
        self.fb.def_var(self.v.bcn, m);
        self.fb.def_var(self.v.bcn0, m);
    }

    /// One op's quota gate — bcgen's `gateq!()`/`gatep!()`: `paused` is only
    /// loaded when the static fallthrough predecessor can run foreign code
    /// (`may_pause`); a `bcn == 0` trip (or the pause flag) routes to a
    /// per-op cold trampoline running `gexit` — settle, then the driver's
    /// ordered exit reasons, then re-arm and resume the op. Returns the
    /// continuation block the op body emits into.
    fn gate(&mut self, i: usize, off: usize) -> (Block, Block) {
        let o = self.iconst(off as i64);
        self.fb.def_var(self.v.cur_ip, o);
        let cont = self.fb.create_block();
        let tramp = self.fb.create_block();
        self.fb.set_cold_block(tramp);
        if i > 0 && may_pause(&self.ops[i - 1].1) {
            let p = self.fb.ins().load(I8, tf(), self.env.paused_p, 0);
            let g = self.fb.create_block();
            self.fb.ins().brif(p, tramp, &[], g, &[]);
            self.fb.switch_to_block(g);
        }
        let bcnv = self.fb.use_var(self.v.bcn);
        let z = self.iconst(0);
        let bz = self.fb.ins().icmp(IntCC::Equal, bcnv, z);
        self.fb.ins().brif(bz, tramp, &[], cont, &[]);
        self.fb.switch_to_block(cont);
        // `use_var` again — the trampoline's re-arm path also lands here, so
        // the decrement must read the merged value, not the pre-branch one.
        let cur = self.fb.use_var(self.v.bcn);
        let b1 = self.fb.ins().iadd_imm_s(cur, -1);
        self.fb.def_var(self.v.bcn, b1);
        (cont, tramp)
    }

    /// Fill a gate trampoline (created by [`Em::gate`] or `estep_g`): settle,
    /// then `paused || fuel == 0` → `enext`, `ops_left == 0` → `eoof`, else
    /// re-arm and resume at `cont`. Call when the current block is terminated.
    fn fill_gate_tramp(&mut self, tramp: Block, cont: Block) {
        self.fb.switch_to_block(tramp);
        let (f, ol) = self.settle_seq();
        let p = self.fb.ins().load(I8, tf(), self.env.paused_p, 0);
        let z = self.iconst(0);
        let fz = self.fb.ins().icmp(IntCC::Equal, f, z);
        let nx = self.fb.ins().bor(p, fz);
        let t2 = self.fb.create_block();
        self.fb.ins().brif(nx, self.ex.enext, &[], t2, &[]);
        self.fb.switch_to_block(t2);
        let oz = self.fb.ins().icmp(IntCC::Equal, ol, z);
        let t3 = self.fb.create_block();
        self.fb.ins().brif(oz, self.ex.eoof, &[], t3, &[]);
        self.fb.switch_to_block(t3);
        self.rearm_seq();
        self.fb.ins().jump(cont, &[]);
    }
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
    lyt: &Layout,
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
    let bodies_gv = module.declare_data_in_func(bodies_data, &mut ctx.func);
    let map_gv = module.declare_data_in_func(map_data, &mut ctx.func);
    // `CallDirect`'s inline fast path calls the callee's native body directly —
    // one FuncRef per distinct target.
    let call_refs: HashMap<u32, FuncRef> = ops
        .iter()
        .filter_map(|(_, op)| match op {
            Op::CallDirect { body, .. } => Some(*body),
            _ => None,
        })
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|b| {
            (
                b.index() as u32,
                module.declare_func_in_func(body_ids[b.index()], &mut ctx.func),
            )
        })
        .collect();

    let mut fb = FunctionBuilder::new(&mut ctx.func, fbc);

    // ---- vars ----
    let v_regs = fb.declare_var(I64);
    let v_cur_ip = fb.declare_var(I64);
    let v_ekind = fb.declare_var(I8);
    let v_bcn = fb.declare_var(I64);
    let v_bcn0 = fb.declare_var(I64);
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
    let estep_g = fb.create_block();
    let eerr = fb.create_block();
    let eret = fb.create_block();
    let eend = fb.create_block();
    fb.set_cold_block(edef);
    fb.set_cold_block(estep);
    fb.set_cold_block(estep_g);
    fb.set_cold_block(eerr);
    fb.set_cold_block(eoof);
    let blocks: Vec<Block> = ops.iter().map(|_| fb.create_block()).collect();

    let off2idx: HashMap<usize, usize> =
        ops.iter().enumerate().map(|(i, (o, _))| (*o, i)).collect();

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
        env,
        v: Vs {
            regs: v_regs,
            cur_ip: v_cur_ip,
            ekind: v_ekind,
            bcn: v_bcn,
            bcn0: v_bcn0,
            int: int_vars,
            float: float_vars,
        },
        ex: Ex {
            dispatch,
            edef,
            enext,
            eoof,
            estep,
            estep_g,
            eerr,
            eret,
            eend,
        },
        sh: &sh,
        ops: &ops,
        lyt: *lyt,
        prog: program,
        call_refs,
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

    // hoist environment pointers — `&state.paused` is one add off `Ctx`'s
    // second word now that `State`'s layout is probed; everything else —
    // `&code.ip`, `&thread.ops_left`, the top frame's `base`, the `regs`
    // buffer — is a handful of loads over the probed layout. `nregs` is the
    // chunk's own `regs` field: a compile-time constant.
    let paused_p = em
        .fb
        .ins()
        .iadd_imm_s(em.env.ctx1, lyt.state_paused as i64);
    let opsleft_p = em
        .fb
        .ins()
        .iadd_imm_s(em.env.thread, lyt.ops_left_off as i64);
    let ip_p = em.fb.ins().iadd_imm_s(em.env.code, lyt.code_ip as i64);
    // `thread.frames.last().unwrap().base`
    let fptr = em.fb.ins().load(
        I64,
        tf(),
        em.env.thread,
        (lyt.frames_off + lyt.vec_ptr) as i32,
    );
    let flen = em.fb.ins().load(
        I64,
        tf(),
        em.env.thread,
        (lyt.frames_off + lyt.vec_len) as i32,
    );
    let fm1 = em.fb.ins().iadd_imm_s(flen, -1);
    let foff = em.fb.ins().imul_imm_s(fm1, lyt.frame_size as i64);
    let faddr = em.fb.ins().iadd(fptr, foff);
    let base = em.fb.ins().load(I64, tf(), faddr, lyt.frame_base as i32);
    // `thread.regs.as_mut_ptr()`
    let rp = em.fb.ins().load(
        I64,
        tf(),
        em.env.thread,
        (lyt.regs_off + lyt.vec_ptr) as i32,
    );
    let boff = em.fb.ins().imul_imm_s(base, lyt.val_size as i64);
    let regs0 = em.fb.ins().iadd(rp, boff);
    let nregs = em.iconst(chunk.regs as i64);
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
    // arm the batched quota: `bcn = min(*fuel, ops_left)` (bcgen's entry arm).
    // The driver already checked paused/fuel/ops_left before invoking the
    // body, so bcn >= 1 here.
    em.rearm_seq();

    // shadow init: `(v, ok) = match regs[r] { Int(v) => (v, true), _ => (0,false) }`
    let int_keys: Vec<u32> = {
        let mut k: Vec<u32> = em.v.int.keys().copied().collect();
        k.sort_unstable();
        k
    };
    for r in int_keys {
        let (sv, ok) = em.v.int[&r];
        let a = em.vaddr(regs0, r);
        let t = em.ld_tag(a);
        let want = em.tconst(lyt.t_int);
        let k = em.fb.ins().icmp(IntCC::Equal, t, want);
        let v = em.fb.ins().load(I64, tf(), a, lyt.val_pay as i32);
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
        let a = em.vaddr(regs0, r);
        let t = em.ld_tag(a);
        let want = em.tconst(lyt.t_float);
        let k = em.fb.ins().icmp(IntCC::Equal, t, want);
        let v = em.fb.ins().load(F64, tf(), a, lyt.val_pay as i32);
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
        em.fb.ins().jump(estep_g, &[]);
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
        let table = em.fb.create_jump_table(JumpTableData::new(def_bc, &calls));
        em.fb.ins().br_table(idx, table);
        // `edef` needs the raw ip for the step fallback
        em.fb.switch_to_block(edef);
        em.fb.def_var(v_cur_ip, ip);
        em.fb.ins().jump(estep_g, &[]);
    }

    // ---- shared exits ----
    // `estep_g` — the quota gate for `edef`/`eend`: `run_dispatch` runs its
    // paused/fuel/ops_left checks and charges one op even for a garbage
    // decode, so the fallback entry gates `bcn` exactly like an op would
    // (`gatep` — the stepped op before a re-dispatch can be anything).
    em.fb.switch_to_block(estep_g);
    let g_cont = em.fb.create_block();
    let g_tramp = em.fb.create_block();
    em.fb.set_cold_block(g_tramp);
    let pv = em.fb.ins().load(I8, tf(), em.env.paused_p, 0);
    em.fb.ins().brif(pv, g_tramp, &[], g_cont, &[]);
    em.fb.switch_to_block(g_cont);
    let g2 = em.fb.create_block();
    let bcnv = em.fb.use_var(v_bcn);
    let g0 = em.iconst(0);
    let bz = em.fb.ins().icmp(IntCC::Equal, bcnv, g0);
    em.fb.ins().brif(bz, g_tramp, &[], g2, &[]);
    em.fb.switch_to_block(g2);
    // merged `bcn` (the gate-tramp re-arm also lands here) — not `bcnv`
    let cur = em.fb.use_var(v_bcn);
    let b1 = em.fb.ins().iadd_imm_s(cur, -1);
    em.fb.def_var(v_bcn, b1);
    em.fb.ins().jump(estep, &[]);
    em.fill_gate_tramp(g_tramp, g2);

    em.fb.switch_to_block(enext);
    em.settle_seq();
    em.flush_seq();
    let ip = em.fb.use_var(v_cur_ip);
    em.store_ip(ip);
    em.hcall(H::OutNext, &[em.env.out]);
    em.fb.ins().return_(&[]);

    em.fb.switch_to_block(eoof);
    em.settle_seq();
    em.flush_seq();
    let ip = em.fb.use_var(v_cur_ip);
    em.store_opip(ip);
    let k = em.iconst8(ERR_OOF);
    em.hcall(H::OutErr, &[em.env.out, k]);
    em.fb.ins().return_(&[]);

    em.fb.switch_to_block(estep);
    em.settle_seq();
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
    em.settle_seq();
    em.flush_seq();
    let ip = em.fb.use_var(v_cur_ip);
    em.store_opip(ip);
    let kk = em.fb.use_var(v_ekind);
    em.hcall(H::OutErr, &[em.env.out, kk]);
    em.fb.ins().return_(&[]);

    em.fb.switch_to_block(eret);
    em.settle_seq();
    em.fb.ins().return_(&[]);

    em.fb.switch_to_block(eend);
    let max = em.iconst(-1); // usize::MAX — garbage decode, same as bcgen's `_ =>`
    em.fb.def_var(v_cur_ip, max);
    em.fb.ins().jump(estep_g, &[]);

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
            eprintln!(
                "verifier errors in body {body}:\n{errs}\n{}",
                ctx.func.display()
            );
        }
        e
    })
}

impl Em<'_> {
    /// One op block: the driver's per-op bookkeeping as a batched-quota gate
    /// (bcgen's `gateq!`/`gatep!`), then semantics.
    fn emit_op(&mut self, i: usize, off: usize, next: usize, op: &Op) {
        self.fb.switch_to_block(self.blocks[i]);
        let (_cont, tramp) = self.gate(i, off);
        self.semantics(i, off, next, op);
        // the gate's cold trampoline — `fill_gate_tramp` needs the current
        // block terminated, which `semantics` guarantees (every arm ends in a
        // branch or return).
        self.fill_gate_tramp(tramp, _cont);
    }

    fn semantics(&mut self, i: usize, off: usize, next: usize, op: &Op) {
        match op {
            Op::Move { dst, src } => {
                self.emit_move(*dst, *src);
                {
                    let nb = self.next_blk(i);
                    self.fb.ins().jump(nb, &[]);
                }
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
                // `regs[cond] == Val::Bool(is_true)` — tag + payload byte.
                let regs = self.regs();
                let a = self.vaddr(regs, cond.index() as u32);
                let t = self.ld_tag(a);
                let want = self.tconst(self.lyt.t_bool);
                let tb2 = self.fb.ins().icmp(IntCC::Equal, t, want);
                let pb = self.fb.ins().load(I8, tf(), a, self.lyt.bool_pay as i32);
                let bv = self.iconst8(*is_true as i64);
                let peq = self.fb.ins().icmp(IntCC::Equal, pb, bv);
                let k = self.fb.ins().band(tb2, peq);
                let hit = self.mask_by_shadow(*cond, k);
                let (tb, fb) = (self.tgt_blk(target), self.next_blk(i));
                self.fb.ins().brif(hit, tb, &[], fb, &[]);
            }
            Op::ForNext { idx, bound, target } => {
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
                self.settle_seq();
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
                    {
                        let nb = self.next_blk(i);
                        self.fb.ins().jump(nb, &[]);
                    }
                }
                Constant::Float(v) => {
                    let v = self.fb.ins().f64const(*v);
                    self.wr_float_dst(dst.index() as u32, v);
                    {
                        let nb = self.next_blk(i);
                        self.fb.ins().jump(nb, &[]);
                    }
                }
                Constant::Bool(b) => {
                    let v = self.iconst8(*b as i64);
                    let regs = self.regs();
                    let a = self.vaddr(regs, dst.index() as u32);
                    self.st_bool(a, v);
                    {
                        let nb = self.next_blk(i);
                        self.fb.ins().jump(nb, &[]);
                    }
                }
                Constant::Null => {
                    let regs = self.regs();
                    let a = self.vaddr(regs, dst.index() as u32);
                    self.st_null(a);
                    {
                        let nb = self.next_blk(i);
                        self.fb.ins().jump(nb, &[]);
                    }
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
                let regs = self.regs();
                let a = self.vaddr(regs, dst.index() as u32);
                let b = self.iconst32(body.index() as i64);
                self.st_fn(a, b);
                {
                    let nb = self.next_blk(i);
                    self.fb.ins().jump(nb, &[]);
                }
            }
            Op::LoadEntry { dst, slot } => {
                // entry-frame absolute slot: tb = regs - base*VS; v = tb[slot]
                let (regs, base) = (self.regs(), self.env.base);
                let boff = self.fb.ins().imul_imm_s(base, self.lyt.val_size as i64);
                let tb = self.fb.ins().isub(regs, boff);
                let vp = self
                    .fb
                    .ins()
                    .iadd_imm_s(tb, slot.index() as i64 * self.lyt.val_size as i64);
                let d = self.vaddr(regs, dst.index() as u32);
                self.cpy_val(d, vp);
                {
                    let nb = self.next_blk(i);
                    self.fb.ins().jump(nb, &[]);
                }
            }
            Op::StoreEntry { slot, src } => {
                let vp = self.val_ptr(src.index() as u32);
                let (regs, base) = (self.regs(), self.env.base);
                let boff = self.fb.ins().imul_imm_s(base, self.lyt.val_size as i64);
                let tb = self.fb.ins().isub(regs, boff);
                let d = self
                    .fb
                    .ins()
                    .iadd_imm_s(tb, slot.index() as i64 * self.lyt.val_size as i64);
                self.cpy_val(d, vp);
                {
                    let nb = self.next_blk(i);
                    self.fb.ins().jump(nb, &[]);
                }
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
                {
                    let nb = self.next_blk(i);
                    self.fb.ins().jump(nb, &[]);
                }
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
                {
                    let nb = self.next_blk(i);
                    self.fb.ins().jump(nb, &[]);
                }
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
            Op::AddFloat { dst, left, right } => self.emit_farit(i, *dst, *left, *right, FOp::Add),
            Op::SubFloat { dst, left, right } => self.emit_farit(i, *dst, *left, *right, FOp::Sub),
            Op::MultFloat { dst, left, right } => self.emit_farit(i, *dst, *left, *right, FOp::Mul),
            Op::DivFloat { dst, left, right } => self.emit_farit(i, *dst, *left, *right, FOp::Div),
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
            } => self.emit_brr(
                i,
                target,
                *left,
                Some(*right),
                None,
                *is_true,
                IntCC::SignedLessThan,
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
            } => self.emit_brr(
                i,
                target,
                *left,
                Some(*right),
                None,
                *is_true,
                IntCC::NotEqual,
            ),
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
            } => self.emit_brr(
                i,
                target,
                *left,
                None,
                Some(*val),
                *is_true,
                IntCC::NotEqual,
            ),
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
                FloatCC::LessThan,
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
            } => self.emit_brf(
                i,
                target,
                *left,
                Some(*right),
                None,
                *is_true,
                FloatCC::Equal,
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
                FloatCC::NotEqual,
            ),
            Op::BFloatLtImm {
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
                FloatCC::LessThan,
            ),
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
            } => self.emit_brf(
                i,
                target,
                *left,
                None,
                Some(*val),
                *is_true,
                FloatCC::NotEqual,
            ),
            Op::ToFloat { dst, src } => {
                let iv = self.int_opnd(src.index() as u32);
                let f = self.fb.ins().fcvt_from_sint(F64, iv);
                self.wr_float_dst(dst.index() as u32, f);
                {
                    let nb = self.next_blk(i);
                    self.fb.ins().jump(nb, &[]);
                }
            }
            Op::Sqrt { dst, src } => {
                let f = self.float_opnd(src.index() as u32);
                let r = self.fb.ins().sqrt(f);
                self.wr_float_dst(dst.index() as u32, r);
                {
                    let nb = self.next_blk(i);
                    self.fb.ins().jump(nb, &[]);
                }
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
                self.emit_set_index(i, off, next, *set, *index, *value);
            }
            Op::GetIndex {
                dst,
                set,
                index,
                kind,
            } => {
                self.emit_get_index(i, off, next, *dst, *set, *index, *kind);
            }
            Op::GetField {
                dst,
                src,
                slot,
                kind,
            } => {
                self.emit_get_field(i, off, next, *dst, *src, *slot, *kind);
            }
            Op::SetField {
                receiver,
                slot,
                value,
            } => {
                self.emit_set_field(i, off, next, *receiver, *slot, *value);
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
                {
                    let nb = self.next_blk(i);
                    self.fb.ins().jump(nb, &[]);
                }
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
        {
            let nb = self.next_blk(i);
            self.fb.ins().jump(nb, &[]);
        }
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
        {
            let nb = self.next_blk(i);
            self.fb.ins().jump(nb, &[]);
        }
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
        let regs = self.regs();
        let a = self.vaddr(regs, idx);
        let t = self.ld_tag(a);
        let want = self.tconst(self.lyt.t_bool);
        let ok = self.fb.ins().icmp(IntCC::Equal, t, want);
        self.fb.ins().brif(ok, good, &[], self.ex.estep, &[]);
        self.fb.switch_to_block(good);
        self.fb.ins().load(I8, tf(), a, self.lyt.bool_pay as i32)
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
        {
            let nb = self.next_blk(i);
            self.fb.ins().jump(nb, &[]);
        }
        self.fill_err(tramp, ERR_MOD0);
    }

    fn emit_eval_i(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, cc: IntCC) {
        let a = self.int_opnd(left.index() as u32);
        let b = self.int_opnd(right.index() as u32);
        let c = self.fb.ins().icmp(cc, a, b);
        self.wr_bool_dst(dst.index() as u32, c);
        {
            let nb = self.next_blk(i);
            self.fb.ins().jump(nb, &[]);
        }
    }

    fn emit_eval_imm_i(&mut self, i: usize, dst: Reg, left: Reg, val: i64, cc: IntCC) {
        let a = self.int_opnd(left.index() as u32);
        let b = self.iconst(val);
        let c = self.fb.ins().icmp(cc, a, b);
        self.wr_bool_dst(dst.index() as u32, c);
        {
            let nb = self.next_blk(i);
            self.fb.ins().jump(nb, &[]);
        }
    }

    fn emit_eval_f(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, cc: FloatCC) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.float_opnd(right.index() as u32);
        let c = self.fb.ins().fcmp(cc, a, b);
        self.wr_bool_dst(dst.index() as u32, c);
        {
            let nb = self.next_blk(i);
            self.fb.ins().jump(nb, &[]);
        }
    }

    fn emit_eval_fimm(&mut self, i: usize, dst: Reg, left: Reg, val: i64, cc: FloatCC) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.fb.ins().f64const(f64::from_bits(val as u64));
        let c = self.fb.ins().fcmp(cc, a, b);
        self.wr_bool_dst(dst.index() as u32, c);
        {
            let nb = self.next_blk(i);
            self.fb.ins().jump(nb, &[]);
        }
    }

    fn emit_farit(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, o: FOp) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.float_opnd(right.index() as u32);
        let v = o.emit(&mut self.fb, a, b);
        self.wr_float_dst(dst.index() as u32, v);
        {
            let nb = self.next_blk(i);
            self.fb.ins().jump(nb, &[]);
        }
    }

    fn emit_farit_imm(&mut self, i: usize, dst: Reg, left: Reg, val: i64, o: FOp) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.fb.ins().f64const(f64::from_bits(val as u64));
        let v = o.emit(&mut self.fb, a, b);
        self.wr_float_dst(dst.index() as u32, v);
        {
            let nb = self.next_blk(i);
            self.fb.ins().jump(nb, &[]);
        }
    }

    /// `StrEq`/`StrNe`: the (Str, Str) fast path via `bin_str`; anything else
    /// runs the op through `step` (which reaches `bin_cold` for the general
    /// pair — the interpreter's own behavior).
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

    // ---- typed container access (the physics path) ----
    //
    // `(Array, Int)` index and `Instance`/`Array` field ops inline over the
    // probed layout instead of FFI-ing a helper. Every fast-path miss —
    // wrong tags, borrow-flag contention, out-of-bounds (the interpreter's
    // own indexing panics), Gc-carrying writes (the write barrier `borrow_mut`
    // performs) — routes to the same helper the op used before, so nothing
    // observable changes: the helpers hold the verbatim arm.
    //
    // Barrier rule: a `Val` store may adopt a `Gc` pointer, which requires
    // `Gc::write`'s backward barrier — that's what `borrow_mut(&ctx)` does
    // behind the helper. The inline store is only reached when the value's
    // tag is Null/Bool/Int/Float/Fn — non-Gc payloads adopt nothing, so the
    // barrier is a no-op for them and may be skipped.

    /// `regs[r]`'s discriminant widened to i64 for mask arithmetic.
    fn ld_tag64(&mut self, a: Value) -> Value {
        let t = self.ld_tag(a);
        if self.lyt.tag_size < 8 {
            self.fb.ins().uextend(I64, t)
        } else {
            t
        }
    }

    /// `tag` (i64) names a non-Gc `Val` variant → the value adopts nothing and
    /// a container write may skip `Gc::write`'s barrier.
    fn is_non_gc_tag(&mut self, t64: Value) -> Value {
        let mask: u64 = [self.lyt.t_null, self.lyt.t_bool, self.lyt.t_int, self.lyt.t_float, self
            .lyt
            .t_fn]
            .iter()
            .map(|t| {
                assert!(*t < 64, "Val discriminant exceeds mask domain");
                1u64 << t
            })
            .sum();
        let m = self.iconst(mask as i64);
        let one = self.iconst(1);
        let bit = self.fb.ins().ishl(one, t64);
        let hit = self.fb.ins().band(bit, m);
        // guard the shift domain: a tag >= 64 can't be a real discriminant,
        // but a masked-out answer must be *certain*, not just plausible
        let inr = self
            .fb
            .ins()
            .icmp_imm_u(IntCC::UnsignedLessThan, t64, 64);
        let nz = self.fb.ins().icmp_imm_u(IntCC::NotEqual, hit, 0);
        self.fb.ins().band(nz, inr)
    }

    /// Load a container's `RefCell` borrow flag; returns the `flag passes`
    /// condition (`mutable` picks `borrow_mut`'s flag==0 precondition, reads
    /// `borrow()`'s flag>=0). Folded into the elem-access `brif` — a failed
    /// flag routes to `slow`, where the helper's own `borrow`/`borrow_mut`
    /// reproduces the panic.
    fn borrow_ok(&mut self, gc: Value, flag_off: usize, mutable: bool) -> Value {
        let flag = self.fb.ins().load(I64, tf(), gc, flag_off as i32);
        let z = self.iconst(0);
        let cc = if mutable {
            IntCC::Equal
        } else {
            IntCC::SignedGreaterThanOrEqual
        };
        self.fb.ins().icmp(cc, flag, z)
    }

    /// `*dst = data[slot]` under `pre && slot < len` — the shared tail of the
    /// container reads. A miss → `slow` (the helper/indexing panic lives
    /// there). `pre` carries the borrow-flag (and Fields-tag / non-Gc)
    /// checks, folded into the same `brif` so the fast path is one branch.
    /// Also refreshes `dst`'s scalar shadows: `GetField`/`GetIndex` dsts are
    /// `W::DynCheck`-eligible, so a hit keeps float temps in registers.
    #[allow(clippy::too_many_arguments)]
    fn read_elem(
        &mut self,
        i: usize,
        dst: Reg,
        slot: Value,
        len: Value,
        data: Value,
        pre: Value,
        slow: Block,
    ) {
        let inb = self
            .fb
            .ins()
            .icmp(IntCC::UnsignedLessThan, slot, len);
        let k = self.fb.ins().band(pre, inb);
        let good = self.fb.create_block();
        self.fb.ins().brif(k, good, &[], slow, &[]);
        self.fb.switch_to_block(good);
        let off = self
            .fb
            .ins()
            .imul_imm_s(slot, self.lyt.val_size as i64);
        let sa = self.fb.ins().iadd(data, off);
        let d = dst.index() as u32;
        if self.v.int.contains_key(&d) || self.v.float.contains_key(&d) {
            // DynCheck shadow refresh: tag-probe the loaded elem, keep the
            // payload in `sv` + `ok` for downstream scalar ops.
            let t = self.ld_tag(sa);
            if let Some(&(sv, ok)) = self.v.int.get(&d) {
                let want = self.tconst(self.lyt.t_int);
                let k = self.fb.ins().icmp(IntCC::Equal, t, want);
                let pv = self.fb.ins().load(I64, tf(), sa, self.lyt.val_pay as i32);
                self.fb.def_var(sv, pv);
                self.fb.def_var(ok, k);
            }
            if let Some(&(sv, ok)) = self.v.float.get(&d) {
                let want = self.tconst(self.lyt.t_float);
                let k = self.fb.ins().icmp(IntCC::Equal, t, want);
                let pv = self.fb.ins().load(F64, tf(), sa, self.lyt.val_pay as i32);
                self.fb.def_var(sv, pv);
                self.fb.def_var(ok, k);
            }
        }
        let regs = self.regs();
        let dd = self.vaddr(regs, d);
        self.cpy_val(dd, sa);
        let nb = self.next_blk(i);
        self.fb.ins().jump(nb, &[]);
    }

    /// `data[slot] = *srcv` under `pre && slot < len` — the shared tail of the
    /// container writes.
    fn write_elem(
        &mut self,
        i: usize,
        slot: Value,
        len: Value,
        data: Value,
        srcv: Value,
        pre: Value,
        slow: Block,
    ) {
        let inb = self
            .fb
            .ins()
            .icmp(IntCC::UnsignedLessThan, slot, len);
        let k = self.fb.ins().band(pre, inb);
        let good = self.fb.create_block();
        self.fb.ins().brif(k, good, &[], slow, &[]);
        self.fb.switch_to_block(good);
        let off = self
            .fb
            .ins()
            .imul_imm_s(slot, self.lyt.val_size as i64);
        let sa = self.fb.ins().iadd(data, off);
        self.cpy_val(sa, srcv);
        let nb = self.next_blk(i);
        self.fb.ins().jump(nb, &[]);
    }

    /// `(len, data)` of an `Instance`'s `Fields` at `fp`, restricted to the
    /// `Inline` variant: returns the `(len, data, isinl)` triple so callers
    /// fold the tag check into the access `brif`; `Spilled` therefore routes
    /// to `slow`, where the helper's `Fields` `Index` covers it identically.
    /// The `len` byte sits inside the enum's allocation either way, so the
    /// unconditional load is safe.
    fn fields_inline(&mut self, fp: Value) -> (Value, Value, Value) {
        let fty = match self.lyt.fld_tsz {
            1 => I8,
            2 => types::I16,
            4 => I32,
            8 => I64,
            d => unreachable!("bad Fields tag width {d}"),
        };
        let ftag = self.fb.ins().load(fty, tf(), fp, self.lyt.fld_tag as i32);
        let want = self.fb.ins().iconst(fty, self.lyt.fld_inline as i64);
        let isinl = self.fb.ins().icmp(IntCC::Equal, ftag, want);
        let l8 = self.fb.ins().load(I8, tf(), fp, self.lyt.fld_len as i32);
        let len = self.fb.ins().uextend(I64, l8);
        let data = self.fb.ins().iadd_imm_s(fp, self.lyt.fld_data as i64);
        (len, data, isinl)
    }

    /// `Op::GetIndex` — inline `(Array, Int)`; `kind == Option` and every
    /// miss go through the `mj_get_index` helper.
    fn emit_get_index(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        dst: Reg,
        set: Reg,
        index: Reg,
        kind: AccessKind,
    ) {
        if kind != AccessKind::Direct {
            let (regs, d, s, ii, kk) = (
                self.regs(),
                self.iconst(dst.index() as i64),
                self.iconst(set.index() as i64),
                self.iconst(index.index() as i64),
                self.iconst8(kind as i64),
            );
            return self.helper_op_read(
                i,
                off,
                next,
                H::GetIndex,
                &[regs, d, s, ii, kk, self.env.ctx0, self.env.ctx1, self.env.out],
                dst,
            );
        }
        // No `flush_seq`/`mark_op` on the fast path: the only `regs` slot read
        // here is `set` (materialized by `val_ptr`), `estep` marks from
        // `cur_ip` and re-flushes, and the `slow` helper re-marks anyway.
        let slow = self.fb.create_block();
        self.fb.set_cold_block(slow);
        // index must be Int — a shadow/tag miss runs the op under `step`,
        // which reaches the same `get_index` arm
        let iv = self.int_opnd(index.index() as u32);
        let sa = self.val_ptr(set.index() as u32);
        let st = self.ld_tag(sa);
        let tarr = self.tconst(self.lyt.t_array);
        let isarr = self.fb.ins().icmp(IntCC::Equal, st, tarr);
        let arrb = self.fb.create_block();
        self.fb.ins().brif(isarr, arrb, &[], slow, &[]);
        self.fb.switch_to_block(arrb);
        let gc = self
            .fb
            .ins()
            .load(I64, tf(), sa, self.lyt.arr_pay as i32);
        let fok = self.borrow_ok(gc, self.lyt.rl_flag, false);
        let vp = self
            .fb
            .ins()
            .load(I64, tf(), gc, (self.lyt.rl_vec + self.lyt.vec_ptr) as i32);
        let vl = self
            .fb
            .ins()
            .load(I64, tf(), gc, (self.lyt.rl_vec + self.lyt.vec_len) as i32);
        self.read_elem(i, dst, iv, vl, vp, fok, slow);
        // ---- slow: verbatim helper + DynCheck shadow refresh ----
        self.fb.switch_to_block(slow);
        let (regs, d, s, ii) = (
            self.regs(),
            self.iconst(dst.index() as i64),
            self.iconst(set.index() as i64),
            self.iconst(index.index() as i64),
        );
        let kk = self.iconst8(kind as i64);
        self.helper_op_read(
            i,
            off,
            next,
            H::GetIndex,
            &[regs, d, s, ii, kk, self.env.ctx0, self.env.ctx1, self.env.out],
            dst,
        );
    }

    /// `Op::SetIndex` — inline `(Array, Int, non-Gc)`; everything else goes
    /// through the `mj_set_index` helper (which carries the write barrier).
    fn emit_set_index(&mut self, i: usize, off: usize, next: usize, set: Reg, index: Reg, value: Reg) {
        // selective materialization instead of `flush_seq` — see
        // `emit_get_index`.
        let slow = self.fb.create_block();
        self.fb.set_cold_block(slow);
        let iv = self.int_opnd(index.index() as u32);
        let sa = self.val_ptr(set.index() as u32);
        let st = self.ld_tag(sa);
        let tarr = self.tconst(self.lyt.t_array);
        let isarr = self.fb.ins().icmp(IntCC::Equal, st, tarr);
        let arrb = self.fb.create_block();
        self.fb.ins().brif(isarr, arrb, &[], slow, &[]);
        self.fb.switch_to_block(arrb);
        let gc = self
            .fb
            .ins()
            .load(I64, tf(), sa, self.lyt.arr_pay as i32);
        let fok = self.borrow_ok(gc, self.lyt.rl_flag, true);
        // value must adopt no Gc pointer → the write barrier is a no-op
        let va = self.val_ptr(value.index() as u32);
        let vt = self.ld_tag64(va);
        let ngc = self.is_non_gc_tag(vt);
        let pre = self.fb.ins().band(fok, ngc);
        let vp = self
            .fb
            .ins()
            .load(I64, tf(), gc, (self.lyt.rl_vec + self.lyt.vec_ptr) as i32);
        let vl = self
            .fb
            .ins()
            .load(I64, tf(), gc, (self.lyt.rl_vec + self.lyt.vec_len) as i32);
        self.write_elem(i, iv, vl, vp, va, pre, slow);
        // ---- slow: verbatim helper ----
        self.fb.switch_to_block(slow);
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

    /// `Op::GetField` — inline `Instance`/`Array` receivers at `kind ==
    /// Direct`; `Option` and misses go through the `mj_get_field` helper.
    #[allow(clippy::too_many_arguments)]
    fn emit_get_field(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        dst: Reg,
        src: Reg,
        slot: u32,
        kind: AccessKind,
    ) {
        if kind != AccessKind::Direct {
            let (regs, d, s, sl, kk) = (
                self.regs(),
                self.iconst(dst.index() as i64),
                self.iconst(src.index() as i64),
                self.iconst(slot as i64),
                self.iconst8(kind as i64),
            );
            return self.helper_op_read(
                i,
                off,
                next,
                H::GetField,
                &[regs, d, s, sl, kk, self.env.out],
                dst,
            );
        }
        // selective materialization instead of `flush_seq` — see
        // `emit_get_index`
        let slow = self.fb.create_block();
        self.fb.set_cold_block(slow);
        let ra = self.val_ptr(src.index() as u32);
        let rt = self.ld_tag(ra);
        let tinst = self.tconst(self.lyt.t_instance);
        let isinst = self.fb.ins().icmp(IntCC::Equal, rt, tinst);
        let instb = self.fb.create_block();
        let noti = self.fb.create_block();
        self.fb.ins().brif(isinst, instb, &[], noti, &[]);
        // ---- receiver is Array: `a.0.borrow()[slot]` ----
        self.fb.switch_to_block(noti);
        let tarr = self.tconst(self.lyt.t_array);
        let isarr = self.fb.ins().icmp(IntCC::Equal, rt, tarr);
        let arrb = self.fb.create_block();
        self.fb.ins().brif(isarr, arrb, &[], slow, &[]);
        self.fb.switch_to_block(arrb);
        let gc = self
            .fb
            .ins()
            .load(I64, tf(), ra, self.lyt.arr_pay as i32);
        let fok = self.borrow_ok(gc, self.lyt.rl_flag, false);
        let vp = self
            .fb
            .ins()
            .load(I64, tf(), gc, (self.lyt.rl_vec + self.lyt.vec_ptr) as i32);
        let vl = self
            .fb
            .ins()
            .load(I64, tf(), gc, (self.lyt.rl_vec + self.lyt.vec_len) as i32);
        let sv = self.iconst(slot as i64);
        self.read_elem(i, dst, sv, vl, vp, fok, slow);
        // ---- receiver is Instance: `i.0.borrow().fields[slot]` ----
        // (`Fields::Spilled` goes to `slow` — the helper's `Index` covers it.)
        self.fb.switch_to_block(instb);
        let gc = self
            .fb
            .ins()
            .load(I64, tf(), ra, self.lyt.inst_pay as i32);
        let fok = self.borrow_ok(gc, self.lyt.rl_flag_i, false);
        let fp = self
            .fb
            .ins()
            .iadd_imm_s(gc, (self.lyt.rl_inst + self.lyt.id_fields) as i64);
        let (len, data, isinl) = self.fields_inline(fp);
        let pre = self.fb.ins().band(fok, isinl);
        let sv = self.iconst(slot as i64);
        self.read_elem(i, dst, sv, len, data, pre, slow);
        // ---- slow: verbatim helper + DynCheck shadow refresh ----
        self.fb.switch_to_block(slow);
        let (regs, d, s, sl, kk) = (
            self.regs(),
            self.iconst(dst.index() as i64),
            self.iconst(src.index() as i64),
            self.iconst(slot as i64),
            self.iconst8(kind as i64),
        );
        self.helper_op_read(
            i,
            off,
            next,
            H::GetField,
            &[regs, d, s, sl, kk, self.env.out],
            dst,
        );
    }

    /// `Op::SetField` — inline `Instance`/`Array` receivers storing non-Gc
    /// values; misses and Gc-carrying values go through `mj_set_field`.
    fn emit_set_field(&mut self, i: usize, off: usize, next: usize, receiver: Reg, slot: u32, value: Reg) {
        // selective materialization instead of `flush_seq` — see
        // `emit_get_index`
        let slow = self.fb.create_block();
        self.fb.set_cold_block(slow);
        let ra = self.val_ptr(receiver.index() as u32);
        let rt = self.ld_tag(ra);
        // the value's tag decides whether the write needs the GC barrier —
        // checked once here, in the dominating block
        let va = self.val_ptr(value.index() as u32);
        let vt = self.ld_tag64(va);
        let ngc = self.is_non_gc_tag(vt);
        let tinst = self.tconst(self.lyt.t_instance);
        let isinst = self.fb.ins().icmp(IntCC::Equal, rt, tinst);
        let instb = self.fb.create_block();
        let noti = self.fb.create_block();
        self.fb.ins().brif(isinst, instb, &[], noti, &[]);
        // ---- receiver is Array: `a.0.borrow_mut(&ctx)[slot] = v` ----
        self.fb.switch_to_block(noti);
        let tarr = self.tconst(self.lyt.t_array);
        let isarr = self.fb.ins().icmp(IntCC::Equal, rt, tarr);
        let arrb = self.fb.create_block();
        self.fb.ins().brif(isarr, arrb, &[], slow, &[]);
        self.fb.switch_to_block(arrb);
        let gc = self
            .fb
            .ins()
            .load(I64, tf(), ra, self.lyt.arr_pay as i32);
        let fok = self.borrow_ok(gc, self.lyt.rl_flag, true);
        let pre = self.fb.ins().band(fok, ngc);
        let vp = self
            .fb
            .ins()
            .load(I64, tf(), gc, (self.lyt.rl_vec + self.lyt.vec_ptr) as i32);
        let vl = self
            .fb
            .ins()
            .load(I64, tf(), gc, (self.lyt.rl_vec + self.lyt.vec_len) as i32);
        let sv = self.iconst(slot as i64);
        self.write_elem(i, sv, vl, vp, va, pre, slow);
        // ---- receiver is Instance: `i.0.borrow_mut(&ctx).fields[slot] = v` ----
        self.fb.switch_to_block(instb);
        let gc = self
            .fb
            .ins()
            .load(I64, tf(), ra, self.lyt.inst_pay as i32);
        let fok = self.borrow_ok(gc, self.lyt.rl_flag_i, true);
        let fp = self
            .fb
            .ins()
            .iadd_imm_s(gc, (self.lyt.rl_inst + self.lyt.id_fields) as i64);
        let (len, data, isinl) = self.fields_inline(fp);
        let pre = self.fb.ins().band(fok, ngc);
        let pre = self.fb.ins().band(pre, isinl);
        let sv = self.iconst(slot as i64);
        self.write_elem(i, sv, len, data, va, pre, slow);
        // ---- slow: verbatim helper ----
        self.fb.switch_to_block(slow);
        let (regs, r, sl, v) = (
            self.regs(),
            self.iconst(receiver.index() as i64),
            self.iconst(slot as i64),
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
            let vp = self.vaddr(regs, s);
            let dd = self.vaddr(regs, d);
            self.cpy_val(dd, vp);
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
            let vp = self.vaddr(regs, s);
            let dd = self.vaddr(regs, d);
            self.cpy_val(dd, vp);
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
            let regs = self.regs();
            let dd = self.vaddr(regs, d);
            if dst_is_int {
                self.st_float(dd, v);
            } else {
                self.st_int(dd, v);
            }
            let z = self.iconst8(0);
            self.fb.def_var(dok, z);
            self.fb.ins().jump(done, &[]);
            self.fb.switch_to_block(genb);
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
        let w8 = self.tconst(want);
        let k = self.fb.ins().icmp(IntCC::Equal, t, w8);
        self.fb.ins().brif(k, good, &[], miss, &[]);
        self.fb.switch_to_block(good);
        let v = self.fb.ins().load(
            if dst_is_int { I64 } else { F64 },
            tf(),
            sa,
            self.lyt.val_pay as i32,
        );
        self.fb.def_var(dsv, v);
        let one = self.iconst8(1);
        self.fb.def_var(dok, one);
        self.fb.ins().jump(done, &[]);
        self.fb.switch_to_block(miss);
        let regs = self.regs();
        let vp = self.vaddr(regs, s);
        let dd = self.vaddr(regs, d);
        self.cpy_val(dd, vp);
        let z = self.iconst8(0);
        self.fb.def_var(dok, z);
        self.fb.ins().jump(done, &[]);
        self.fb.switch_to_block(done);
    }

    /// `CallDirect`. Fast path (inline, no FFI): depth below
    /// `INLINE_CALL_DEPTH`, `thread.regs`/`thread.frames` capacity already
    /// sufficient, static arity match — then the `enter_call_regs` frame push
    /// is emitted inline over the probed layout, the callee runs as a direct
    /// native `call`, and `mj_pop_return` runs the driver's `Flow::Return`
    /// handling. Anything slower — capacity growth, depth cap (which must
    /// produce `Flow::Call`), or a malformed arity — routes to the
    /// `mj_call_body` megashim, unchanged.
    fn emit_call_direct(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        dst: Reg,
        body: compile::BodyId,
        args: &[Reg],
    ) {
        let cchunk = &self.prog.chunks[body];
        // Compile-time gates: a static arity miss and a callee with captures
        // (only ever entered through `Val::Closure`) keep the megashim, which
        // reports `WrongArity` / hits the captures debug_assert exactly like
        // before. Large frames take the shim too — the unrolled Null fill
        // isn't worth it there.
        let fast_ok = args.len() == cchunk.args as usize
            && cchunk.captures.is_empty()
            && (cchunk.regs as usize) <= 64;
        if fast_ok {
            return self.emit_call_direct_inline(i, off, next, dst, body, args, cchunk);
        }
        let ap = self.reg_list_slot(args);
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
            self.iconst(body.index() as i64),
            self.iconst(dst.index() as i64),
            self.iconst(nargs as i64),
        );
        let rp = self
            .hcall(
                H::CallBody,
                &[
                    self.env.thread,
                    self.env.code,
                    self.env.chunks,
                    self.env.bodies_tbl,
                    b,
                    d,
                    ap,
                    n,
                    self.env.ctx0,
                    self.env.ctx1,
                    self.env.strs,
                    self.env.sigs,
                    self.env.fuel_p,
                    self.env.opip_p,
                    self.env.out,
                ],
            )
            .unwrap();
        self.post_call(i, dst, rp);
    }

    /// The inline `CallDirect` fast path — see [`Em::emit_call_direct`].
    #[allow(clippy::too_many_arguments)]
    fn emit_call_direct_inline(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        dst: Reg,
        body: compile::BodyId,
        args: &[Reg],
        cchunk: &compile::Chunk,
    ) {
        let vs = self.lyt.val_size as i64;
        let fsz = self.lyt.frame_size as i64;
        self.flush_seq();
        // `code.ip = next; *op_ip = off` — on the fast path `code.ip` gets
        // overwritten with the callee offset below, but `*op_ip` locates this
        // op for any propagated error, and the slow path needs both.
        self.mark_op(off, next);
        self.settle_seq();

        // Runtime gates, all checked in the pre-branch block so the fast
        // block can reuse the loaded Vec headers:
        //   frames.len() < INLINE_CALL_DEPTH    (else Flow::Call to driver)
        //   regs.cap - regs.len >= callee.regs  (else Vec grow → shim)
        //   frames.len() < frames.cap           (ditto for the push)
        let flen = self.frames_len();
        let dcap = self.iconst(INLINE_CALL_DEPTH as i64);
        let depth_ok = self.fb.ins().icmp(IntCC::UnsignedLessThan, flen, dcap);
        let rlen = self.fb.ins().load(
            I64,
            tf(),
            self.env.thread,
            (self.lyt.regs_off + self.lyt.vec_len) as i32,
        );
        let rcap = self.fb.ins().load(
            I64,
            tf(),
            self.env.thread,
            (self.lyt.regs_off + self.lyt.vec_cap) as i32,
        );
        let slack = self.fb.ins().isub(rcap, rlen);
        let need = self.iconst(cchunk.regs as i64);
        let regs_ok = self
            .fb
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, slack, need);
        let fcap = self.fb.ins().load(
            I64,
            tf(),
            self.env.thread,
            (self.lyt.frames_off + self.lyt.vec_cap) as i32,
        );
        let frames_ok = self.fb.ins().icmp(IntCC::UnsignedLessThan, flen, fcap);
        let ok01 = self.fb.ins().band(depth_ok, regs_ok);
        let ok = self.fb.ins().band(ok01, frames_ok);
        let fast = self.fb.create_block();
        let slow = self.fb.create_block();
        self.fb.set_cold_block(slow);
        self.fb.ins().brif(ok, fast, &[], slow, &[]);

        // ---- slow: the megashim handles Flow::Call-at-cap, grows, errors ----
        self.fb.switch_to_block(slow);
        let ap = self.reg_list_slot(args);
        let (b, d, n) = (
            self.iconst(body.index() as i64),
            self.iconst(dst.index() as i64),
            self.iconst(args.len() as i64),
        );
        let rp = self
            .hcall(
                H::CallBody,
                &[
                    self.env.thread,
                    self.env.code,
                    self.env.chunks,
                    self.env.bodies_tbl,
                    b,
                    d,
                    ap,
                    n,
                    self.env.ctx0,
                    self.env.ctx1,
                    self.env.strs,
                    self.env.sigs,
                    self.env.fuel_p,
                    self.env.opip_p,
                    self.env.out,
                ],
            )
            .unwrap();
        self.post_call(i, dst, rp);

        // ---- fast: enter_call_regs inline ----
        self.fb.switch_to_block(fast);
        let rp0 = self.fb.ins().load(
            I64,
            tf(),
            self.env.thread,
            (self.lyt.regs_off + self.lyt.vec_ptr) as i32,
        );
        // new_base = old regs.len (loaded above as `rlen`)
        let nboff = self.fb.ins().imul_imm_s(rlen, vs);
        let nwin = self.fb.ins().iadd(rp0, nboff); // callee window base
        // resize fill: `Val::Null` — only the tag byte is ever read
        let tnull = self.tconst(self.lyt.t_null);
        for k in 0..cchunk.regs as i32 {
            let a = self
                .fb
                .ins()
                .iadd_imm_s(nwin, k as i64 * vs + self.lyt.val_tag as i64);
            self.fb.ins().store(tf(), tnull, a, 0);
        }
        let nlen = self.fb.ins().iadd_imm_s(rlen, cchunk.regs as i64);
        self.fb.ins().store(
            tf(),
            nlen,
            self.env.thread,
            (self.lyt.regs_off + self.lyt.vec_len) as i32,
        );
        // arg copies: `regs[new_base + param] = regs[caller_base + arg]` —
        // both indices are compile-time constants; the caller window is
        // `v.regs` (unmoved — capacity was checked).
        let caller = self.regs();
        for (i, &pr) in cchunk.params.iter().enumerate() {
            let s = self.vaddr(caller, args[i].index() as u32);
            let d = self.vaddr(nwin, pr.index() as u32);
            self.cpy_val(d, s);
        }
        // caller frame's saved ip = resume offset (what `code.ip` held)
        let fptr = self.fb.ins().load(
            I64,
            tf(),
            self.env.thread,
            (self.lyt.frames_off + self.lyt.vec_ptr) as i32,
        );
        let fm1 = self.fb.ins().iadd_imm_s(flen, -1);
        let cfo = self.fb.ins().imul_imm_s(fm1, fsz);
        let cf = self.fb.ins().iadd(fptr, cfo);
        let nxt = self.iconst(next as i64);
        self.fb.ins().store(tf(), nxt, cf, self.lyt.frame_ip as i32);
        // push the callee frame
        let nfo = self.fb.ins().imul_imm_s(flen, fsz);
        let nf = self.fb.ins().iadd(fptr, nfo);
        let bi = self.iconst32(body.index() as i64);
        self.fb.ins().store(tf(), bi, nf, self.lyt.frame_chunk as i32);
        let cip = self.iconst(cchunk.offset as i64);
        self.fb.ins().store(tf(), cip, nf, self.lyt.frame_ip as i32);
        let rr = self.iconst32(dst.index() as i64);
        self.fb.ins().store(tf(), rr, nf, self.lyt.frame_ret as i32);
        self.fb.ins().store(tf(), rlen, nf, self.lyt.frame_base as i32);
        let nfl = self.fb.ins().iadd_imm_s(flen, 1);
        self.fb.ins().store(
            tf(),
            nfl,
            self.env.thread,
            (self.lyt.frames_off + self.lyt.vec_len) as i32,
        );
        // code.ip = callee chunk offset
        self.store_ip(cip);
        // the call itself — a direct native call, same `out` slot
        let fref = self.call_refs[&(body.index() as u32)];
        self.fb.ins().call(
            fref,
            &[
                self.env.thread,
                self.env.code,
                self.env.ctx0,
                self.env.ctx1,
                self.env.strs,
                self.env.chunks,
                self.env.sigs,
                self.env.fuel_p,
                self.env.opip_p,
                self.env.out,
            ],
        );
        // driver's `Flow::Return` pop, inlined over the probed
        // `RtResult<Flow>`/Frame layout: `*out == Ok(Flow::Return(v))` → pop
        // the callee frame, truncate `regs`, restore the caller's saved ip,
        // write `v` into the `return_reg` slot; anything else propagates
        // `out` verbatim through `eret` (what `mj_pop_return`'s non-2 tags
        // did).
        let oty = match self.lyt.out_tsz {
            1 => I8,
            2 => types::I16,
            4 => I32,
            8 => I64,
            d => unreachable!("bad out tag width {d}"),
        };
        let otag = self
            .fb
            .ins()
            .load(oty, tf(), self.env.out, self.lyt.out_tag as i32);
        let orwant = self.fb.ins().iconst(oty, self.lyt.out_ret as i64);
        let isret = self.fb.ins().icmp(IntCC::Equal, otag, orwant);
        let resumed = self.fb.create_block();
        self.fb.ins().brif(isret, resumed, &[], self.ex.eret, &[]);
        self.fb.switch_to_block(resumed);
        // popped = frames[flen-1]; frames.len -= 1; regs.len = popped.base
        let flen2 = self.frames_len();
        let fptr2 = self.fb.ins().load(
            I64,
            tf(),
            self.env.thread,
            (self.lyt.frames_off + self.lyt.vec_ptr) as i32,
        );
        let fm2 = self.fb.ins().iadd_imm_s(flen2, -1);
        let pfo = self.fb.ins().imul_imm_s(fm2, fsz);
        let pf = self.fb.ins().iadd(fptr2, pfo);
        let pbase = self.fb.ins().load(I64, tf(), pf, self.lyt.frame_base as i32);
        let pret32 = self
            .fb
            .ins()
            .load(I32, tf(), pf, self.lyt.frame_ret as i32);
        let pret = self.fb.ins().uextend(I64, pret32);
        self.fb.ins().store(
            tf(),
            fm2,
            self.env.thread,
            (self.lyt.frames_off + self.lyt.vec_len) as i32,
        );
        self.fb.ins().store(
            tf(),
            pbase,
            self.env.thread,
            (self.lyt.regs_off + self.lyt.vec_len) as i32,
        );
        // caller = frames[flen-2]; code.ip = caller.ip
        let fm3 = self.fb.ins().iadd_imm_s(flen2, -2);
        let cfo2 = self.fb.ins().imul_imm_s(fm3, fsz);
        let cf2 = self.fb.ins().iadd(fptr2, cfo2);
        let cip2 = self.fb.ins().load(I64, tf(), cf2, self.lyt.frame_ip as i32);
        self.store_ip(cip2);
        let cbase = self
            .fb
            .ins()
            .load(I64, tf(), cf2, self.lyt.frame_base as i32);
        // `thread.regs` may have moved under the callee — rebuild the window
        // and write the return value into `caller_base + return_reg`
        let rp2 = self.fb.ins().load(
            I64,
            tf(),
            self.env.thread,
            (self.lyt.regs_off + self.lyt.vec_ptr) as i32,
        );
        let widx = self.fb.ins().iadd(cbase, pret);
        let woff = self.fb.ins().imul_imm_s(widx, vs);
        let daddr = self.fb.ins().iadd(rp2, woff);
        let retp = self
            .fb
            .ins()
            .iadd_imm_s(self.env.out, self.lyt.out_ret_pay as i64);
        self.cpy_val(daddr, retp);
        let boff2 = self.fb.ins().imul_imm_s(cbase, vs);
        let regs2 = self.fb.ins().iadd(rp2, boff2);
        self.fb.def_var(self.v.regs, regs2);
        self.rearm_seq();
        self.dst_refresh(i, dst, regs2);
    }

    /// `Call` — one `mj_call_dyn` hop: callee resolution (incl. the
    /// `CallTarget::Value` signature check), depth cap, enter/run/pop.
    fn emit_call(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        dst: Reg,
        callee: Reg,
        args: &[Reg],
    ) {
        // Flush first — the shim reads the callee Val and arg slots out of
        // the window. `code.ip = next; *op_ip = off` before the call: a
        // `not_callable` error locates via `op_ip` exactly like the
        // interpreter's decoder-advanced arm.
        self.flush_seq();
        self.mark_op(off, next);
        self.settle_seq();
        let ap = self.reg_list_slot(args);
        let nargs = args.len();
        let (regs, c, d, n) = (
            self.regs(),
            self.iconst(callee.index() as i64),
            self.iconst(dst.index() as i64),
            self.iconst(nargs as i64),
        );
        let rp = self
            .hcall(
                H::CallDyn,
                &[
                    self.env.thread,
                    self.env.code,
                    self.env.chunks,
                    self.env.sigs,
                    regs,
                    c,
                    d,
                    ap,
                    n,
                    self.env.bodies_tbl,
                    self.env.ctx0,
                    self.env.ctx1,
                    self.env.strs,
                    self.env.fuel_p,
                    self.env.opip_p,
                    self.env.out,
                ],
            )
            .unwrap();
        self.post_call(i, dst, rp);
    }

    /// After an inlined callee returns through the megashim: null means
    /// propagate `out` verbatim (`eret`); otherwise the returned pointer is
    /// the caller's rebuilt window — re-pin it, re-arm the quota (the callee
    /// consumed fuel/ops_left through its own `bcn`), refresh dst's shadow,
    /// and resume at the next op.
    fn post_call(&mut self, i: usize, dst: Reg, rp: Value) {
        let z = self.iconst(0);
        let isz = self.fb.ins().icmp(IntCC::Equal, rp, z);
        let resumed = self.fb.create_block();
        self.fb.ins().brif(isz, self.ex.eret, &[], resumed, &[]);
        self.fb.switch_to_block(resumed);
        self.rearm_seq();
        self.fb.def_var(self.v.regs, rp);
        self.dst_refresh(i, dst, rp);
    }

    /// Post-call tail: `regs[dst]` now holds the callee's return value —
    /// refresh its scalar shadow (if any) and resume at the next op.
    fn dst_refresh(&mut self, i: usize, dst: Reg, regs2: Value) {
        let d = dst.index() as u32;
        if let Some(&(sv, ok)) = self.v.int.get(&d) {
            let a = self.vaddr(regs2, d);
            let t = self.ld_tag(a);
            let want = self.tconst(self.lyt.t_int);
            let k = self.fb.ins().icmp(IntCC::Equal, t, want);
            let v = self.fb.ins().load(I64, tf(), a, self.lyt.val_pay as i32);
            let z = self.iconst(0);
            let sv0 = self.fb.ins().select(k, v, z);
            self.fb.def_var(sv, sv0);
            self.fb.def_var(ok, k);
        } else if let Some(&(sv, ok)) = self.v.float.get(&d) {
            let a = self.vaddr(regs2, d);
            let t = self.ld_tag(a);
            let want = self.tconst(self.lyt.t_float);
            let k = self.fb.ins().icmp(IntCC::Equal, t, want);
            let v = self.fb.ins().load(F64, tf(), a, self.lyt.val_pay as i32);
            let z = self.fb.ins().f64const(0.0);
            let sv0 = self.fb.ins().select(k, v, z);
            self.fb.def_var(sv, sv0);
            self.fb.def_var(ok, k);
        }
        {
            let nb = self.next_blk(i);
            self.fb.ins().jump(nb, &[]);
        }
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

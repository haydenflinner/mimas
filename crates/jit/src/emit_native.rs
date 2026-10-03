//! Frameless "native" calling convention for scalar-only bodies.
//!
//! A body is *native-eligible* when every reg's scalar kind is provable
//! statically and no op can observe a frame: no heap access (GC scans the
//! window), no suspension, no host reflection, no estep-reachable ops, and
//! all `CallDirect` targets are themselves eligible (SCC fixpoint — mutual
//! recursion is fine). Eligible bodies get a second compiled function:
//!
//! ```text
//! native_body(env: *const BodyEnv, depth: i64, a0..a_{n-1}: i64)
//!     -> (status: i8, val: i64)
//! ```
//!
//! All args arrive as raw `i64` payloads — every param reg is statically
//! `Int`; callers prove it (native callers have kinds, framed callers
//! tag-check). `val` is the `Int` return payload on `status == 0`.
//!
//! Status `1` propagates `*out` verbatim (RtErr exits); status `2` is the
//! unwind-retry escape: ops_left exhaustion mid-body, the
//! native-depth cap, or an edge case a flat op can't inline — the call
//! made no progress and the *framed* ancestor's call site re-runs it through
//! the ordinary frame path. Whitelisted bodies have no side effects, so
//! discarding partial work is safe. Quota is self-accounting: the callee
//! arms `bcn`/`bcn0` from `ops_left` at entry and settles at every exit,
//! so callers never prepay and accounting stays exact even on early
//! returns. `fuel` (the cooperative batch counter) is left alone — a
//! frameless body can't suspend mid-body anyway, so honoring the batch
//! would only force a doomed `ST_RETRY` every ~1024 ops.

use std::collections::HashMap;

use compile::{AccessKind, BlockTarget, Constant, Op, Program, UnaryOp};
use cranelift_codegen::Context;
use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::stackslot::{StackSlotData, StackSlotKind};
use cranelift_codegen::ir::{
    self, AbiParam, Block, FuncRef, InstBuilder, MemFlagsData, Signature, Value, types,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};
use vm::bc::jit::Layout;

use crate::H;
use crate::emit::{
    ENV_CODE, ENV_MC, ENV_OPIP, ENV_OUT, ENV_ST, ENV_STRS, ENV_THREAD, ERR_MOD0, ERR_OVFW,
    ERR_PANIC,
};

const I64: ir::Type = types::I64;
const I8: ir::Type = types::I8;
const I16: ir::Type = types::I16;
const I32: ir::Type = types::I32;
const F64: ir::Type = types::F64;

/// VM state region — same index as `emit::tfs`.
fn tfs() -> MemFlagsData {
    MemFlagsData::trusted().with_alias_region(Some(ir::AliasRegion::new(2)))
}

/// Reg-window region — same index as `emit::tfw`.
fn tfw() -> MemFlagsData {
    MemFlagsData::trusted().with_alias_region(Some(ir::AliasRegion::new(0)))
}

/// GC-heap header region — same index as `emit::tfhd`.
fn tfhd() -> MemFlagsData {
    MemFlagsData::trusted().with_alias_region(Some(ir::AliasRegion::new(3)))
}

/// GC-heap element region — same index as `emit::tfel`.
fn tfel() -> MemFlagsData {
    MemFlagsData::trusted().with_alias_region(Some(ir::AliasRegion::new(4)))
}

/// Max native-call nesting — over this we unwind to a framed retry well
/// before the host stack is at risk.
pub const NATIVE_DEPTH: i64 = 512;

/// Status codes on the `(i8, i64)` return.
pub const ST_OK: i64 = 0;
pub const ST_PROP: i64 = 1; // `*out` holds a value to propagate verbatim
pub const ST_RETRY: i64 = 2; // no progress made; retry the call framed

/// Static scalar kind of a flat reg var (`Bool`/`Null` ride `I64`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NK {
    Int,
    Float,
    Bool,
    Null,
}

/// Per-body native emission plan: `kinds[r]` is the proven scalar kind of
/// reg `r`; `checkpoints[i]` marks ops that can't run frameless — the body
/// *suspends before them* (spill proven regs to the window, `code.ip =
/// off`, `*out = Ok(Flow::Next)`, `ST_PROP`) so the framed lane resumes
/// the suffix; `unsafe_cp` means some emitted op writes a reg with no
/// proven kind, so its live value can't be spilled — every checkpoint
/// then degrades to `ST_RETRY`, still correct but unproductive.
pub(crate) struct NativePlan {
    pub kinds: Vec<Option<NK>>,
    /// `conflict[r]`: reg `r` is written by ops producing different kinds.
    /// Such regs (and never-written `kinds == None` regs) are *window-backed*:
    /// reads emit a window tag-check + payload load, writes store tag +
    /// payload straight into the window slot — only legal at `depth == -1`,
    /// so `windowed` bodies gate on it at entry.
    pub conflict: Vec<bool>,
    pub checkpoints: Vec<bool>,
    pub unsafe_cp: bool,
    /// Some emitted op touches a window-backed reg.
    pub windowed: bool,
}

/// `Some(plan)` when `chunks[body]` is worth a frameless body: every op is
/// either flat-safe ([`native_kinds`]' `need()` rules) or a checkpoint,
/// and at least one op is actually emitted. Call-graph eligibility is
/// layered on by [`analyze`].
pub(crate) fn native_kinds(prog: &Program, body: u32) -> Option<NativePlan> {
    let bid = compile::BodyId::from(body);
    let chunk = &prog.chunks[bid];
    if !chunk.captures.is_empty() {
        return None;
    }
    let ops = prog.ops(bid);
    let nregs = chunk.regs as usize;
    // Flow-insensitive kind lattice: each reg's kind is the meet of all its
    // writes; conflicting writes poison the reg (a typed read of it then
    // fails eligibility below).
    let mut kinds: Vec<Option<NK>> = vec![None; nregs];
    let mut conflict = vec![false; nregs];
    for &p in &chunk.params {
        kinds[p.index()] = Some(NK::Int);
    }
    // dst kind rules
    let mut dwrites: Vec<(u32, Option<NK>)> = Vec::new();
    let mut pkinds: Vec<Option<NK>> = vec![None; ops.len()];
    for (i, (_, op)) in ops.iter().enumerate() {
        let w: Option<(u32, Option<NK>)> = match *op {
            Op::Move { dst, src } => Some((dst.index() as u32, kinds[src.index()])),
            Op::LoadConst { dst, ref constant } => match *constant {
                Constant::Int(_) => Some((dst.index() as u32, Some(NK::Int))),
                Constant::Float(_) => Some((dst.index() as u32, Some(NK::Float))),
                Constant::Bool(_) => Some((dst.index() as u32, Some(NK::Bool))),
                Constant::Null => Some((dst.index() as u32, Some(NK::Null))),
                _ => None, // non-scalar const → checkpoint, no write
            },
            Op::AddInt { dst, .. }
            | Op::SubInt { dst, .. }
            | Op::MultInt { dst, .. }
            | Op::ModInt { dst, .. }
            | Op::AddIntImm { dst, .. }
            | Op::SubIntImm { dst, .. }
            | Op::MultIntImm { dst, .. }
            | Op::ModIntImm { dst, .. }
            | Op::CallDirect { dst, .. }
            | Op::Len { dst, .. }
            | Op::ForNext { idx: dst, .. } => Some((dst.index() as u32, Some(NK::Int))),
            Op::IntLt { dst, .. }
            | Op::IntLe { dst, .. }
            | Op::IntGt { dst, .. }
            | Op::IntGe { dst, .. }
            | Op::IntEq { dst, .. }
            | Op::IntNe { dst, .. }
            | Op::IntLtImm { dst, .. }
            | Op::IntLeImm { dst, .. }
            | Op::IntGtImm { dst, .. }
            | Op::IntGeImm { dst, .. }
            | Op::IntEqImm { dst, .. }
            | Op::IntNeImm { dst, .. }
            | Op::FloatLt { dst, .. }
            | Op::FloatLe { dst, .. }
            | Op::FloatGt { dst, .. }
            | Op::FloatGe { dst, .. }
            | Op::FloatEq { dst, .. }
            | Op::FloatNe { dst, .. }
            | Op::FloatLtImm { dst, .. }
            | Op::FloatLeImm { dst, .. }
            | Op::FloatGtImm { dst, .. }
            | Op::FloatGeImm { dst, .. }
            | Op::FloatEqImm { dst, .. }
            | Op::FloatNeImm { dst, .. }
            | Op::BoolEq { dst, .. }
            | Op::BoolNe { dst, .. } => Some((dst.index() as u32, Some(NK::Bool))),
            Op::AddFloat { dst, .. }
            | Op::SubFloat { dst, .. }
            | Op::MultFloat { dst, .. }
            | Op::DivFloat { dst, .. }
            | Op::AddFloatImm { dst, .. }
            | Op::SubFloatImm { dst, .. }
            | Op::MultFloatImm { dst, .. }
            | Op::ModFloatImm { dst, .. }
            | Op::Sqrt { dst, .. }
            | Op::ToFloat { dst, .. } => Some((dst.index() as u32, Some(NK::Float))),
            Op::Jump { .. }
            | Op::JumpIf { .. }
            | Op::BIntLt { .. }
            | Op::BIntLe { .. }
            | Op::BIntGt { .. }
            | Op::BIntGe { .. }
            | Op::BIntEq { .. }
            | Op::BIntNe { .. }
            | Op::BIntLtImm { .. }
            | Op::BIntLeImm { .. }
            | Op::BIntGtImm { .. }
            | Op::BIntGeImm { .. }
            | Op::BIntEqImm { .. }
            | Op::BIntNeImm { .. }
            | Op::BFloatLt { .. }
            | Op::BFloatLe { .. }
            | Op::BFloatGt { .. }
            | Op::BFloatGe { .. }
            | Op::BFloatEq { .. }
            | Op::BFloatNe { .. }
            | Op::BFloatLtImm { .. }
            | Op::BFloatLeImm { .. }
            | Op::BFloatGtImm { .. }
            | Op::BFloatGeImm { .. }
            | Op::BFloatEqImm { .. }
            | Op::BFloatNeImm { .. }
            | Op::Return { .. }
            | Op::Panic {} => None,
            _ => None, // checkpoint op — runs framed, writes nothing here
        };
        if let Some((_, k)) = w {
            pkinds[i] = k;
        }
        if let Some(w) = w {
            dwrites.push(w);
        }
    }
    // Fixpoint until Move edges propagate everywhere.
    loop {
        let mut edges: Vec<(u32, NK)> = dwrites
            .iter()
            .filter_map(|&(r, k)| k.map(|k| (r, k)))
            .collect();
        for (_, op) in &ops {
            if let Op::Move { dst, src } = *op {
                if let Some(k) = kinds[src.index()] {
                    edges.push((dst.index() as u32, k));
                }
            }
        }
        let mut changed = false;
        for (r, k) in edges {
            if conflict[r as usize] {
                continue;
            }
            match kinds[r as usize] {
                None => {
                    kinds[r as usize] = Some(k);
                    changed = true;
                }
                Some(x) if x != k => {
                    conflict[r as usize] = true;
                    changed = true;
                }
                _ => {}
            }
        }
        if !changed {
            break;
        }
    }
    // Helper-emitted ops whose dst is written straight into the window by
    // the extern call: the produced `Val`'s kind is dynamic, so the reg
    // must be window-backed — `conflict` it. (`Len` is exempt: its helper
    // returns an `i64` we `dv` ourselves.)
    for (_, op) in ops.iter() {
        let writes_dst = matches!(
            *op,
            Op::GetIndex { .. }
                | Op::GetField { .. }
                | Op::Bin { .. }
                | Op::Unary { .. }
                | Op::NewArray { .. }
                | Op::NewDict { .. }
                | Op::NewInstance { .. }
                | Op::NewClosure { .. }
                | Op::CallNative { .. }
                | Op::In { .. }
                | Op::IsInstance { .. }
                | Op::IsRaised { .. }
                | Op::Unwrap { .. }
                | Op::UnwrapRaised { .. }
                | Op::UnwrapUnit { .. }
                | Op::LoadEntry { .. }
                | Op::LoadBody { .. }
        ) || matches!(op, Op::LoadConst { constant: Constant::Str(_), .. });
        if writes_dst {
            if let Some(d) = op.reg() {
                conflict[d.index()] = true;
            }
        }
    }
    // Operand requirements — every typed read must see the right kind.
    // A window-backed reg (conflicted or unproven) satisfies any need: the
    // read emits a runtime tag-check that suspends on a miss, so the op
    // stays emittable instead of becoming a checkpoint.
    let used_win = std::cell::Cell::new(false);
    let need = |r: compile::Reg, k: NK| {
        let i = r.index();
        if conflict[i] || kinds[i].is_none() {
            used_win.set(true);
            return true;
        }
        kinds[i] == Some(k)
    };
    let int2 = |l: compile::Reg, r: compile::Reg| need(l, NK::Int) && need(r, NK::Int);
    let flt2 = |l: compile::Reg, r: compile::Reg| need(l, NK::Float) && need(r, NK::Float);
    let mut checkpoints = vec![false; ops.len()];
    let mut emitted = 0usize;
    let mut windowed = false;
    for (i, (_, op)) in ops.iter().enumerate() {
        used_win.set(false);
        let ok = match *op {
            Op::Move { .. } => true,
            Op::Jump { .. } | Op::Panic {} => true,
            Op::LoadConst { ref constant, .. } => {
                if matches!(*constant, Constant::Str(_)) {
                    used_win.set(true);
                }
                matches!(
                    *constant,
                    Constant::Int(_)
                        | Constant::Float(_)
                        | Constant::Bool(_)
                        | Constant::Null
                        | Constant::Str(_)
                )
            }
            Op::AddInt { left, right, .. }
            | Op::SubInt { left, right, .. }
            | Op::MultInt { left, right, .. }
            | Op::ModInt { left, right, .. }
            | Op::IntLt { left, right, .. }
            | Op::IntLe { left, right, .. }
            | Op::IntGt { left, right, .. }
            | Op::IntGe { left, right, .. }
            | Op::IntEq { left, right, .. }
            | Op::IntNe { left, right, .. }
            | Op::BIntLt { left, right, .. }
            | Op::BIntLe { left, right, .. }
            | Op::BIntGt { left, right, .. }
            | Op::BIntGe { left, right, .. }
            | Op::BIntEq { left, right, .. }
            | Op::BIntNe { left, right, .. } => int2(left, right),
            Op::AddIntImm { left, .. }
            | Op::SubIntImm { left, .. }
            | Op::MultIntImm { left, .. }
            | Op::ModIntImm { left, .. }
            | Op::IntLtImm { left, .. }
            | Op::IntLeImm { left, .. }
            | Op::IntGtImm { left, .. }
            | Op::IntGeImm { left, .. }
            | Op::IntEqImm { left, .. }
            | Op::IntNeImm { left, .. }
            | Op::BIntLtImm { left, .. }
            | Op::BIntLeImm { left, .. }
            | Op::BIntGtImm { left, .. }
            | Op::BIntGeImm { left, .. }
            | Op::BIntEqImm { left, .. }
            | Op::BIntNeImm { left, .. } => need(left, NK::Int),
            Op::ForNext { idx, bound, .. } => int2(idx, bound),
            Op::AddFloat { left, right, .. }
            | Op::SubFloat { left, right, .. }
            | Op::MultFloat { left, right, .. }
            | Op::DivFloat { left, right, .. }
            | Op::FloatLt { left, right, .. }
            | Op::FloatLe { left, right, .. }
            | Op::FloatGt { left, right, .. }
            | Op::FloatGe { left, right, .. }
            | Op::FloatEq { left, right, .. }
            | Op::FloatNe { left, right, .. }
            | Op::BFloatLt { left, right, .. }
            | Op::BFloatLe { left, right, .. }
            | Op::BFloatGt { left, right, .. }
            | Op::BFloatGe { left, right, .. } => flt2(left, right),
            Op::AddFloatImm { left, .. }
            | Op::SubFloatImm { left, .. }
            | Op::MultFloatImm { left, .. }
            | Op::ModFloatImm { left, .. }
            | Op::FloatLtImm { left, .. }
            | Op::FloatLeImm { left, .. }
            | Op::FloatGtImm { left, .. }
            | Op::FloatGeImm { left, .. }
            | Op::FloatEqImm { left, .. }
            | Op::FloatNeImm { left, .. }
            | Op::BFloatLtImm { left, .. }
            | Op::BFloatLeImm { left, .. }
            | Op::BFloatGtImm { left, .. }
            | Op::BFloatGeImm { left, .. }
            | Op::BFloatEqImm { left, .. }
            | Op::BFloatNeImm { left, .. } => need(left, NK::Float),
            Op::BoolEq { left, right, .. } | Op::BoolNe { left, right, .. } => {
                need(left, NK::Bool) && need(right, NK::Bool)
            }
            Op::JumpIf { cond, .. } => need(cond, NK::Bool),
            Op::ToFloat { src, .. } => need(src, NK::Int),
            Op::Sqrt { src, .. } => need(src, NK::Float),
            Op::CallDirect { ref args, .. } => args.iter().all(|a| need(*a, NK::Int)),
            Op::Return { val } => need(val, NK::Int),
            // Heap/state ops → verbatim extern-helper calls over the
            // register window: operands are flushed first, the helper
            // reads/writes window slots directly. Always emittable; they
            // make the body `windowed` (helpers need `regs0`, which only
            // a frame-owning activation has).
            Op::GetIndex { .. }
            | Op::GetField { .. }
            | Op::SetIndex { .. }
            | Op::SetField { .. }
            | Op::Len { .. }
            | Op::Bin { .. }
            | Op::Unary { .. }
            | Op::Push { .. }
            | Op::Insert { .. }
            | Op::NewArray { .. }
            | Op::NewDict { .. }
            | Op::NewInstance { .. }
            | Op::NewClosure { .. }
            | Op::CallNative { .. }
            | Op::In { .. }
            | Op::IsInstance { .. }
            | Op::IsRaised { .. }
            | Op::Unwrap { .. }
            | Op::UnwrapRaised { .. }
            | Op::UnwrapUnit { .. }
            | Op::Raise { .. }
            | Op::LoadEntry { .. }
            | Op::StoreEntry { .. }
            | Op::LoadBody { .. } => {
                used_win.set(true);
                true
            }
            _ => false,
        };
        // A var-backed dst can only take this op's produced kind (a `dv`
        // of mistyped bits would corrupt it); a window-backed dst accepts
        // anything — the write stores tag+payload into the slot. For
        // `Move`, both sides var-backed means their static kinds must
        // match; a window-backed side is checked (or copied raw) at
        // runtime instead.
        let dst_ok = if ok {
            match (op.reg(), op) {
                (Some(d), Op::Move { src, .. }) => {
                    let (d, s) = (d.index(), src.index());
                    let d_var = kinds[d].is_some() && !conflict[d];
                    let s_var = kinds[s].is_some() && !conflict[s];
                    if !d_var || !s_var {
                        used_win.set(true);
                    }
                    !(d_var && s_var && kinds[d] != kinds[s])
                }
                (Some(d), _) => {
                    let d = d.index();
                    if conflict[d] || kinds[d].is_none() {
                        used_win.set(true);
                        true
                    } else {
                        pkinds[i].is_none_or(|k| kinds[d] == Some(k))
                    }
                }
                (None, _) => true,
            }
        } else {
            false
        };
        if ok && dst_ok {
            emitted += 1;
            windowed |= used_win.get();
        } else {
            checkpoints[i] = true;
        }
    }
    if emitted == 0 {
        return None;
    }
    // A checkpoint spills proven-kind regs marked written in the `wrote`
    // mask; `unsafe_cp` degrades every checkpoint to `ST_RETRY` when the
    // mask can't cover the body's regs.
    let unsafe_cp = checkpoints.iter().any(|&c| c) && nregs > 64;
    if windowed && nregs > 64 {
        // Window writes are visible progress — a `ST_RETRY` would re-run
        // the body framed on polluted regs — so every unwind must suspend,
        // and the `wrote` mask can't cover > 64 regs.
        return None;
    }
    Some(NativePlan {
        kinds,
        conflict,
        checkpoints,
        unsafe_cp,
        windowed,
    })
}

/// Per-body plans. A `CallDirect` whose target isn't eligible degrades to
/// a checkpoint (the framed lane runs the call instead) — eligibility
/// itself is just `native_kinds`, iterated to a fixpoint since degrading
/// can turn a body into all-checkpoints (`None`).
pub(crate) fn analyze(prog: &Program) -> Vec<Option<NativePlan>> {
    let n = prog.chunks.len();
    let mut plans: Vec<Option<NativePlan>> =
        (0..n).map(|b| native_kinds(prog, b as u32)).collect();
    loop {
        let elig: Vec<bool> = plans.iter().map(|p| p.is_some()).collect();
        let mut changed = false;
        for b in 0..n {
            let Some(plan) = &mut plans[b] else {
                continue;
            };
            for (i, (_, op)) in prog
                .ops(compile::BodyId::from(b as u32))
                .iter()
                .enumerate()
            {
                if let Op::CallDirect { body: t, .. } = op {
                    if !elig[t.index()] && !plan.checkpoints[i] {
                        plan.checkpoints[i] = true;
                        changed = true;
                    }
                }
            }
            if plan.checkpoints.iter().all(|&c| c) {
                plans[b] = None;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    plans
}

/// The `(env, depth, a0..) -> (i8, i64)` signature for a `nargs`-arg body.
pub(crate) fn native_sig(module: &mut JITModule, nargs: usize) -> Signature {
    let mut s = module.make_signature();
    s.params.push(AbiParam::new(I64)); // env
    s.params.push(AbiParam::new(I64)); // depth
    for _ in 0..nargs {
        s.params.push(AbiParam::new(I64));
    }
    s.returns.push(AbiParam::new(I8));
    s.returns.push(AbiParam::new(I64));
    s
}

struct Ne<'a, 'b> {
    fb: FunctionBuilder<'a>,
    hrefs: &'b [FuncRef],
    nrefs: &'b HashMap<u32, FuncRef>,
    vars: Vec<Variable>,
    blocks: Vec<Block>,
    off2idx: HashMap<usize, usize>,
    ops: Vec<(usize, Op)>,
    /// `run_len[i] > 0` at run heads (op 0 + branch targets): ops until the
    /// next head — the head-gate charge bound. `run_end[i]` = index just
    /// past i's run — the post-call re-arm bound.
    run_len: Vec<usize>,
    run_end: Vec<usize>,
    envp: Value,
    depth: Value,
    opsleft_off: i64,
    lyt: &'b Layout,
    kinds: &'b [Option<NK>],
    conflict: &'b [bool],
    checkpoints: &'b [bool],
    unsafe_cp: bool,
    /// Base of this frame's register window — only `Some` when the plan is
    /// `windowed` (an emitted op touches a window-backed reg). Valid for
    /// the whole activation: native callees never push a frame, so `regs`
    /// can't move.
    regs0: Option<Variable>,
    /// Deferred tag-miss suspend blocks, one per op index — filled with
    /// [`suspend_at`](Self::suspend_at) after the op loop.
    cpb: Vec<Option<Block>>,
    /// Window-backed regs in play: every mid-body unwind must *suspend*
    /// (state preserved), never `ST_RETRY` — window writes already
    /// happened, and a framed re-run would read them as stale inputs.
    windowed: bool,
    /// Set when the body has checkpoints: bit `r` = "reg r was written
    /// natively on this dynamic path" — the checkpoint spills proven regs
    /// whose bit is set; unwritten regs keep their window value (arg or
    /// `Null`), matching semantics exactly.
    wrote: Option<Variable>,
    bcn: Variable,
    bcn0: Variable,
    etrip: Block,
    errs: Vec<(Block, i64)>,
}

impl<'a, 'b> Ne<'a, 'b> {
    fn iconst(&mut self, v: i64) -> Value {
        self.fb.ins().iconst(I64, v)
    }

    fn el(&mut self, off: i32) -> Value {
        let ev = self.envp;
        self.fb.ins().load(I64, tfs(), ev, off)
    }

    fn out_p(&mut self) -> Value {
        self.el(ENV_OUT)
    }

    fn opsleft_p(&mut self) -> Value {
        let t = self.el(ENV_THREAD);
        self.fb.ins().iadd_imm_s(t, self.opsleft_off)
    }

    /// Write the batched `bcn` spend back to `ops_left`. `fuel` (the
    /// cooperative `run_dispatch` batch counter) is *not* charged — a
    /// frameless body can't suspend mid-body, so a fuel-bounded arm would
    /// just `ST_RETRY` every ~1024 ops and discard the whole attempt;
    /// eligible bodies can't allocate either, so the GC-debt and pause
    /// bookkeeping a batch boundary exists for has nothing to do while
    /// one runs. The spend is *subtracted* (not stored) so a propagating
    /// callee's earlier settle isn't clobbered.
    fn settle(&mut self) {
        let b0 = self.fb.use_var(self.bcn0);
        let b = self.fb.use_var(self.bcn);
        let spent = self.fb.ins().isub(b0, b);
        self.fb.def_var(self.bcn0, b);
        let op = self.opsleft_p();
        let ol = self.fb.ins().load(I64, tfs(), op, 0);
        let ol2 = self.fb.ins().isub(ol, spent);
        self.fb.ins().store(tfs(), ol2, op, 0);
    }

    /// `bcn = bcn0 = *ops_left` — at entry and after each call.
    fn arm(&mut self) {
        let op = self.opsleft_p();
        let ol = self.fb.ins().load(I64, tfs(), op, 0);
        self.fb.def_var(self.bcn, ol);
        self.fb.def_var(self.bcn0, ol);
    }

    /// Run-granular quota gate: `bcn < l` at a run head or re-arm point →
    /// unwind-retry. Ops inside a run emit no check — the head guarantees
    /// `bcn >= l` so the per-op [`dec`](Self::dec) chain can't underflow
    /// mid-run, and skipped ops simply never decrement, keeping accounting
    /// exact with no refunds. Tripping is conservative: a thin `bcn` might
    /// have covered a shorter dynamic path, but `ST_RETRY` just hands the
    /// call to the framed machinery which prices the thin tail precisely.
    fn head_gate(&mut self, i: usize, l: i64) {
        let cur = self.fb.use_var(self.bcn);
        let lv = self.iconst(l);
        let short = self.fb.ins().icmp(IntCC::UnsignedLessThan, cur, lv);
        let cont = self.fb.create_block();
        let e = self.bail(i);
        self.fb.ins().brif(short, e, &[], cont, &[]);
        self.fb.switch_to_block(cont);
    }

    /// One op's quota charge — no check; [`head_gate`](Self::head_gate)
    /// guarantees headroom for the whole run.
    fn dec(&mut self) {
        let cur = self.fb.use_var(self.bcn);
        let b1 = self.fb.ins().iadd_imm_s(cur, -1);
        self.fb.def_var(self.bcn, b1);
    }

    /// Window-backed reg: conflicted or never natively written — its
    /// authoritative `Val` lives in the frame's window slot.
    fn wb(&self, r: usize) -> bool {
        self.conflict[r] || self.kinds[r].is_none()
    }

    /// Unwind target for op `i`: a suspend block in windowed bodies (a
    /// retry would re-run on polluted window regs), else `etrip`.
    fn bail(&mut self, i: usize) -> Block {
        if self.windowed {
            self.cp_block(i)
        } else {
            self.etrip
        }
    }

    /// The op `i` suspend block for a runtime tag-miss — created lazily,
    /// filled after the op loop.
    fn cp_block(&mut self, i: usize) -> Block {
        if let Some(b) = self.cpb[i] {
            return b;
        }
        let b = self.fb.create_block();
        self.fb.set_cold_block(b);
        self.cpb[i] = Some(b);
        b
    }

    fn tty(&self) -> ir::Type {
        match self.lyt.tag_size {
            1 => I8,
            2 => I16,
            4 => I32,
            _ => I64,
        }
    }

    fn ktag(&self, k: NK) -> u64 {
        match k {
            NK::Int => self.lyt.t_int,
            NK::Float => self.lyt.t_float,
            NK::Bool => self.lyt.t_bool,
            NK::Null => self.lyt.t_null,
        }
    }

    /// Read reg `r`'s payload. Var-backed regs come straight from the var;
    /// window-backed regs load the slot's tag, check it against `k` (miss →
    /// suspend at op `i`, where the framed lane reads the real `Val`), then
    /// load the payload — `F64` for floats, `u8→i64` for bools, `i64` else.
    fn rv(&mut self, i: usize, r: compile::Reg, k: NK) -> Value {
        if !self.wb(r.index()) {
            return self.fb.use_var(self.vars[r.index()]);
        }
        let regs0 = self.fb.use_var(self.regs0.expect("windowed plan"));
        let roff = (r.index() as i64) * self.lyt.val_size as i64;
        let a = self.fb.ins().iadd_imm_s(regs0, roff);
        let (tty, ktag) = (self.tty(), self.ktag(k));
        let tag = self.fb.ins().load(tty, tfw(), a, self.lyt.val_tag as i32);
        let want = self.fb.ins().iconst(tty, ktag as i64);
        let hit = self.fb.ins().icmp(IntCC::Equal, tag, want);
        let okb = self.fb.create_block();
        let cpb = self.cp_block(i);
        self.fb.ins().brif(hit, okb, &[], cpb, &[]);
        self.fb.switch_to_block(okb);
        match k {
            NK::Float => self.fb.ins().load(F64, tfw(), a, self.lyt.val_pay as i32),
            NK::Bool => {
                let b = self.fb.ins().load(I8, tfw(), a, self.lyt.bool_pay as i32);
                self.fb.ins().uextend(I64, b)
            }
            _ => self.fb.ins().load(I64, tfw(), a, self.lyt.val_pay as i32),
        }
    }

    /// Write `v` (kind `k`) to reg `r` — the var when var-backed, or a
    /// `tag + payload` store into the window slot when window-backed.
    fn dv(&mut self, r: compile::Reg, v: Value, k: NK) {
        if self.wb(r.index()) {
            let regs0 = self.fb.use_var(self.regs0.expect("windowed plan"));
            let roff = (r.index() as i64) * self.lyt.val_size as i64;
            let a = self.fb.ins().iadd_imm_s(regs0, roff);
            let (tty, ktag) = (self.tty(), self.ktag(k));
            let tv = self.fb.ins().iconst(tty, ktag as i64);
            self.fb.ins().store(tfw(), tv, a, self.lyt.val_tag as i32);
            match k {
                NK::Null => {}
                NK::Float => {
                    self.fb.ins().store(tfw(), v, a, self.lyt.val_pay as i32);
                }
                NK::Bool => {
                    let v8 = self.fb.ins().ireduce(I8, v);
                    self.fb.ins().store(tfw(), v8, a, self.lyt.bool_pay as i32);
                }
                _ => {
                    self.fb.ins().store(tfw(), v, a, self.lyt.val_pay as i32);
                }
            }
            return;
        }
        self.fb.def_var(self.vars[r.index()], v);
        if let Some(wv) = self.wrote {
            let w = self.fb.use_var(wv);
            let w2 = self
                .fb
                .ins()
                .bor_imm_u(w, (1u64 << r.index()) as i64);
            self.fb.def_var(wv, w2);
        }
    }

    fn tgt(&self, t: &BlockTarget) -> Block {
        let BlockTarget::ByteOffset(o) = t else {
            panic!("unresolved BlockTarget")
        };
        self.blocks[self.off2idx[o]]
    }

    fn next(&self, i: usize) -> Block {
        if i + 1 < self.ops.len() {
            self.blocks[i + 1]
        } else {
            self.etrip
        }
    }

    /// Cold trampoline: settle + `*out = Err(kind)` + return `ST_PROP`.
    fn err_tramp(&mut self, kind: i64) -> Block {
        let t = self.fb.create_block();
        self.fb.set_cold_block(t);
        self.errs.push((t, kind));
        t
    }

    /// `brif` on an int condition toward `target`/`next` per `is_true`.
    fn bri(&mut self, i: usize, hit: Value, target: &BlockTarget, is_true: bool) {
        let (tb, nb) = (self.tgt(target), self.next(i));
        let (t, f) = if is_true { (tb, nb) } else { (nb, tb) };
        self.fb.ins().brif(hit, t, &[], f, &[]);
    }

    /// Two-operand int compare op → `Bool` var (i64 0/1).
    fn eval_i(&mut self, i: usize, dst: compile::Reg, l: compile::Reg, r: compile::Reg, cc: IntCC) {
        let a = self.rv(i, l, NK::Int);
        let b = self.rv(i, r, NK::Int);
        let c = self.fb.ins().icmp(cc, a, b);
        let c64 = self.fb.ins().uextend(I64, c);
        self.dv(dst, c64, NK::Bool);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    fn eval_i_imm(&mut self, i: usize, dst: compile::Reg, l: compile::Reg, v: i64, cc: IntCC) {
        let a = self.rv(i, l, NK::Int);
        let b = self.iconst(v);
        let c = self.fb.ins().icmp(cc, a, b);
        let c64 = self.fb.ins().uextend(I64, c);
        self.dv(dst, c64, NK::Bool);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    fn eval_f(&mut self, i: usize, dst: compile::Reg, l: compile::Reg, r: compile::Reg, cc: FloatCC) {
        let a = self.rv(i, l, NK::Float);
        let b = self.rv(i, r, NK::Float);
        let c = self.fb.ins().fcmp(cc, a, b);
        let c64 = self.fb.ins().uextend(I64, c);
        self.dv(dst, c64, NK::Bool);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    fn eval_f_imm(&mut self, i: usize, dst: compile::Reg, l: compile::Reg, v: i64, cc: FloatCC) {
        let a = self.rv(i, l, NK::Float);
        let b = self.fb.ins().f64const(f64::from_bits(v as u64));
        let c = self.fb.ins().fcmp(cc, a, b);
        let c64 = self.fb.ins().uextend(I64, c);
        self.dv(dst, c64, NK::Bool);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    /// `checked_add/sub` — overflow → `ERR_OVFW`.
    fn checked(&mut self, i: usize, dst: compile::Reg, a: Value, b: Value, sub: bool) {
        let (v, of) = if sub {
            self.fb.ins().ssub_overflow(a, b)
        } else {
            self.fb.ins().sadd_overflow(a, b)
        };
        let okb = self.fb.create_block();
        let t = self.err_tramp(ERR_OVFW);
        self.fb.ins().brif(of, t, &[], okb, &[]);
        self.fb.switch_to_block(okb);
        self.dv(dst, v, NK::Int);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }


    /// `srem` — `b == 0` → `ERR_MOD0`; `MIN % -1` panics in the interpreter
    /// so unwind to the framed path which reproduces the crash identically.
    fn modint(&mut self, i: usize, dst: compile::Reg, a: Value, b: Value) {
        let z = self.iconst(0);
        let bz = self.fb.ins().icmp(IntCC::Equal, b, z);
        let run = self.fb.create_block();
        let tramp = self.err_tramp(ERR_MOD0);
        self.fb.ins().brif(bz, tramp, &[], run, &[]);
        self.fb.switch_to_block(run);
        let c_min = self.iconst(i64::MIN);
        let c_m1 = self.iconst(-1);
        let amin = self.fb.ins().icmp(IntCC::Equal, a, c_min);
        let bm1 = self.fb.ins().icmp(IntCC::Equal, b, c_m1);
        let bad = self.fb.ins().band(amin, bm1);
        let run2 = self.fb.create_block();
        let bail = self.bail(i);
        self.fb.ins().brif(bad, bail, &[], run2, &[]);
        self.fb.switch_to_block(run2);
        let v = self.fb.ins().srem(a, b);
        self.dv(dst, v, NK::Int);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    /// `CallDirect` to an eligible callee — depth cap → unwind; status 1
    /// or 2 propagates verbatim (settle first); 0 writes `dst`. `paused`
    /// needs no check here: it can only be set by a host native, which no
    /// eligible body can reach — a pending flag is observed at the driver
    /// loop or the framed ancestor's next `gatep`, coarser but sound.
    fn ncall(&mut self, i: usize, dst: compile::Reg, tb: u32, args: &[compile::Reg]) {
        let d = self.depth;
        let d2 = self.fb.ins().iadd_imm_s(d, 1);
        let lim = self.iconst(NATIVE_DEPTH);
        let over = self.fb.ins().icmp(IntCC::SignedGreaterThanOrEqual, d2, lim);
        let cont2 = self.fb.create_block();
        let cap = self.bail(i);
        self.fb.ins().brif(over, cap, &[], cont2, &[]);
        self.fb.switch_to_block(cont2);
        let envp = self.envp;
        let mut cargs = Vec::with_capacity(2 + args.len());
        cargs.push(envp);
        cargs.push(d2);
        for a in args {
            cargs.push(self.rv(i, *a, NK::Int));
        }
        let fr = self.nrefs[&tb];
        let inst = self.fb.ins().call(fr, &cargs);
        let st = self.fb.inst_results(inst)[0];
        let v = self.fb.inst_results(inst)[1];
        let okb = self.fb.create_block();
        let pb = self.fb.create_block();
        self.fb.set_cold_block(pb);
        let s0 = self.fb.ins().iconst(I8, ST_OK);
        let is0 = self.fb.ins().icmp(IntCC::Equal, st, s0);
        self.fb.ins().brif(is0, okb, &[], pb, &[]);
        self.fb.switch_to_block(pb);
        // `ST_PROP` propagates verbatim (`*out` already holds the result —
        // the callee's progress is kept). `ST_RETRY` means the callee did
        // nothing: a windowed body can't retry either (its window writes
        // are progress), so it suspends and the framed lane re-runs just
        // this call.
        let s1 = self.fb.ins().iconst(I8, ST_PROP);
        let isprop = self.fb.ins().icmp(IntCC::Equal, st, s1);
        let propb = self.fb.create_block();
        let rb = self.fb.create_block();
        self.fb.ins().brif(isprop, propb, &[], rb, &[]);
        self.fb.switch_to_block(propb);
        self.settle();
        let zz = self.iconst(0);
        self.fb.ins().return_(&[st, zz]);
        self.fb.switch_to_block(rb);
        if self.windowed {
            self.suspend_at(i);
        } else {
            self.settle();
            let zz = self.iconst(0);
            self.fb.ins().return_(&[st, zz]);
        }
        self.fb.switch_to_block(okb);
        self.arm();
        // The re-armed `bcn` must cover the ops left in this run — a thin
        // budget can't underflow the `dec` chain mid-run; unwind so the
        // framed path reproduces the precise exhaustion point.
        let rem = (self.run_end[i] - i - 1) as i64;
        if rem > 0 {
            self.head_gate(i, rem);
        }
        self.dv(dst, v, NK::Int);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    /// Suspend the activation back to the framed lane *before* this op:
    /// settle the quota, spill every proven-kind reg into this frame's
    /// window, point `code.ip` at the op, write `*out = Ok(Flow::Next)`,
    /// return `ST_PROP`. The driver re-dispatches; the shim's
    /// `code.ip != entry` test then sends it to the framed body, which
    /// resumes at exactly this op — so any op can suspend, and only the
    /// suspended op itself costs a framed step.
    ///
    /// Legal only at `depth == -1`, the shim's "this activation owns a
    /// frame" marker — a frameless callee's regs aren't in any window
    /// (`frames.last()` there is the *caller's*), so it unwinds `ST_RETRY`
    /// and the framed caller's `estep` retries the call through the real
    /// frame machinery. `unsafe_cp` bodies can't spill all live state —
    /// their checkpoints always retry.
    fn checkpoint(&mut self, off: usize) {
        let e = self.etrip;
        if self.unsafe_cp {
            self.fb.ins().jump(e, &[]);
            return;
        }
        let d = self.depth;
        let m1 = self.iconst(-1);
        let fresh = self.fb.ins().icmp(IntCC::Equal, d, m1);
        let okb = self.fb.create_block();
        self.fb.ins().brif(fresh, okb, &[], e, &[]);
        self.fb.switch_to_block(okb);
        self.settle();
        // `regs[frames.last().base + r]` — same derivation as the shim.
        let t = self.el(ENV_THREAD);
        let fptr = self
            .fb
            .ins()
            .load(I64, tfs(), t, (self.lyt.frames_off + self.lyt.vec_ptr) as i32);
        let flen = self
            .fb
            .ins()
            .load(I64, tfs(), t, (self.lyt.frames_off + self.lyt.vec_len) as i32);
        let fm1 = self.fb.ins().iadd_imm_s(flen, -1);
        let foff = self.fb.ins().imul_imm_s(fm1, self.lyt.frame_size as i64);
        let faddr = self.fb.ins().iadd(fptr, foff);
        let base = self
            .fb
            .ins()
            .load(I64, tfs(), faddr, self.lyt.frame_base as i32);
        let rp = self
            .fb
            .ins()
            .load(I64, tfs(), t, (self.lyt.regs_off + self.lyt.vec_ptr) as i32);
        let boff = self.fb.ins().imul_imm_s(base, self.lyt.val_size as i64);
        let regs0 = self.fb.ins().iadd(rp, boff);
        let tty = match self.lyt.tag_size {
            1 => I8,
            2 => I16,
            4 => I32,
            _ => I64,
        };
        // Spill only regs written on *this* dynamic path — the `wrote` bit.
        // An unwritten reg's window slot already holds its semantic value
        // (a param's arg, or the `Null` every non-param gets at frame
        // entry); spilling its init-`0` var would corrupt it.
        let wv = self.wrote.expect("checkpointed body has a wrote mask");
        for r in 0..self.kinds.len() {
            if self.conflict[r] {
                continue;
            }
            let Some(k) = self.kinds[r] else {
                continue;
            };
            let w = self.fb.use_var(wv);
            let hit = self.fb.ins().band_imm_u(w, (1u64 << r) as i64);
            let spillb = self.fb.create_block();
            let skipb = self.fb.create_block();
            self.fb.ins().brif(hit, spillb, &[], skipb, &[]);
            self.fb.switch_to_block(spillb);
            let a = self
                .fb
                .ins()
                .iadd_imm_s(regs0, (r as i64) * self.lyt.val_size as i64);
            let tag = match k {
                NK::Int => self.lyt.t_int,
                NK::Float => self.lyt.t_float,
                NK::Bool => self.lyt.t_bool,
                NK::Null => self.lyt.t_null,
            };
            let tv = self.fb.ins().iconst(tty, tag as i64);
            self.fb.ins().store(tfw(), tv, a, self.lyt.val_tag as i32);
            if k != NK::Null {
                let v = self.fb.use_var(self.vars[r]);
                match k {
                    NK::Float => self.fb.ins().store(tfw(), v, a, self.lyt.val_pay as i32),
                    NK::Bool => {
                        let v8 = self.fb.ins().ireduce(I8, v);
                        self.fb.ins().store(tfw(), v8, a, self.lyt.bool_pay as i32)
                    }
                    _ => self.fb.ins().store(tfw(), v, a, self.lyt.val_pay as i32),
                };
            }
            self.fb.ins().jump(skipb, &[]);
            self.fb.switch_to_block(skipb);
        }
        let cp = self.el(ENV_CODE);
        let o = self.iconst(off as i64);
        self.fb
            .ins()
            .store(tfs(), o, cp, self.lyt.code_ip as i32);
        let op = self.out_p();
        self.fb
            .ins()
            .call(self.hrefs[H::OutNext as usize], &[op]);
        let s1 = self.fb.ins().iconst(I8, ST_PROP);
        let zz = self.iconst(0);
        self.fb.ins().return_(&[s1, zz]);
    }

    /// Suspend *before* op `i`: the framed resume runs it and the rest of
    /// its run uncharged — framed ops between heads are "prepaid" by the
    /// head's upfront charge, which never ran here — so a mid-run suspend
    /// first charges `run_end[i] - i` (etrip when it can't fit, exactly
    /// like a run gate). A *head* suspend skips that: the framed resume's
    /// own head-gate charges the whole run.
    fn suspend_at(&mut self, i: usize) {
        if self.run_len[i] == 0 {
            let rem = (self.run_end[i] - i) as i64;
            if rem > 0 {
                if self.windowed {
                    // Can't bail — window writes already happened. Clamp
                    // instead: charge at most `bcn`, flooring `ops_left`
                    // near 0 — the resume's own gates price the tail.
                    let cur = self.fb.use_var(self.bcn);
                    let c = self.iconst(rem);
                    let m = self.fb.ins().umin(cur, c);
                    let n = self.fb.ins().isub(cur, m);
                    self.fb.def_var(self.bcn, n);
                } else {
                    self.head_gate(i, rem);
                    let cur = self.fb.use_var(self.bcn);
                    let n = self.fb.ins().iadd_imm_s(cur, -rem);
                    self.fb.def_var(self.bcn, n);
                }
            }
        }
        self.checkpoint(self.ops[i].0);
    }

    // ---- extern-helper ops -------------------------------------------------
    // Heap/state ops call the same `mj_*` extern fns the framed lane uses.
    // They run over `regs0` (this frame's window — legal only at
    // `depth == -1`, already gated at entry), so every operand reg must be
    // authoritative in its slot: window-backed regs already are, var-backed
    // regs get a `tag + payload` flush. Helper-written dsts are always
    // window-backed per the plan.

    fn iconst8(&mut self, v: i64) -> Value {
        self.fb.ins().iconst(I8, v)
    }

    fn iconst32(&mut self, v: i64) -> Value {
        self.fb.ins().iconst(I32, v)
    }

    /// `(mc, st)` — `Ctx`'s two words.
    fn ctx2(&mut self) -> (Value, Value) {
        (self.el(ENV_MC), self.el(ENV_ST))
    }

    /// `regs` vec base — for entry-frame slots (`LoadEntry`/`StoreEntry`),
    /// which index `regs` absolutely (the entry frame's base is 0).
    fn regs_ptr(&mut self) -> Value {
        let t = self.el(ENV_THREAD);
        self.fb
            .ins()
            .load(I64, tfs(), t, (self.lyt.regs_off + self.lyt.vec_ptr) as i32)
    }

    /// Byte offset of the op following `i` — helpers like `call_native`
    /// expect `code.ip` already past the op.
    fn next_off(&self, i: usize) -> usize {
        self.ops.get(i + 1).map(|(o, _)| *o).unwrap_or(self.ops[i].0)
    }

    /// `code.ip = next; *op_ip = off` — the framed lane's pre-helper idiom,
    /// so error/DebugInfo positions stay correct.
    fn mark_op(&mut self, i: usize) {
        let (off, next) = (self.ops[i].0, self.next_off(i));
        let n = self.iconst(next as i64);
        let o = self.iconst(off as i64);
        let cp = self.el(ENV_CODE);
        self.fb.ins().store(tfs(), n, cp, self.lyt.code_ip as i32);
        let pp = self.el(ENV_OPIP);
        self.fb.ins().store(tfs(), o, pp, 0);
    }

    /// Write a var-backed reg's `Val` into its window slot so a helper
    /// reads it. Window-backed regs are already authoritative; a
    /// var-backed reg is always written before it's read (`kinds[r]` is
    /// `Some` exactly when a producing op exists), so its var holds the
    /// live value.
    fn flush(&mut self, r: compile::Reg) {
        let ri = r.index();
        if self.wb(ri) {
            return;
        }
        let k = self.kinds[ri].expect("var-backed reg kind");
        let regs0 = self.fb.use_var(self.regs0.expect("windowed"));
        let a = self
            .fb
            .ins()
            .iadd_imm_s(regs0, (ri as i64) * self.lyt.val_size as i64);
        let (tty, ktag) = (self.tty(), self.ktag(k));
        let tv = self.fb.ins().iconst(tty, ktag as i64);
        self.fb.ins().store(tfw(), tv, a, self.lyt.val_tag as i32);
        if k == NK::Null {
            return;
        }
        let v = self.fb.use_var(self.vars[ri]);
        match k {
            NK::Float => self.fb.ins().store(tfw(), v, a, self.lyt.val_pay as i32),
            NK::Bool => {
                let v8 = self.fb.ins().ireduce(I8, v);
                self.fb.ins().store(tfw(), v8, a, self.lyt.bool_pay as i32)
            }
            _ => self.fb.ins().store(tfw(), v, a, self.lyt.val_pay as i32),
        };
    }

    fn hcall(&mut self, h: H, args: &[Value]) -> Option<Value> {
        let inst = self.fb.ins().call(self.hrefs[h as usize], args);
        self.fb.inst_results(inst).first().copied()
    }

    /// Flush `ops`'s operand regs, `mark_op`, then call a `u8`-status
    /// helper: nonzero → settle + `ST_PROP` (`*out` already holds the
    /// error — progress stays); zero → leave the builder in the
    /// continuation block.
    fn hstatus(&mut self, i: usize, h: H, args: &[Value], flush: &[compile::Reg]) {
        for &r in flush {
            self.flush(r);
        }
        self.mark_op(i);
        let k = self.hcall(h, args).expect("u8 helper");
        let post = self.fb.create_block();
        let pb = self.fb.create_block();
        self.fb.set_cold_block(pb);
        self.fb.ins().brif(k, pb, &[], post, &[]);
        self.fb.switch_to_block(pb);
        self.settle();
        let s1 = self.fb.ins().iconst(I8, ST_PROP);
        let zz = self.iconst(0);
        self.fb.ins().return_(&[s1, zz]);
        self.fb.switch_to_block(post);
    }

    /// Same but for a `void` helper — always continues.
    fn hvoid(&mut self, i: usize, h: H, args: &[Value], flush: &[compile::Reg]) {
        for &r in flush {
            self.flush(r);
        }
        self.mark_op(i);
        self.hcall(h, args);
    }

    /// `paused` can flip inside helpers that run foreign code
    /// (`CallNative`, `Bin`/`Unary` instance-op impls) — suspend at the
    /// next op when set, the framed `gatep` equivalent. (Every body ends
    /// in `Return`/`Panic`, so `i + 1` always indexes a real op.)
    fn paused_gate(&mut self, i: usize) {
        let c1 = self.el(ENV_ST);
        let pp = self.fb.ins().iadd_imm_s(c1, self.lyt.state_paused as i64);
        let p = self.fb.ins().load(I8, tfs(), pp, 0);
        let okb = self.fb.create_block();
        let cpb = self.cp_block(i + 1);
        self.fb.ins().brif(p, cpb, &[], okb, &[]);
        self.fb.switch_to_block(okb);
    }

    /// A `u32` reg-index list spilled to a stack slot — the helper ABI for
    /// call args / field regs / captures (emit.rs's `reg_list_slot`).
    fn reg_list(&mut self, regs_idx: &[compile::Reg]) -> Value {
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

    /// Window-slot address of reg `r` — `regs0 + r*val_size`. Window-backed
    /// regs are authoritative there; a var-backed reg needs `flush` first.
    fn wslot(&mut self, r: compile::Reg) -> Value {
        let regs0 = self.fb.use_var(self.regs0.expect("windowed"));
        self.fb
            .ins()
            .iadd_imm_s(regs0, (r.index() as i64) * self.lyt.val_size as i64)
    }

    fn ld_tag(&mut self, a: Value) -> Value {
        let ty = self.tty();
        self.fb.ins().load(ty, tfw(), a, self.lyt.val_tag as i32)
    }

    fn ld_tag64(&mut self, a: Value) -> Value {
        let t = self.ld_tag(a);
        if self.lyt.tag_size < 8 {
            self.fb.ins().uextend(I64, t)
        } else {
            t
        }
    }

    fn tconst(&mut self, t: u64) -> Value {
        let ty = self.tty();
        self.fb.ins().iconst(ty, t as i64)
    }

    /// The `ArrayStore` discriminant's CLIF type (see `emit::asty`).
    fn asty(&self) -> ir::Type {
        match self.lyt.as_tsz {
            1 => I8,
            2 => I16,
            4 => I32,
            8 => I64,
            d => unreachable!("bad ArrayStore tag width {d}"),
        }
    }

    fn aconst(&mut self, t: u64) -> Value {
        let ty = self.asty();
        self.fb.ins().iconst(ty, t as i64)
    }

    /// `RefCell` borrow flag check — see `emit::borrow_ok`.
    fn borrow_ok(&mut self, gc: Value, flag_off: usize, mutable: bool) -> Value {
        let flag = self.fb.ins().load(I64, tfhd(), gc, flag_off as i32);
        let z = self.iconst(0);
        let cc = if mutable {
            IntCC::Equal
        } else {
            IntCC::SignedGreaterThanOrEqual
        };
        self.fb.ins().icmp(cc, flag, z)
    }

    /// `st` is `t_intarray`/`t_floatarray` → `seqb`, else `miss`.
    fn is_seq_tag(&mut self, st: Value, seqb: Block, miss: Block) {
        let wi = self.tconst(self.lyt.t_intarray);
        let a = self.fb.ins().icmp(IntCC::Equal, st, wi);
        let wf = self.tconst(self.lyt.t_floatarray);
        let b = self.fb.ins().icmp(IntCC::Equal, st, wf);
        let k = self.fb.ins().bor(a, b);
        self.fb.ins().brif(k, seqb, &[], miss, &[]);
    }

    /// `(len, data, isinl)` of an `Instance`'s `Fields` at `fp`,
    /// `Inline`-only — see `emit::fields_inline`.
    fn fields_inline(&mut self, fp: Value) -> (Value, Value, Value) {
        let fty = match self.lyt.fld_tsz {
            1 => I8,
            2 => I16,
            4 => I32,
            8 => I64,
            d => unreachable!("bad Fields tag width {d}"),
        };
        let ftag = self.fb.ins().load(fty, tfhd(), fp, self.lyt.fld_tag as i32);
        let want = self.fb.ins().iconst(fty, self.lyt.fld_inline as i64);
        let isinl = self.fb.ins().icmp(IntCC::Equal, ftag, want);
        let l8 = self.fb.ins().load(I8, tfhd(), fp, self.lyt.fld_len as i32);
        let len = self.fb.ins().uextend(I64, l8);
        let data = self.fb.ins().iadd_imm_s(fp, self.lyt.fld_data as i64);
        (len, data, isinl)
    }

    /// `tag` (i64) names a non-Gc `Val` variant — see `emit::is_non_gc_tag`.
    fn is_non_gc_tag(&mut self, t64: Value) -> Value {
        let mask: u64 = [
            self.lyt.t_null,
            self.lyt.t_bool,
            self.lyt.t_int,
            self.lyt.t_float,
            self.lyt.t_fn,
        ]
        .iter()
        .map(|t| 1u64 << t)
        .sum();
        let m = self.iconst(mask as i64);
        let one = self.iconst(1);
        let bit = self.fb.ins().ishl(one, t64);
        let hit = self.fb.ins().band(bit, m);
        let inr = self.fb.ins().icmp_imm_u(IntCC::UnsignedLessThan, t64, 64);
        let nz = self.fb.ins().icmp_imm_u(IntCC::NotEqual, hit, 0);
        self.fb.ins().band(nz, inr)
    }

    /// Static check: `r` can supply an `Int` operand inline — wb regs
    /// tag-check at runtime; a var reg only when proven `Int`.
    fn iv_maybe(&self, r: compile::Reg) -> bool {
        self.wb(r.index()) || self.kinds[r.index()] == Some(NK::Int)
    }

    /// `r`'s `i64` payload, tagged-`Int` — wb regs tag-check into `ok`, a
    /// miss landing in `slow` (the helper then handles non-Int operands
    /// verbatim). Var regs must be `Some(Int)`; anything else returns
    /// `None` — caller emits the helper-only path.
    fn iv_or_slow(&mut self, r: compile::Reg, slow: Block) -> Option<Value> {
        if !self.wb(r.index()) {
            return match self.kinds[r.index()] {
                Some(NK::Int) => Some(self.fb.use_var(self.vars[r.index()])),
                _ => None,
            };
        }
        let a = self.wslot(r);
        let tag = self.ld_tag(a);
        let want = self.tconst(self.lyt.t_int);
        let hit = self.fb.ins().icmp(IntCC::Equal, tag, want);
        let okb = self.fb.create_block();
        self.fb.ins().brif(hit, okb, &[], slow, &[]);
        self.fb.switch_to_block(okb);
        Some(self.fb.ins().load(I64, tfw(), a, self.lyt.val_pay as i32))
    }

    /// `*da = data[slot]` — a full `Val` copy under `pre && slot < len`;
    /// a miss lands in `slow`. Native `emit::read_elem` analogue: `da` is
    /// the dst's *window* slot, so the copy lands where readers expect.
    fn n_read_elem(
        &mut self,
        i: usize,
        da: Value,
        slot: Value,
        len: Value,
        data: Value,
        pre: Value,
        slow: Block,
    ) {
        let inb = self.fb.ins().icmp(IntCC::UnsignedLessThan, slot, len);
        let k = self.fb.ins().band(pre, inb);
        let good = self.fb.create_block();
        self.fb.ins().brif(k, good, &[], slow, &[]);
        self.fb.switch_to_block(good);
        let vs = self.lyt.val_size as i64;
        let off = self.fb.ins().imul_imm_s(slot, vs);
        let ea = self.fb.ins().iadd(data, off);
        for w in (0..self.lyt.val_size).step_by(8) {
            let v = self.fb.ins().load(I64, tfel(), ea, w as i32);
            self.fb.ins().store(tfw(), v, da, w as i32);
        }
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    /// `*da = Int|Float(data[slot])` — a raw `i64`/`f64` load boxed into
    /// the dst's window slot under `pre && slot < len`.
    #[allow(clippy::too_many_arguments)]
    fn n_read_prim(
        &mut self,
        i: usize,
        da: Value,
        slot: Value,
        len: Value,
        data: Value,
        pre: Value,
        is_int: bool,
        slow: Block,
    ) {
        let inb = self.fb.ins().icmp(IntCC::UnsignedLessThan, slot, len);
        let k = self.fb.ins().band(pre, inb);
        let good = self.fb.create_block();
        self.fb.ins().brif(k, good, &[], slow, &[]);
        self.fb.switch_to_block(good);
        let off = self.fb.ins().ishl_imm_s(slot, 3);
        let ea = self.fb.ins().iadd(data, off);
        let ty = if is_int { I64 } else { F64 };
        let v = self.fb.ins().load(ty, tfel(), ea, 0);
        let tag = if is_int {
            self.lyt.t_int
        } else {
            self.lyt.t_float
        };
        let tv = self.tconst(tag);
        self.fb.ins().store(tfw(), tv, da, self.lyt.val_tag as i32);
        self.fb.ins().store(tfw(), v, da, self.lyt.val_pay as i32);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    /// `data[slot] = *sa` — a `Val` copy into the element store under
    /// `pre && slot < len`.
    fn n_write_elem(
        &mut self,
        i: usize,
        sa: Value,
        slot: Value,
        len: Value,
        data: Value,
        pre: Value,
        slow: Block,
    ) {
        let inb = self.fb.ins().icmp(IntCC::UnsignedLessThan, slot, len);
        let k = self.fb.ins().band(pre, inb);
        let good = self.fb.create_block();
        self.fb.ins().brif(k, good, &[], slow, &[]);
        self.fb.switch_to_block(good);
        let vs = self.lyt.val_size as i64;
        let off = self.fb.ins().imul_imm_s(slot, vs);
        let ea = self.fb.ins().iadd(data, off);
        for w in (0..self.lyt.val_size).step_by(8) {
            let v = self.fb.ins().load(I64, tfw(), sa, w as i32);
            self.fb.ins().store(tfel(), v, ea, w as i32);
        }
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    /// `data[slot] = v` — a raw `i64`/`f64` store under
    /// `pre && slot < len`.
    fn n_write_prim(
        &mut self,
        i: usize,
        v: Value,
        slot: Value,
        len: Value,
        data: Value,
        pre: Value,
        slow: Block,
    ) {
        let inb = self.fb.ins().icmp(IntCC::UnsignedLessThan, slot, len);
        let k = self.fb.ins().band(pre, inb);
        let good = self.fb.create_block();
        self.fb.ins().brif(k, good, &[], slow, &[]);
        self.fb.switch_to_block(good);
        let off = self.fb.ins().ishl_imm_s(slot, 3);
        let ea = self.fb.ins().iadd(data, off);
        self.fb.ins().store(tfel(), v, ea, 0);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    /// Typed-array `GetIndex` tail — `gc` is the `Seq` payload (see
    /// `emit::seq_read`): `Ints`/`Floats`/`Vals` fan out per store kind.
    fn n_seq_read(&mut self, i: usize, da: Value, slot: Value, gc: Value, slow: Block) {
        let fok = self.borrow_ok(gc, self.lyt.rl_flag_s, false);
        let aty = self.asty();
        let stag = self
            .fb
            .ins()
            .load(aty, tfhd(), gc, (self.lyt.rl_seq + self.lyt.as_tag) as i32);
        let wi = self.aconst(self.lyt.as_ints);
        let isints = self.fb.ins().icmp(IntCC::Equal, stag, wi);
        let intsb = self.fb.create_block();
        let noti = self.fb.create_block();
        self.fb.ins().brif(isints, intsb, &[], noti, &[]);
        self.fb.switch_to_block(intsb);
        {
            let vp = self.fb.ins().load(
                I64,
                tfhd(),
                gc,
                (self.lyt.rl_seq + self.lyt.as_ints_vec + self.lyt.vec_ptr) as i32,
            );
            let vl = self.fb.ins().load(
                I64,
                tfhd(),
                gc,
                (self.lyt.rl_seq + self.lyt.as_ints_vec + self.lyt.vec_len) as i32,
            );
            self.n_read_prim(i, da, slot, vl, vp, fok, true, slow);
        }
        self.fb.switch_to_block(noti);
        let wf = self.aconst(self.lyt.as_floats);
        let isflts = self.fb.ins().icmp(IntCC::Equal, stag, wf);
        let fltsb = self.fb.create_block();
        let notf = self.fb.create_block();
        self.fb.ins().brif(isflts, fltsb, &[], notf, &[]);
        self.fb.switch_to_block(fltsb);
        {
            let vp = self.fb.ins().load(
                I64,
                tfhd(),
                gc,
                (self.lyt.rl_seq + self.lyt.as_floats_vec + self.lyt.vec_ptr) as i32,
            );
            let vl = self.fb.ins().load(
                I64,
                tfhd(),
                gc,
                (self.lyt.rl_seq + self.lyt.as_floats_vec + self.lyt.vec_len) as i32,
            );
            self.n_read_prim(i, da, slot, vl, vp, fok, false, slow);
        }
        self.fb.switch_to_block(notf);
        let wv = self.aconst(self.lyt.as_vals);
        let isvals = self.fb.ins().icmp(IntCC::Equal, stag, wv);
        let valsb = self.fb.create_block();
        self.fb.ins().brif(isvals, valsb, &[], slow, &[]);
        self.fb.switch_to_block(valsb);
        {
            let arr = self
                .fb
                .ins()
                .load(I64, tfhd(), gc, (self.lyt.rl_seq + self.lyt.as_vals_arr) as i32);
            let fok2 = self.borrow_ok(arr, self.lyt.rl_flag, false);
            let pre = self.fb.ins().band(fok, fok2);
            let vp = self
                .fb
                .ins()
                .load(I64, tfhd(), arr, (self.lyt.rl_vec + self.lyt.vec_ptr) as i32);
            let vl = self
                .fb
                .ins()
                .load(I64, tfhd(), arr, (self.lyt.rl_vec + self.lyt.vec_len) as i32);
            self.n_read_elem(i, da, slot, vl, vp, pre, slow);
        }
    }

    /// Typed-array write tail — `Ints`/`Floats` take the matching scalar,
    /// `Vals` takes a whole non-Gc `Val`; everything else → `slow` (see
    /// `emit::seq_write`).
    fn n_seq_write(&mut self, i: usize, slot: Value, gc: Value, va: Value, slow: Block) {
        let fok = self.borrow_ok(gc, self.lyt.rl_flag_s, true);
        let aty = self.asty();
        let stag = self
            .fb
            .ins()
            .load(aty, tfhd(), gc, (self.lyt.rl_seq + self.lyt.as_tag) as i32);
        let vt = self.ld_tag(va);
        let wi = self.aconst(self.lyt.as_ints);
        let isints = self.fb.ins().icmp(IntCC::Equal, stag, wi);
        let intsb = self.fb.create_block();
        let noti = self.fb.create_block();
        self.fb.ins().brif(isints, intsb, &[], noti, &[]);
        self.fb.switch_to_block(intsb);
        {
            let want = self.tconst(self.lyt.t_int);
            let vis = self.fb.ins().icmp(IntCC::Equal, vt, want);
            let pre = self.fb.ins().band(fok, vis);
            let vp = self.fb.ins().load(
                I64,
                tfhd(),
                gc,
                (self.lyt.rl_seq + self.lyt.as_ints_vec + self.lyt.vec_ptr) as i32,
            );
            let vl = self.fb.ins().load(
                I64,
                tfhd(),
                gc,
                (self.lyt.rl_seq + self.lyt.as_ints_vec + self.lyt.vec_len) as i32,
            );
            let xv = self.fb.ins().load(I64, tfw(), va, self.lyt.val_pay as i32);
            self.n_write_prim(i, xv, slot, vl, vp, pre, slow);
        }
        self.fb.switch_to_block(noti);
        let wf = self.aconst(self.lyt.as_floats);
        let isflts = self.fb.ins().icmp(IntCC::Equal, stag, wf);
        let fltsb = self.fb.create_block();
        let notf = self.fb.create_block();
        self.fb.ins().brif(isflts, fltsb, &[], notf, &[]);
        self.fb.switch_to_block(fltsb);
        {
            let want = self.tconst(self.lyt.t_float);
            let vis = self.fb.ins().icmp(IntCC::Equal, vt, want);
            let pre = self.fb.ins().band(fok, vis);
            let vp = self.fb.ins().load(
                I64,
                tfhd(),
                gc,
                (self.lyt.rl_seq + self.lyt.as_floats_vec + self.lyt.vec_ptr) as i32,
            );
            let vl = self.fb.ins().load(
                I64,
                tfhd(),
                gc,
                (self.lyt.rl_seq + self.lyt.as_floats_vec + self.lyt.vec_len) as i32,
            );
            let xv = self.fb.ins().load(F64, tfw(), va, self.lyt.val_pay as i32);
            self.n_write_prim(i, xv, slot, vl, vp, pre, slow);
        }
        self.fb.switch_to_block(notf);
        let wv = self.aconst(self.lyt.as_vals);
        let isvals = self.fb.ins().icmp(IntCC::Equal, stag, wv);
        let valsb = self.fb.create_block();
        self.fb.ins().brif(isvals, valsb, &[], slow, &[]);
        self.fb.switch_to_block(valsb);
        {
            let arr = self
                .fb
                .ins()
                .load(I64, tfhd(), gc, (self.lyt.rl_seq + self.lyt.as_vals_arr) as i32);
            let fok2 = self.borrow_ok(arr, self.lyt.rl_flag, true);
            let vt64 = self.ld_tag64(va);
            let ngc = self.is_non_gc_tag(vt64);
            let pre = self.fb.ins().band(fok, fok2);
            let pre = self.fb.ins().band(pre, ngc);
            let vp = self
                .fb
                .ins()
                .load(I64, tfhd(), arr, (self.lyt.rl_vec + self.lyt.vec_ptr) as i32);
            let vl = self
                .fb
                .ins()
                .load(I64, tfhd(), arr, (self.lyt.rl_vec + self.lyt.vec_len) as i32);
            self.n_write_elem(i, va, slot, vl, vp, pre, slow);
        }
    }

    fn emit_op(&mut self, i: usize, op: &Op) {
        let op = op.clone();
        self.fb.switch_to_block(self.blocks[i]);
        let l = self.run_len[i];
        if l > 0 {
            self.head_gate(i, l as i64);
        }
        if self.checkpoints[i] {
            self.suspend_at(i);
            return;
        }
        self.dec();
        match op {
            Op::Move { dst, src } => {
                let (d, s) = (dst.index(), src.index());
                match (self.wb(d), self.wb(s)) {
                    (false, false) | (true, false) => {
                        let v = self.fb.use_var(self.vars[s]);
                        self.dv(dst, v, self.kinds[s].unwrap());
                    }
                    (false, true) => {
                        let k = self.kinds[d].unwrap();
                        let v = self.rv(i, src, k);
                        self.dv(dst, v, k);
                    }
                    (true, true) => {
                        // Raw `Val` copy, slot → slot — every payload byte,
                        // not just the union slot: `Bool`/`Fn` payloads sit
                        // outside `val_pay`.
                        let regs0 = self.fb.use_var(self.regs0.expect("windowed"));
                        let vs = self.lyt.val_size as i64;
                        let sa = self.fb.ins().iadd_imm_s(regs0, (s as i64) * vs);
                        let da = self.fb.ins().iadd_imm_s(regs0, (d as i64) * vs);
                        for off in (0..vs).step_by(8) {
                            let w = self.fb.ins().load(I64, tfw(), sa, off as i32);
                            self.fb.ins().store(tfw(), w, da, off as i32);
                        }
                    }
                }
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::LoadConst { dst, constant } => {
                match constant {
                    Constant::Int(v) => {
                        let c = self.iconst(v);
                        self.dv(dst, c, NK::Int);
                    }
                    Constant::Float(v) => {
                        let c = self.fb.ins().f64const(v);
                        self.dv(dst, c, NK::Float);
                    }
                    Constant::Bool(v) => {
                        let c = self.iconst(v as i64);
                        self.dv(dst, c, NK::Bool);
                    }
                    Constant::Null => {
                        let c = self.iconst(0);
                        self.dv(dst, c, NK::Null);
                    }
                    Constant::Str(sid) => {
                        let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                        let (d, s32) = (
                            self.iconst(dst.index() as i64),
                            self.iconst32(sid.index() as i64),
                        );
                        let (c0, c1) = self.ctx2();
                        let st = self.el(ENV_STRS);
                        self.hvoid(i, H::LoadConstStr, &[r0, d, s32, c0, c1, st], &[]);
                    }
                    _ => unreachable!("non-scalar const in native body"),
                }
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::AddInt { dst, left, right } => {
                let (a, b) = (self.rv(i, left, NK::Int), self.rv(i, right, NK::Int));
                self.checked(i, dst, a, b, false);
            }
            Op::SubInt { dst, left, right } => {
                let (a, b) = (self.rv(i, left, NK::Int), self.rv(i, right, NK::Int));
                self.checked(i, dst, a, b, true);
            }
            Op::MultInt { dst, left, right } => {
                let (a, b) = (self.rv(i, left, NK::Int), self.rv(i, right, NK::Int));
                let (v, of) = self.fb.ins().smul_overflow(a, b);
                let okb = self.fb.create_block();
                let t = self.err_tramp(ERR_OVFW);
                self.fb.ins().brif(of, t, &[], okb, &[]);
                self.fb.switch_to_block(okb);
                self.dv(dst, v, NK::Int);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::ModInt { dst, left, right } => {
                let (a, b) = (self.rv(i, left, NK::Int), self.rv(i, right, NK::Int));
                self.modint(i, dst, a, b);
            }
            Op::AddIntImm { dst, left, val } => {
                let (a, b) = (self.rv(i, left, NK::Int), self.iconst(val));
                self.checked(i, dst, a, b, false);
            }
            Op::SubIntImm { dst, left, val } => {
                let (a, b) = (self.rv(i, left, NK::Int), self.iconst(val));
                self.checked(i, dst, a, b, true);
            }
            Op::MultIntImm { dst, left, val } => {
                let (a, b) = (self.rv(i, left, NK::Int), self.iconst(val));
                let (v, of) = self.fb.ins().smul_overflow(a, b);
                let okb = self.fb.create_block();
                let t = self.err_tramp(ERR_OVFW);
                self.fb.ins().brif(of, t, &[], okb, &[]);
                self.fb.switch_to_block(okb);
                self.dv(dst, v, NK::Int);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::ModIntImm { dst, left, val } => {
                let (a, b) = (self.rv(i, left, NK::Int), self.iconst(val));
                self.modint(i, dst, a, b);
            }
            Op::IntLt { dst, left, right } => self.eval_i(i, dst, left, right, IntCC::SignedLessThan),
            Op::IntLe { dst, left, right } => {
                self.eval_i(i, dst, left, right, IntCC::SignedLessThanOrEqual)
            }
            Op::IntGt { dst, left, right } => {
                self.eval_i(i, dst, left, right, IntCC::SignedGreaterThan)
            }
            Op::IntGe { dst, left, right } => {
                self.eval_i(i, dst, left, right, IntCC::SignedGreaterThanOrEqual)
            }
            Op::IntEq { dst, left, right } => self.eval_i(i, dst, left, right, IntCC::Equal),
            Op::IntNe { dst, left, right } => self.eval_i(i, dst, left, right, IntCC::NotEqual),
            Op::IntLtImm { dst, left, val } => {
                self.eval_i_imm(i, dst, left, val, IntCC::SignedLessThan)
            }
            Op::IntLeImm { dst, left, val } => {
                self.eval_i_imm(i, dst, left, val, IntCC::SignedLessThanOrEqual)
            }
            Op::IntGtImm { dst, left, val } => {
                self.eval_i_imm(i, dst, left, val, IntCC::SignedGreaterThan)
            }
            Op::IntGeImm { dst, left, val } => {
                self.eval_i_imm(i, dst, left, val, IntCC::SignedGreaterThanOrEqual)
            }
            Op::IntEqImm { dst, left, val } => self.eval_i_imm(i, dst, left, val, IntCC::Equal),
            Op::IntNeImm { dst, left, val } => self.eval_i_imm(i, dst, left, val, IntCC::NotEqual),
            Op::BoolEq { dst, left, right } | Op::BoolNe { dst, left, right } => {
                let cc = if matches!(op, Op::BoolEq { .. }) {
                    IntCC::Equal
                } else {
                    IntCC::NotEqual
                };
                self.eval_i(i, dst, left, right, cc);
            }
            Op::AddFloat { dst, left, right }
            | Op::SubFloat { dst, left, right }
            | Op::MultFloat { dst, left, right }
            | Op::DivFloat { dst, left, right } => {
                let a = self.rv(i, left, NK::Float);
                let b = self.rv(i, right, NK::Float);
                let v = match op {
                    Op::AddFloat { .. } => self.fb.ins().fadd(a, b),
                    Op::SubFloat { .. } => self.fb.ins().fsub(a, b),
                    Op::MultFloat { .. } => self.fb.ins().fmul(a, b),
                    _ => self.fb.ins().fdiv(a, b),
                };
                self.dv(dst, v, NK::Float);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::AddFloatImm { dst, left, val }
            | Op::SubFloatImm { dst, left, val }
            | Op::MultFloatImm { dst, left, val }
            | Op::ModFloatImm { dst, left, val } => {
                let a = self.rv(i, left, NK::Float);
                let b = self.fb.ins().f64const(f64::from_bits(val as u64));
                let v = match &op {
                    Op::AddFloatImm { .. } => self.fb.ins().fadd(a, b),
                    Op::SubFloatImm { .. } => self.fb.ins().fsub(a, b),
                    Op::MultFloatImm { .. } => self.fb.ins().fmul(a, b),
                    _ => {
                        // ModFloatImm — no frem in clif; unwind to framed
                        let t = self.bail(i);
                        self.fb.ins().jump(t, &[]);
                        return;
                    }
                };
                self.dv(dst, v, NK::Float);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::FloatLt { dst, left, right } => self.eval_f(i, dst, left, right, FloatCC::LessThan),
            Op::FloatLe { dst, left, right } => {
                self.eval_f(i, dst, left, right, FloatCC::LessThanOrEqual)
            }
            Op::FloatGt { dst, left, right } => {
                self.eval_f(i, dst, left, right, FloatCC::GreaterThan)
            }
            Op::FloatGe { dst, left, right } => {
                self.eval_f(i, dst, left, right, FloatCC::GreaterThanOrEqual)
            }
            Op::FloatEq { dst, left, right } => self.eval_f(i, dst, left, right, FloatCC::Equal),
            Op::FloatNe { dst, left, right } => self.eval_f(i, dst, left, right, FloatCC::NotEqual),
            Op::FloatLtImm { dst, left, val } => {
                self.eval_f_imm(i, dst, left, val, FloatCC::LessThan)
            }
            Op::FloatLeImm { dst, left, val } => {
                self.eval_f_imm(i, dst, left, val, FloatCC::LessThanOrEqual)
            }
            Op::FloatGtImm { dst, left, val } => {
                self.eval_f_imm(i, dst, left, val, FloatCC::GreaterThan)
            }
            Op::FloatGeImm { dst, left, val } => {
                self.eval_f_imm(i, dst, left, val, FloatCC::GreaterThanOrEqual)
            }
            Op::FloatEqImm { dst, left, val } => self.eval_f_imm(i, dst, left, val, FloatCC::Equal),
            Op::FloatNeImm { dst, left, val } => {
                self.eval_f_imm(i, dst, left, val, FloatCC::NotEqual)
            }
            Op::BIntLt { ref target, left, right, is_true }
            | Op::BIntLe { ref target, left, right, is_true }
            | Op::BIntGt { ref target, left, right, is_true }
            | Op::BIntGe { ref target, left, right, is_true }
            | Op::BIntEq { ref target, left, right, is_true }
            | Op::BIntNe { ref target, left, right, is_true } => {
                let cc = match &op {
                    Op::BIntLt { .. } => IntCC::SignedLessThan,
                    Op::BIntLe { .. } => IntCC::SignedLessThanOrEqual,
                    Op::BIntGt { .. } => IntCC::SignedGreaterThan,
                    Op::BIntGe { .. } => IntCC::SignedGreaterThanOrEqual,
                    Op::BIntEq { .. } => IntCC::Equal,
                    _ => IntCC::NotEqual,
                };
                let hit = {
                    let a = self.rv(i, left, NK::Int);
                    let b = self.rv(i, right, NK::Int);
                    self.fb.ins().icmp(cc, a, b)
                };
                self.bri(i, hit, target, is_true);
            }
            Op::BIntLtImm { ref target, left, val, is_true }
            | Op::BIntLeImm { ref target, left, val, is_true }
            | Op::BIntGtImm { ref target, left, val, is_true }
            | Op::BIntGeImm { ref target, left, val, is_true }
            | Op::BIntEqImm { ref target, left, val, is_true }
            | Op::BIntNeImm { ref target, left, val, is_true } => {
                let cc = match &op {
                    Op::BIntLtImm { .. } => IntCC::SignedLessThan,
                    Op::BIntLeImm { .. } => IntCC::SignedLessThanOrEqual,
                    Op::BIntGtImm { .. } => IntCC::SignedGreaterThan,
                    Op::BIntGeImm { .. } => IntCC::SignedGreaterThanOrEqual,
                    Op::BIntEqImm { .. } => IntCC::Equal,
                    _ => IntCC::NotEqual,
                };
                let hit = {
                    let a = self.rv(i, left, NK::Int);
                    let b = self.iconst(val);
                    self.fb.ins().icmp(cc, a, b)
                };
                self.bri(i, hit, target, is_true);
            }
            Op::BFloatLt { ref target, left, right, is_true }
            | Op::BFloatLe { ref target, left, right, is_true }
            | Op::BFloatGt { ref target, left, right, is_true }
            | Op::BFloatGe { ref target, left, right, is_true }
            | Op::BFloatEq { ref target, left, right, is_true }
            | Op::BFloatNe { ref target, left, right, is_true } => {
                let cc = match &op {
                    Op::BFloatLt { .. } => FloatCC::LessThan,
                    Op::BFloatLe { .. } => FloatCC::LessThanOrEqual,
                    Op::BFloatGt { .. } => FloatCC::GreaterThan,
                    Op::BFloatGe { .. } => FloatCC::GreaterThanOrEqual,
                    Op::BFloatEq { .. } => FloatCC::Equal,
                    _ => FloatCC::NotEqual,
                };
                let hit = {
                    let a = self.rv(i, left, NK::Float);
                    let b = self.rv(i, right, NK::Float);
                    self.fb.ins().fcmp(cc, a, b)
                };
                self.bri(i, hit, target, is_true);
            }
            Op::BFloatLtImm { ref target, left, val, is_true }
            | Op::BFloatLeImm { ref target, left, val, is_true }
            | Op::BFloatGtImm { ref target, left, val, is_true }
            | Op::BFloatGeImm { ref target, left, val, is_true }
            | Op::BFloatEqImm { ref target, left, val, is_true }
            | Op::BFloatNeImm { ref target, left, val, is_true } => {
                let cc = match &op {
                    Op::BFloatLtImm { .. } => FloatCC::LessThan,
                    Op::BFloatLeImm { .. } => FloatCC::LessThanOrEqual,
                    Op::BFloatGtImm { .. } => FloatCC::GreaterThan,
                    Op::BFloatGeImm { .. } => FloatCC::GreaterThanOrEqual,
                    Op::BFloatEqImm { .. } => FloatCC::Equal,
                    _ => FloatCC::NotEqual,
                };
                let hit = {
                    let a = self.rv(i, left, NK::Float);
                    let b = self.fb.ins().f64const(f64::from_bits(val as u64));
                    self.fb.ins().fcmp(cc, a, b)
                };
                self.bri(i, hit, target, is_true);
            }
            Op::Jump { ref target } => {
                let t = self.tgt(target);
                self.fb.ins().jump(t, &[]);
            }
            Op::JumpIf { cond, ref target, is_true } => {
                let c = self.rv(i, cond, NK::Bool);
                let k = self.iconst(is_true as i64);
                let hit = self.fb.ins().icmp(IntCC::Equal, c, k);
                self.bri(i, hit, target, true);
            }
            Op::ForNext { idx, bound, ref target } => {
                let iv = self.rv(i, idx, NK::Int);
                let bv = self.rv(i, bound, NK::Int);
                let i2 = self.fb.ins().iadd_imm_s(iv, 1);
                self.dv(idx, i2, NK::Int);
                let hit = self.fb.ins().icmp(IntCC::SignedLessThan, i2, bv);
                self.bri(i, hit, target, true);
            }
            Op::ToFloat { dst, src } => {
                let iv = self.rv(i, src, NK::Int);
                let f = self.fb.ins().fcvt_from_sint(F64, iv);
                self.dv(dst, f, NK::Float);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::Sqrt { dst, src } => {
                let f = self.rv(i, src, NK::Float);
                let r = self.fb.ins().sqrt(f);
                self.dv(dst, r, NK::Float);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::CallDirect { dst, body, args } => {
                self.ncall(i, dst, body.index() as u32, &args);
            }
            Op::Return { val } => {
                let v = self.rv(i, val, NK::Int);
                self.settle();
                let s0 = self.fb.ins().iconst(I8, ST_OK);
                self.fb.ins().return_(&[s0, v]);
            }
            Op::Panic {} => {
                let t = self.err_tramp(ERR_PANIC);
                self.fb.ins().jump(t, &[]);
            }
            // ---- extern-helper ops ----
            // The container reads/writes get the framed lane's inline fast
            // paths: `Array`/`Seq` element and `Instance`/`Array` field access
            // inline under tag+borrow+bounds checks; every miss lands in the
            // same `slow` block that runs the verbatim helper.
            Op::GetIndex {
                dst,
                set,
                index,
                kind,
            } => {
                if kind != AccessKind::Direct || !self.iv_maybe(index) {
                    let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                    let (d, s, ii, k8) = (
                        self.iconst(dst.index() as i64),
                        self.iconst(set.index() as i64),
                        self.iconst(index.index() as i64),
                        self.iconst8(kind as i64),
                    );
                    let (c0, c1) = self.ctx2();
                    let op_ = self.out_p();
                    self.hstatus(
                        i,
                        H::GetIndex,
                        &[r0, d, s, ii, k8, c0, c1, op_],
                        &[set, index],
                    );
                    let nb = self.next(i);
                    self.fb.ins().jump(nb, &[]);
                    return;
                }
                let slow = self.fb.create_block();
                self.fb.set_cold_block(slow);
                self.flush(set);
                let Some(iv) = self.iv_or_slow(index, slow) else {
                    unreachable!("iv_maybe")
                };
                let sa = self.wslot(set);
                let st = self.ld_tag(sa);
                let tarr = self.tconst(self.lyt.t_array);
                let isarr = self.fb.ins().icmp(IntCC::Equal, st, tarr);
                let arrb = self.fb.create_block();
                let notarr = self.fb.create_block();
                self.fb.ins().brif(isarr, arrb, &[], notarr, &[]);
                self.fb.switch_to_block(notarr);
                let seqb = self.fb.create_block();
                self.is_seq_tag(st, seqb, slow);
                self.fb.switch_to_block(seqb);
                {
                    let da = self.wslot(dst);
                    let gc = self.fb.ins().load(I64, tfw(), sa, self.lyt.seq_pay as i32);
                    self.n_seq_read(i, da, iv, gc, slow);
                }
                self.fb.switch_to_block(arrb);
                {
                    let da = self.wslot(dst);
                    let gc = self.fb.ins().load(I64, tfw(), sa, self.lyt.arr_pay as i32);
                    let fok = self.borrow_ok(gc, self.lyt.rl_flag, false);
                    let vp = self
                        .fb
                        .ins()
                        .load(I64, tfhd(), gc, (self.lyt.rl_vec + self.lyt.vec_ptr) as i32);
                    let vl = self
                        .fb
                        .ins()
                        .load(I64, tfhd(), gc, (self.lyt.rl_vec + self.lyt.vec_len) as i32);
                    self.n_read_elem(i, da, iv, vl, vp, fok, slow);
                }
                self.fb.switch_to_block(slow);
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let (d, s, ii, k8) = (
                    self.iconst(dst.index() as i64),
                    self.iconst(set.index() as i64),
                    self.iconst(index.index() as i64),
                    self.iconst8(kind as i64),
                );
                let (c0, c1) = self.ctx2();
                let op_ = self.out_p();
                self.hstatus(i, H::GetIndex, &[r0, d, s, ii, k8, c0, c1, op_], &[index]);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::SetIndex {
                set,
                index,
                value,
            } => {
                if !self.iv_maybe(index) {
                    let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                    let (s, ii, v) = (
                        self.iconst(set.index() as i64),
                        self.iconst(index.index() as i64),
                        self.iconst(value.index() as i64),
                    );
                    let (c0, c1) = self.ctx2();
                    let op_ = self.out_p();
                    self.hstatus(
                        i,
                        H::SetIndex,
                        &[r0, s, ii, v, c0, c1, op_],
                        &[set, index, value],
                    );
                    let nb = self.next(i);
                    self.fb.ins().jump(nb, &[]);
                    return;
                }
                let slow = self.fb.create_block();
                self.fb.set_cold_block(slow);
                self.flush(set);
                self.flush(value);
                let Some(iv) = self.iv_or_slow(index, slow) else {
                    unreachable!("iv_maybe")
                };
                let sa = self.wslot(set);
                let va = self.wslot(value);
                let st = self.ld_tag(sa);
                let tarr = self.tconst(self.lyt.t_array);
                let isarr = self.fb.ins().icmp(IntCC::Equal, st, tarr);
                let arrb = self.fb.create_block();
                let notarr = self.fb.create_block();
                self.fb.ins().brif(isarr, arrb, &[], notarr, &[]);
                self.fb.switch_to_block(notarr);
                let seqb = self.fb.create_block();
                self.is_seq_tag(st, seqb, slow);
                self.fb.switch_to_block(seqb);
                {
                    let gc = self.fb.ins().load(I64, tfw(), sa, self.lyt.seq_pay as i32);
                    self.n_seq_write(i, iv, gc, va, slow);
                }
                self.fb.switch_to_block(arrb);
                {
                    let gc = self.fb.ins().load(I64, tfw(), sa, self.lyt.arr_pay as i32);
                    let fok = self.borrow_ok(gc, self.lyt.rl_flag, true);
                    let vt = self.ld_tag64(va);
                    let ngc = self.is_non_gc_tag(vt);
                    let pre = self.fb.ins().band(fok, ngc);
                    let vp = self
                        .fb
                        .ins()
                        .load(I64, tfhd(), gc, (self.lyt.rl_vec + self.lyt.vec_ptr) as i32);
                    let vl = self
                        .fb
                        .ins()
                        .load(I64, tfhd(), gc, (self.lyt.rl_vec + self.lyt.vec_len) as i32);
                    self.n_write_elem(i, va, iv, vl, vp, pre, slow);
                }
                self.fb.switch_to_block(slow);
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let (s, ii, v) = (
                    self.iconst(set.index() as i64),
                    self.iconst(index.index() as i64),
                    self.iconst(value.index() as i64),
                );
                let (c0, c1) = self.ctx2();
                let op_ = self.out_p();
                self.hstatus(i, H::SetIndex, &[r0, s, ii, v, c0, c1, op_], &[index]);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::GetField {
                dst,
                src,
                slot,
                kind,
            } => {
                if kind != AccessKind::Direct {
                    let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                    let (d, s, sl, k8) = (
                        self.iconst(dst.index() as i64),
                        self.iconst(src.index() as i64),
                        self.iconst(slot as i64),
                        self.iconst8(kind as i64),
                    );
                    let op_ = self.out_p();
                    self.hstatus(i, H::GetField, &[r0, d, s, sl, k8, op_], &[src]);
                    let nb = self.next(i);
                    self.fb.ins().jump(nb, &[]);
                    return;
                }
                let slow = self.fb.create_block();
                self.fb.set_cold_block(slow);
                self.flush(src);
                let ra = self.wslot(src);
                let rt = self.ld_tag(ra);
                let tinst = self.tconst(self.lyt.t_instance);
                let isinst = self.fb.ins().icmp(IntCC::Equal, rt, tinst);
                let instb = self.fb.create_block();
                let noti = self.fb.create_block();
                self.fb.ins().brif(isinst, instb, &[], noti, &[]);
                self.fb.switch_to_block(noti);
                let tarr = self.tconst(self.lyt.t_array);
                let isarr = self.fb.ins().icmp(IntCC::Equal, rt, tarr);
                let arrb = self.fb.create_block();
                let notarr = self.fb.create_block();
                self.fb.ins().brif(isarr, arrb, &[], notarr, &[]);
                self.fb.switch_to_block(notarr);
                let seqb = self.fb.create_block();
                self.is_seq_tag(rt, seqb, slow);
                self.fb.switch_to_block(seqb);
                {
                    let da = self.wslot(dst);
                    let gc = self.fb.ins().load(I64, tfw(), ra, self.lyt.seq_pay as i32);
                    let sv = self.iconst(slot as i64);
                    self.n_seq_read(i, da, sv, gc, slow);
                }
                self.fb.switch_to_block(arrb);
                {
                    let da = self.wslot(dst);
                    let gc = self.fb.ins().load(I64, tfw(), ra, self.lyt.arr_pay as i32);
                    let fok = self.borrow_ok(gc, self.lyt.rl_flag, false);
                    let vp = self
                        .fb
                        .ins()
                        .load(I64, tfhd(), gc, (self.lyt.rl_vec + self.lyt.vec_ptr) as i32);
                    let vl = self
                        .fb
                        .ins()
                        .load(I64, tfhd(), gc, (self.lyt.rl_vec + self.lyt.vec_len) as i32);
                    let sv = self.iconst(slot as i64);
                    self.n_read_elem(i, da, sv, vl, vp, fok, slow);
                }
                self.fb.switch_to_block(instb);
                {
                    let da = self.wslot(dst);
                    let gc = self.fb.ins().load(I64, tfw(), ra, self.lyt.inst_pay as i32);
                    let fok = self.borrow_ok(gc, self.lyt.rl_flag_i, false);
                    let fp = self
                        .fb
                        .ins()
                        .iadd_imm_s(gc, (self.lyt.rl_inst + self.lyt.id_fields) as i64);
                    let (len, data, isinl) = self.fields_inline(fp);
                    let pre = self.fb.ins().band(fok, isinl);
                    let sv = self.iconst(slot as i64);
                    self.n_read_elem(i, da, sv, len, data, pre, slow);
                }
                self.fb.switch_to_block(slow);
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let (d, s, sl, k8) = (
                    self.iconst(dst.index() as i64),
                    self.iconst(src.index() as i64),
                    self.iconst(slot as i64),
                    self.iconst8(kind as i64),
                );
                let op_ = self.out_p();
                self.hstatus(i, H::GetField, &[r0, d, s, sl, k8, op_], &[]);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::SetField {
                receiver,
                slot,
                value,
            } => {
                let slow = self.fb.create_block();
                self.fb.set_cold_block(slow);
                self.flush(receiver);
                self.flush(value);
                let ra = self.wslot(receiver);
                let va = self.wslot(value);
                let rt = self.ld_tag(ra);
                let vt = self.ld_tag64(va);
                let ngc = self.is_non_gc_tag(vt);
                let tinst = self.tconst(self.lyt.t_instance);
                let isinst = self.fb.ins().icmp(IntCC::Equal, rt, tinst);
                let instb = self.fb.create_block();
                let noti = self.fb.create_block();
                self.fb.ins().brif(isinst, instb, &[], noti, &[]);
                self.fb.switch_to_block(noti);
                let tarr = self.tconst(self.lyt.t_array);
                let isarr = self.fb.ins().icmp(IntCC::Equal, rt, tarr);
                let arrb = self.fb.create_block();
                let notarr = self.fb.create_block();
                self.fb.ins().brif(isarr, arrb, &[], notarr, &[]);
                self.fb.switch_to_block(notarr);
                let seqb = self.fb.create_block();
                self.is_seq_tag(rt, seqb, slow);
                self.fb.switch_to_block(seqb);
                {
                    let gc = self.fb.ins().load(I64, tfw(), ra, self.lyt.seq_pay as i32);
                    let sv = self.iconst(slot as i64);
                    self.n_seq_write(i, sv, gc, va, slow);
                }
                self.fb.switch_to_block(arrb);
                {
                    let gc = self.fb.ins().load(I64, tfw(), ra, self.lyt.arr_pay as i32);
                    let fok = self.borrow_ok(gc, self.lyt.rl_flag, true);
                    let pre = self.fb.ins().band(fok, ngc);
                    let vp = self
                        .fb
                        .ins()
                        .load(I64, tfhd(), gc, (self.lyt.rl_vec + self.lyt.vec_ptr) as i32);
                    let vl = self
                        .fb
                        .ins()
                        .load(I64, tfhd(), gc, (self.lyt.rl_vec + self.lyt.vec_len) as i32);
                    let sv = self.iconst(slot as i64);
                    self.n_write_elem(i, va, sv, vl, vp, pre, slow);
                }
                self.fb.switch_to_block(instb);
                {
                    let gc = self.fb.ins().load(I64, tfw(), ra, self.lyt.inst_pay as i32);
                    let fok = self.borrow_ok(gc, self.lyt.rl_flag_i, true);
                    let fp = self
                        .fb
                        .ins()
                        .iadd_imm_s(gc, (self.lyt.rl_inst + self.lyt.id_fields) as i64);
                    let (len, data, isinl) = self.fields_inline(fp);
                    let pre = self.fb.ins().band(fok, ngc);
                    let pre = self.fb.ins().band(pre, isinl);
                    let sv = self.iconst(slot as i64);
                    self.n_write_elem(i, va, sv, len, data, pre, slow);
                }
                self.fb.switch_to_block(slow);
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let (rc, sl, v) = (
                    self.iconst(receiver.index() as i64),
                    self.iconst(slot as i64),
                    self.iconst(value.index() as i64),
                );
                let (c0, c1) = self.ctx2();
                let op_ = self.out_p();
                self.hstatus(i, H::SetField, &[r0, rc, sl, v, c0, c1, op_], &[]);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::Len { dst, src } => {
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let s = self.iconst(src.index() as i64);
                let slot = self.fb.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    8,
                    3,
                ));
                let vp = self.fb.ins().stack_addr(I64, slot, 0);
                let op_ = self.out_p();
                self.hstatus(i, H::Len, &[r0, s, vp, op_], &[src]);
                let v = self.fb.ins().stack_load(I64, I64, slot, 0);
                self.dv(dst, v, NK::Int);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::Bin {
                dst,
                left,
                op,
                right,
            } => {
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let (d, l, o8, r) = (
                    self.iconst(dst.index() as i64),
                    self.iconst(left.index() as i64),
                    self.iconst8(op as i64),
                    self.iconst(right.index() as i64),
                );
                let (c0, c1) = self.ctx2();
                let op_ = self.out_p();
                self.hstatus(
                    i,
                    H::Bin,
                    &[r0, d, l, o8, r, c0, c1, op_],
                    &[left, right],
                );
                self.paused_gate(i);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::Unary { dst, op, src } => {
                let s = src.index();
                if op == UnaryOp::Not
                    && (self.wb(s) || self.kinds[s] == Some(NK::Bool))
                {
                    // `!b` on a Bool — `rv` tag-checks window-backed srcs,
                    // suspending to the framed `unary` on a miss (incl.
                    // instance-op impls); a proven-Bool var needs no check.
                    let b = self.rv(i, src, NK::Bool);
                    let one = self.iconst(1);
                    let nb2 = self.fb.ins().bxor(b, one);
                    self.dv(dst, nb2, NK::Bool);
                } else {
                    let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                    let (d, o8, s) = (
                        self.iconst(dst.index() as i64),
                        self.iconst8(op as i64),
                        self.iconst(src.index() as i64),
                    );
                    let (c0, c1) = self.ctx2();
                    let op_ = self.out_p();
                    self.hstatus(i, H::Unary, &[r0, d, o8, s, c0, c1, op_], &[src]);
                    self.paused_gate(i);
                }
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::Push { array, value } => {
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let (a, v) = (
                    self.iconst(array.index() as i64),
                    self.iconst(value.index() as i64),
                );
                let (c0, c1) = self.ctx2();
                self.hvoid(i, H::Push, &[r0, a, v, c0, c1], &[array, value]);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::Insert { dict, key, value } => {
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let (d, k, v) = (
                    self.iconst(dict.index() as i64),
                    self.iconst32(key.index() as i64),
                    self.iconst(value.index() as i64),
                );
                let (c0, c1) = self.ctx2();
                let st = self.el(ENV_STRS);
                self.hvoid(
                    i,
                    H::Insert,
                    &[r0, d, k, v, c0, c1, st],
                    &[dict, value],
                );
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::NewArray { dst } | Op::NewDict { dst } => {
                let h = if matches!(op, Op::NewArray { .. }) {
                    H::NewArray
                } else {
                    H::NewDict
                };
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let d = self.iconst(dst.index() as i64);
                let (c0, c1) = self.ctx2();
                self.hvoid(i, h, &[r0, d, c0, c1], &[]);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::NewInstance { dst, adt, fields } => {
                for &f in fields.iter() {
                    self.flush(f);
                }
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let fp = self.reg_list(&fields);
                let (d, a32, n) = (
                    self.iconst(dst.index() as i64),
                    self.iconst32(adt.index() as i64),
                    self.iconst(fields.len() as i64),
                );
                let (c0, c1) = self.ctx2();
                self.hvoid(i, H::NewInstance, &[r0, d, a32, fp, n, c0, c1], &[]);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::NewClosure {
                dst,
                body,
                captures,
            } => {
                for &c in captures.iter() {
                    self.flush(c);
                }
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let cp = self.reg_list(&captures);
                let (d, b32, n) = (
                    self.iconst(dst.index() as i64),
                    self.iconst32(body.index() as i64),
                    self.iconst(captures.len() as i64),
                );
                let (c0, c1) = self.ctx2();
                self.hvoid(i, H::NewClosure, &[r0, d, b32, cp, n, c0, c1], &[]);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::CallNative { dst, id, args } => {
                for &a in args.iter() {
                    self.flush(a);
                }
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let ap = self.reg_list(&args);
                let t = self.el(ENV_THREAD);
                let cd = self.el(ENV_CODE);
                let (d, nid, n) = (
                    self.iconst(dst.index() as i64),
                    self.iconst32(id.index() as i64),
                    self.iconst(args.len() as i64),
                );
                let (c0, c1) = self.ctx2();
                let op_ = self.out_p();
                self.hstatus(
                    i,
                    H::CallNative,
                    &[t, r0, d, nid, ap, n, cd, c0, c1, op_],
                    &[],
                );
                self.paused_gate(i);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::In {
                dst,
                needle,
                haystack,
                condition,
            } => {
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let (d, n, hh, c8) = (
                    self.iconst(dst.index() as i64),
                    self.iconst(needle.index() as i64),
                    self.iconst(haystack.index() as i64),
                    self.iconst8(condition as i64),
                );
                self.hvoid(
                    i,
                    H::ContainsOp,
                    &[r0, d, n, hh, c8],
                    &[needle, haystack],
                );
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::IsInstance { dst, src, adt } => {
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let (d, s, a32) = (
                    self.iconst(dst.index() as i64),
                    self.iconst(src.index() as i64),
                    self.iconst32(adt.index() as i64),
                );
                self.hvoid(i, H::IsInstance, &[r0, d, s, a32], &[src]);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::IsRaised { dst, src } => {
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let (d, s) = (
                    self.iconst(dst.index() as i64),
                    self.iconst(src.index() as i64),
                );
                self.hvoid(i, H::IsRaised, &[r0, d, s], &[src]);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::Unwrap { dst, src } | Op::UnwrapUnit { dst, src } => {
                let h = if matches!(op, Op::Unwrap { .. }) {
                    H::Unwrap
                } else {
                    H::UnwrapUnit
                };
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let (d, s) = (
                    self.iconst(dst.index() as i64),
                    self.iconst(src.index() as i64),
                );
                let op_ = self.out_p();
                self.hstatus(i, h, &[r0, d, s, op_], &[src]);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::UnwrapRaised { dst, src } => {
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let (d, s) = (
                    self.iconst(dst.index() as i64),
                    self.iconst(src.index() as i64),
                );
                self.hvoid(i, H::UnwrapRaised, &[r0, d, s], &[src]);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::Raise { val } => {
                // `*out = Ok(Flow::Return(Raised))` — propagate verbatim.
                self.flush(val);
                self.mark_op(i);
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let s = self.iconst(val.index() as i64);
                let op_ = self.out_p();
                self.hcall(H::Raise, &[r0, s, op_]);
                self.settle();
                let s1 = self.fb.ins().iconst(I8, ST_PROP);
                let zz = self.iconst(0);
                self.fb.ins().return_(&[s1, zz]);
            }
            Op::LoadEntry { dst, slot } => {
                // `regs[slot]` — an absolute entry-frame index — copied raw
                // into dst's window slot (dst is always window-backed).
                let rp = self.regs_ptr();
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let vs = self.lyt.val_size as i64;
                let sa = self
                    .fb
                    .ins()
                    .iadd_imm_s(rp, (slot.index() as i64) * vs);
                let da = self
                    .fb
                    .ins()
                    .iadd_imm_s(r0, (dst.index() as i64) * vs);
                for off in (0..vs).step_by(8) {
                    let w = self.fb.ins().load(I64, tfw(), sa, off as i32);
                    self.fb.ins().store(tfw(), w, da, off as i32);
                }
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::StoreEntry { slot, src } => {
                self.flush(src);
                let rp = self.regs_ptr();
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let vs = self.lyt.val_size as i64;
                let sa = self
                    .fb
                    .ins()
                    .iadd_imm_s(r0, (src.index() as i64) * vs);
                let da = self
                    .fb
                    .ins()
                    .iadd_imm_s(rp, (slot.index() as i64) * vs);
                for off in (0..vs).step_by(8) {
                    let w = self.fb.ins().load(I64, tfw(), sa, off as i32);
                    self.fb.ins().store(tfw(), w, da, off as i32);
                }
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::LoadBody { dst, body } => {
                let r0 = self.fb.use_var(self.regs0.expect("windowed"));
                let (d, b32) = (
                    self.iconst(dst.index() as i64),
                    self.iconst32(body.index() as i64),
                );
                self.hvoid(i, H::WrFn, &[r0, d, b32], &[]);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            _ => unreachable!("checkpoint op dispatched to emitter"),
        }
    }
}

/// Compile `chunks[body]` as a frameless native body into `native_ids[body]`.
/// `kinds` is the proven per-reg kind map from [`analyze`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_native_body(
    module: &mut JITModule,
    prog: &Program,
    body: usize,
    helper_ids: &[FuncId],
    native_ids: &[Option<FuncId>],
    plan: &NativePlan,
    lyt: &Layout,
    fbc: &mut FunctionBuilderContext,
    ctx: &mut Context,
) -> Result<(), String> {
    module.clear_context(ctx);
    let chunk = &prog.chunks[compile::BodyId::from(body as u32)];
    let kinds = &plan.kinds;
    ctx.func.signature = native_sig(module, chunk.params.len());

    // same alias-region indices as emit.rs
    for (id, desc) in [
        (0u32, "reg window"),
        (1, "gc heap"),
        (2, "vm state"),
        (3, "gc heap headers"),
        (4, "gc heap elements"),
    ] {
        let ar = ctx.func.dfg.alias_regions.insert(ir::AliasRegionData {
            user_id: id,
            description: desc.into(),
        });
        debug_assert_eq!(ar.index(), id as usize);
    }

    let hrefs: Vec<FuncRef> = helper_ids
        .iter()
        .map(|id| module.declare_func_in_func(*id, &mut ctx.func))
        .collect();
    let nrefs: HashMap<u32, FuncRef> = native_ids
        .iter()
        .enumerate()
        .filter_map(|(b, id)| id.map(|id| (b as u32, module.declare_func_in_func(id, &mut ctx.func))))
        .collect();

    let mut fb = FunctionBuilder::new(&mut ctx.func, fbc);

    let nregs = chunk.regs as usize;
    let vars: Vec<Variable> = (0..nregs)
        .map(|r| {
            fb.declare_var(if kinds[r] == Some(NK::Float) {
                F64
            } else {
                I64
            })
        })
        .collect();
    let bcn = fb.declare_var(I64);
    let bcn0 = fb.declare_var(I64);
    let wrote = ((plan.checkpoints.iter().any(|&c| c) || plan.windowed)
        && !plan.unsafe_cp)
        .then(|| fb.declare_var(I64));
    let regs0 = plan.windowed.then(|| fb.declare_var(I64));

    let ops = prog.ops(compile::BodyId::from(body as u32));
    let off2idx: HashMap<usize, usize> =
        ops.iter().enumerate().map(|(i, (o, _))| (*o, i)).collect();
    let blocks: Vec<Block> = (0..ops.len()).map(|_| fb.create_block()).collect();

    // Run heads for the batched-quota gate: op 0 plus every branch target —
    // same scheme as emit.rs. `run_len[i]` = head-gate bound at head i;
    // `run_end[i]` = index past i's run (post-call re-arm bound).
    let mut is_head = vec![false; ops.len()];
    is_head[0] = true;
    for (_, op) in ops.iter() {
        for t in op.targets() {
            if let BlockTarget::ByteOffset(o) = t {
                if let Some(&j) = off2idx.get(&o) {
                    is_head[j] = true;
                }
            }
        }
    }
    let mut run_len = vec![0usize; ops.len()];
    let mut next_head = ops.len();
    for i in (0..ops.len()).rev() {
        if is_head[i] {
            run_len[i] = next_head - i;
            next_head = i;
        }
    }
    let mut run_end = vec![0usize; ops.len()];
    let mut head = 0;
    for (i, e) in run_end.iter_mut().enumerate() {
        if is_head[i] {
            head = i;
        }
        *e = head + run_len[head];
    }

    if std::env::var_os("MIMAS_DUMP_NATIVE_CLIF").is_some() {
        eprintln!("-- native body {body}: checkpoints={:?} unsafe_cp={}", plan.checkpoints, plan.unsafe_cp);
    }
    let entry = fb.create_block();
    fb.append_block_params_for_function_params(entry);
    fb.switch_to_block(entry);
    let params: Vec<Value> = fb.block_params(entry).to_vec();
    let envp = params[0];
    let depth = params[1];
    let zi = fb.ins().iconst(I64, 0);
    let zf = fb.ins().f64const(0.0);
    for (r, v) in vars.iter().enumerate() {
        fb.def_var(*v, if kinds[r] == Some(NK::Float) { zf } else { zi });
    }
    for (j, &p) in chunk.params.iter().enumerate() {
        let a = params[2 + j];
        fb.def_var(vars[p.index()], a);
    }
    if let Some(wv) = wrote {
        fb.def_var(wv, zi);
    }

    let etrip = fb.create_block();
    fb.set_cold_block(etrip);
    let nops = ops.len();
    let mut em = Ne {
        fb,
        hrefs: &hrefs,
        nrefs: &nrefs,
        vars,
        blocks,
        off2idx,
        ops,
        run_len,
        run_end,
        envp,
        depth,
        opsleft_off: lyt.ops_left_off as i64,
        lyt,
        kinds: &plan.kinds,
        conflict: &plan.conflict,
        checkpoints: &plan.checkpoints,
        unsafe_cp: plan.unsafe_cp,
        regs0,
        cpb: vec![None; nops],
        windowed: plan.windowed,
        wrote,
        bcn,
        bcn0,
        etrip,
        errs: Vec::new(),
    };
    em.arm();
    if plan.windowed {
        // Window ops are only legal when this activation owns the frame —
        // a frameless callee's `regs0` would index its *caller's* window.
        let m1 = em.iconst(-1);
        let nested = em.fb.ins().icmp(IntCC::NotEqual, depth, m1);
        let okb = em.fb.create_block();
        em.fb.ins().brif(nested, etrip, &[], okb, &[]);
        em.fb.switch_to_block(okb);
        // `regs[frames.last().base]` — same derivation as the shim.
        let t = em.el(ENV_THREAD);
        let fptr = em
            .fb
            .ins()
            .load(I64, tfs(), t, (lyt.frames_off + lyt.vec_ptr) as i32);
        let flen = em
            .fb
            .ins()
            .load(I64, tfs(), t, (lyt.frames_off + lyt.vec_len) as i32);
        let fm1 = em.fb.ins().iadd_imm_s(flen, -1);
        let foff = em.fb.ins().imul_imm_s(fm1, lyt.frame_size as i64);
        let faddr = em.fb.ins().iadd(fptr, foff);
        let base = em
            .fb
            .ins()
            .load(I64, tfs(), faddr, lyt.frame_base as i32);
        let rp = em
            .fb
            .ins()
            .load(I64, tfs(), t, (lyt.regs_off + lyt.vec_ptr) as i32);
        let boff = em.fb.ins().imul_imm_s(base, lyt.val_size as i64);
        let r0 = em.fb.ins().iadd(rp, boff);
        em.fb.def_var(regs0.expect("windowed"), r0);
    }
    let first = em.blocks[0];
    em.fb.ins().jump(first, &[]);
    em.fb.seal_block(entry);

    let nops = em.ops.len();
    for i in 0..nops {
        let op = em.ops[i].1.clone();
        em.emit_op(i, &op);
    }
    // Deferred tag-miss suspends — same body as an op-0 checkpoint.
    for i in 0..nops {
        if let Some(b) = em.cpb[i] {
            em.fb.switch_to_block(b);
            em.suspend_at(i);
        }
    }

    em.fb.switch_to_block(em.etrip);
    em.settle();
    let s2 = em.fb.ins().iconst(I8, ST_RETRY);
    let zz = em.iconst(0);
    em.fb.ins().return_(&[s2, zz]);

    for (t, kind) in std::mem::take(&mut em.errs) {
        em.fb.switch_to_block(t);
        em.settle();
        let k8 = em.fb.ins().iconst(I8, kind);
        let op = em.out_p();
        em.fb.ins().call(em.hrefs[H::OutErr as usize], &[op, k8]);
        let s1 = em.fb.ins().iconst(I8, ST_PROP);
        let zz = em.iconst(0);
        em.fb.ins().return_(&[s1, zz]);
    }

    em.fb.seal_all_blocks();
    let fe_cfg = module.isa().frontend_config();
    em.fb.finalize(fe_cfg);
    if std::env::var_os("MIMAS_DUMP_NATIVE_CLIF").is_some() {
        eprintln!("{}", ctx.func.display());
    }
    if let Err(e) = module.define_function(native_ids[body].unwrap(), ctx) {
        if std::env::var_os("MIMAS_DUMP_NATIVE_CLIF").is_some() {
            eprintln!("{e:?}");
        }
        return Err(format!("native body {body}: {e}"));
    }
    Ok(())
}

/// Compile a `BodyFn`-ABI entry shim for a native-eligible body into
/// `shim_id`: the driver calls `bodies[b](env)` as usual, and the shim
/// forwards straight into the frameless convention —
/// `native(env, 0, args…)` with each arg read from the top frame's window
/// (`regs[frames.last().base + params[j]]`, tag-checked `Int`; a miss
/// can't be a valid call for an all-`Int` body, so it takes the framed
/// path which raises the proper type error). Status handling:
/// `ST_OK` writes `*out = Ok(Flow::Return(Val::Int(v)))` inline;
/// `ST_PROP` returns untouched — the callee's error helper already wrote
/// `*out`; `ST_RETRY` calls the framed body — the call made no progress,
/// so the driver's contract is identical to having invoked the framed
/// body all along.
pub(crate) fn emit_native_shim(
    module: &mut JITModule,
    prog: &Program,
    body: usize,
    plan: &NativePlan,
    shim_id: FuncId,
    framed_id: FuncId,
    native_id: FuncId,
    lyt: &Layout,
    fbc: &mut FunctionBuilderContext,
    ctx: &mut Context,
) -> Result<(), String> {
    module.clear_context(ctx);
    let chunk = &prog.chunks[compile::BodyId::from(body as u32)];
    let mut s = module.make_signature();
    s.params.push(AbiParam::new(I64)); // env
    ctx.func.signature = s;

    // same alias-region indices as emit.rs
    for (id, desc) in [
        (0u32, "reg window"),
        (1, "gc heap"),
        (2, "vm state"),
        (3, "gc heap headers"),
        (4, "gc heap elements"),
    ] {
        let ar = ctx.func.dfg.alias_regions.insert(ir::AliasRegionData {
            user_id: id,
            description: desc.into(),
        });
        debug_assert_eq!(ar.index(), id as usize);
    }

    let nfr = module.declare_func_in_func(native_id, &mut ctx.func);
    let ffr = module.declare_func_in_func(framed_id, &mut ctx.func);

    let mut fb = FunctionBuilder::new(&mut ctx.func, fbc);
    let entry = fb.create_block();
    fb.append_block_params_for_function_params(entry);
    fb.switch_to_block(entry);
    let envp = fb.block_params(entry)[0];

    let ecall = fb.create_block();
    let eframed = fb.create_block(); // tag-miss or ST_RETRY → framed path
    fb.set_cold_block(eframed);

    // A checkpoint *at the entry op* makes `code.ip == entry` ambiguous:
    // it means both "fresh entry" and "suspended before op 0" — taking the
    // native path on the latter would re-hit the checkpoint and loop
    // forever. Framed is correct for both readings, so such bodies never
    // enter natively from the driver (the native body still serves ncalls,
    // where the checkpoint just `ST_RETRY`s).
    if plan.checkpoints.first() == Some(&true) {
        fb.ins().jump(eframed, &[]);
        fb.switch_to_block(eframed);
        fb.ins().call(ffr, &[envp]);
        fb.ins().return_(&[]);
        fb.seal_all_blocks();
        let fe_cfg = module.isa().frontend_config();
        fb.finalize(fe_cfg);
        return module
            .define_function(shim_id, ctx)
            .map_err(|e| format!("native shim {body}: {e}"));
    }

    // Frameless bodies always start at op 0 — a `Flow::Next` resume would
    // lose the whole mid-body `code.ip` resume point, so only a fresh
    // entry (`code.ip == the body's first op offset`) may take the native
    // path; every mid-body dispatch goes straight to the framed body.
    // (Without this check, each `run_dispatch` fuel batch would burn its
    // entire quota re-running the body's head only to ST_RETRY.)
    let entry_off = prog
        .ops(compile::BodyId::from(body as u32))
        .first()
        .map(|(o, _)| *o)
        .unwrap_or(0) as i64;
    let cp = fb.ins().load(I64, tfs(), envp, ENV_CODE);
    let cip = fb.ins().load(I64, tfs(), cp, lyt.code_ip as i32);
    let eoff = fb.ins().iconst(I64, entry_off);
    let at_entry = fb.ins().icmp(IntCC::Equal, cip, eoff);
    let fresh = fb.create_block();
    fb.ins().brif(at_entry, fresh, &[], eframed, &[]);
    fb.switch_to_block(fresh);

    // `regs[frames.last().base + params[j]]`, tag-check each window slot.
    let t = fb.ins().load(I64, tfs(), envp, ENV_THREAD);
    let fptr = fb
        .ins()
        .load(I64, tfs(), t, (lyt.frames_off + lyt.vec_ptr) as i32);
    let flen = fb
        .ins()
        .load(I64, tfs(), t, (lyt.frames_off + lyt.vec_len) as i32);
    let fm1 = fb.ins().iadd_imm_s(flen, -1);
    let foff = fb.ins().imul_imm_s(fm1, lyt.frame_size as i64);
    let faddr = fb.ins().iadd(fptr, foff);
    let base = fb.ins().load(I64, tfs(), faddr, lyt.frame_base as i32);
    let rp = fb
        .ins()
        .load(I64, tfs(), t, (lyt.regs_off + lyt.vec_ptr) as i32);
    let boff = fb.ins().imul_imm_s(base, lyt.val_size as i64);
    let regs0 = fb.ins().iadd(rp, boff);

    let mut cargs = Vec::with_capacity(2 + chunk.params.len());
    cargs.push(envp);
    // `depth = -1` marks "this activation owns a frame": a checkpoint may
    // suspend only there (frameless callees have no window to spill into,
    // their checkpoints unwind `ST_RETRY` instead). `ncall`s count up from
    // -1 → 0, 1, … so the depth cap still bites one level late — harmless.
    let zd = fb.ins().iconst(I64, -1);
    cargs.push(zd);
    let tty = match lyt.tag_size {
        1 => I8,
        2 => I16,
        4 => I32,
        _ => I64,
    };
    for &p in &chunk.params {
        let r = (p.index() * lyt.val_size) as i64;
        let a = fb.ins().iadd_imm_s(regs0, r);
        let tag = fb.ins().load(tty, tfw(), a, lyt.val_tag as i32);
        let tint = fb.ins().iconst(tty, lyt.t_int as i64);
        let hit = fb.ins().icmp(IntCC::Equal, tag, tint);
        let good = fb.create_block();
        fb.ins().brif(hit, good, &[], eframed, &[]);
        fb.switch_to_block(good);
        cargs.push(fb.ins().load(I64, tfw(), a, lyt.val_pay as i32));
    }
    fb.ins().jump(ecall, &[]);

    fb.switch_to_block(ecall);
    let inst = fb.ins().call(nfr, &cargs);
    let st = fb.inst_results(inst)[0];
    let v = fb.inst_results(inst)[1];
    let edone = fb.create_block();
    let eok = fb.create_block();
    let s2 = fb.ins().iconst(I8, ST_RETRY);
    let retry = fb.ins().icmp(IntCC::Equal, st, s2);
    let nretry = fb.create_block();
    fb.ins().brif(retry, eframed, &[], nretry, &[]);
    fb.switch_to_block(nretry);
    let s0 = fb.ins().iconst(I8, ST_OK);
    let isok = fb.ins().icmp(IntCC::Equal, st, s0);
    fb.ins().brif(isok, eok, &[], edone, &[]); // ST_PROP: *out already written

    // `*out = Ok(Flow::Return(Val::Int(v)))` — same inline materialization
    // as the framed Return path.
    fb.switch_to_block(eok);
    let outp = fb.ins().load(I64, tfs(), envp, ENV_OUT);
    let oty = match lyt.out_tsz {
        1 => I8,
        2 => I16,
        4 => I32,
        _ => I64,
    };
    let oret = fb.ins().iconst(oty, lyt.out_ret as i64);
    fb.ins().store(tfs(), oret, outp, lyt.out_tag as i32);
    let dp = fb.ins().iadd_imm_s(outp, lyt.out_ret_pay as i64);
    let tint = fb.ins().iconst(tty, lyt.t_int as i64);
    fb.ins().store(tfs(), tint, dp, lyt.val_tag as i32);
    fb.ins().store(tfs(), v, dp, lyt.val_pay as i32);
    fb.ins().jump(edone, &[]);

    fb.switch_to_block(eframed);
    fb.ins().call(ffr, &[envp]);
    fb.ins().jump(edone, &[]);

    fb.switch_to_block(edone);
    fb.ins().return_(&[]);

    fb.seal_all_blocks();
    let fe_cfg = module.isa().frontend_config();
    fb.finalize(fe_cfg);
    module
        .define_function(shim_id, ctx)
        .map_err(|e| format!("native shim {body}: {e}"))?;
    Ok(())
}

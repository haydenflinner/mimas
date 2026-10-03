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
//! unwind-retry escape: fuel/ops_left exhaustion mid-body, `paused` set,
//! the native-depth cap, or an edge case a flat op can't inline — the call
//! made no progress and the *framed* ancestor's call site re-runs it through
//! the ordinary frame path. Whitelisted bodies have no side effects, so
//! discarding partial work is safe. Quota is self-accounting: the callee
//! arms `bcn`/`bcn0` from `fuel`/`ops_left` at entry and settles at every
//! exit, so callers never prepay and accounting stays exact even on early
//! returns.

use std::collections::HashMap;

use compile::{BlockTarget, Constant, Op, Program};
use cranelift_codegen::Context;
use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    self, AbiParam, Block, FuncRef, InstBuilder, MemFlagsData, Signature, Value, types,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};
use vm::bc::jit::Layout;

use crate::H;
use crate::emit::{ENV_FUEL, ENV_OUT, ENV_ST, ENV_THREAD, ERR_MOD0, ERR_OVFW, ERR_PANIC};

const I64: ir::Type = types::I64;
const I8: ir::Type = types::I8;
const F64: ir::Type = types::F64;

/// VM state region — same index as `emit::tfs`.
fn tfs() -> MemFlagsData {
    MemFlagsData::trusted().with_alias_region(Some(ir::AliasRegion::new(2)))
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

/// `Some(kinds)` when `chunks[body]`'s op set is flat-safe; `kinds[r]` is
/// the proven scalar kind of reg `r`. Call-graph eligibility is layered on
/// by [`analyze`].
pub(crate) fn native_kinds(prog: &Program, body: u32) -> Option<Vec<Option<NK>>> {
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
    for (_, op) in &ops {
        let w: Option<(u32, Option<NK>)> = match *op {
            Op::Move { dst, src } => Some((dst.index() as u32, kinds[src.index()])),
            Op::LoadConst { dst, ref constant } => match *constant {
                Constant::Int(_) => Some((dst.index() as u32, Some(NK::Int))),
                Constant::Float(_) => Some((dst.index() as u32, Some(NK::Float))),
                Constant::Bool(_) => Some((dst.index() as u32, Some(NK::Bool))),
                Constant::Null => Some((dst.index() as u32, Some(NK::Null))),
                _ => return None,
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
            _ => return None,
        };
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
    // Operand requirements — every typed read must see the right kind.
    let need = |r: compile::Reg, k: NK| kinds[r.index()] == Some(k) && !conflict[r.index()];
    let int2 = |l: compile::Reg, r: compile::Reg| need(l, NK::Int) && need(r, NK::Int);
    let flt2 = |l: compile::Reg, r: compile::Reg| need(l, NK::Float) && need(r, NK::Float);
    for (_, op) in &ops {
        let ok = match *op {
            Op::Move { .. } | Op::Jump { .. } | Op::Panic {} | Op::LoadConst { .. } => true,
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
            _ => false,
        };
        if !ok {
            return None;
        }
    }
    Some(kinds)
}

/// Per-body eligibility: `native_kinds` is a necessary condition AND every
/// `CallDirect` target must itself be eligible — fixpoint over the call
/// graph (self/mutual recursion included).
pub(crate) fn analyze(prog: &Program) -> Vec<Option<Vec<Option<NK>>>> {
    let n = prog.chunks.len();
    let mut kinds: Vec<Option<Vec<Option<NK>>>> =
        (0..n).map(|b| native_kinds(prog, b as u32)).collect();
    loop {
        let mut changed = false;
        for b in 0..n {
            if kinds[b].is_none() {
                continue;
            }
            let bad = prog
                .ops(compile::BodyId::from(b as u32))
                .iter()
                .any(|(_, op)| match op {
                    Op::CallDirect { body, .. } => kinds[body.index()].is_none(),
                    _ => false,
                });
            if bad {
                kinds[b] = None;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    kinds
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
    lyt: &'b Layout,
    hrefs: &'b [FuncRef],
    nrefs: &'b HashMap<u32, FuncRef>,
    vars: Vec<Variable>,
    blocks: Vec<Block>,
    off2idx: HashMap<usize, usize>,
    ops: Vec<(usize, Op)>,
    envp: Value,
    depth: Value,
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

    fn fuel_p(&mut self) -> Value {
        self.el(ENV_FUEL)
    }

    fn out_p(&mut self) -> Value {
        self.el(ENV_OUT)
    }

    fn paused_p(&mut self) -> Value {
        let s = self.el(ENV_ST);
        self.fb.ins().iadd_imm_s(s, self.lyt.state_paused as i64)
    }

    fn opsleft_p(&mut self) -> Value {
        let t = self.el(ENV_THREAD);
        self.fb.ins().iadd_imm_s(t, self.lyt.ops_left_off as i64)
    }

    /// Write the batched `bcn` spend back to `fuel`/`ops_left`.
    fn settle(&mut self) {
        let b0 = self.fb.use_var(self.bcn0);
        let b = self.fb.use_var(self.bcn);
        let spent = self.fb.ins().isub(b0, b);
        self.fb.def_var(self.bcn0, b);
        let fp = self.fuel_p();
        let f = self.fb.ins().load(I64, tfs(), fp, 0);
        let f2 = self.fb.ins().isub(f, spent);
        self.fb.ins().store(tfs(), f2, fp, 0);
        let op = self.opsleft_p();
        let ol = self.fb.ins().load(I64, tfs(), op, 0);
        let ol2 = self.fb.ins().isub(ol, spent);
        self.fb.ins().store(tfs(), ol2, op, 0);
    }

    /// `bcn = bcn0 = min(*fuel, *ops_left)` — at entry and after each call.
    fn arm(&mut self) {
        let fp = self.fuel_p();
        let f = self.fb.ins().load(I64, tfs(), fp, 0);
        let op = self.opsleft_p();
        let ol = self.fb.ins().load(I64, tfs(), op, 0);
        let m = self.fb.ins().umin(f, ol);
        self.fb.def_var(self.bcn, m);
        self.fb.def_var(self.bcn0, m);
    }

    /// One op's quota gate: `bcn == 0` → unwind-retry, else decrement.
    fn gate(&mut self) {
        let cur = self.fb.use_var(self.bcn);
        let z = self.iconst(0);
        let bz = self.fb.ins().icmp(IntCC::Equal, cur, z);
        let cont = self.fb.create_block();
        self.fb.ins().brif(bz, self.etrip, &[], cont, &[]);
        self.fb.switch_to_block(cont);
        let b1 = self.fb.ins().iadd_imm_s(cur, -1);
        self.fb.def_var(self.bcn, b1);
    }

    fn rv(&mut self, r: compile::Reg) -> Value {
        self.fb.use_var(self.vars[r.index()])
    }

    fn dv(&mut self, r: compile::Reg, v: Value) {
        self.fb.def_var(self.vars[r.index()], v);
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
        let a = self.rv(l);
        let b = self.rv(r);
        let c = self.fb.ins().icmp(cc, a, b);
        let c64 = self.fb.ins().uextend(I64, c);
        self.dv(dst, c64);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    fn eval_i_imm(&mut self, i: usize, dst: compile::Reg, l: compile::Reg, v: i64, cc: IntCC) {
        let a = self.rv(l);
        let b = self.iconst(v);
        let c = self.fb.ins().icmp(cc, a, b);
        let c64 = self.fb.ins().uextend(I64, c);
        self.dv(dst, c64);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    fn eval_f(&mut self, i: usize, dst: compile::Reg, l: compile::Reg, r: compile::Reg, cc: FloatCC) {
        let a = self.rv(l);
        let b = self.rv(r);
        let c = self.fb.ins().fcmp(cc, a, b);
        let c64 = self.fb.ins().uextend(I64, c);
        self.dv(dst, c64);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    fn eval_f_imm(&mut self, i: usize, dst: compile::Reg, l: compile::Reg, v: i64, cc: FloatCC) {
        let a = self.rv(l);
        let b = self.fb.ins().f64const(f64::from_bits(v as u64));
        let c = self.fb.ins().fcmp(cc, a, b);
        let c64 = self.fb.ins().uextend(I64, c);
        self.dv(dst, c64);
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
        self.dv(dst, v);
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
        self.fb.ins().brif(bad, self.etrip, &[], run2, &[]);
        self.fb.switch_to_block(run2);
        let v = self.fb.ins().srem(a, b);
        self.dv(dst, v);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    /// `CallDirect` to an eligible callee — paused/depth → unwind; status 1
    /// or 2 propagates verbatim (settle first); 0 writes `dst`.
    fn ncall(&mut self, i: usize, dst: compile::Reg, tb: u32, args: &[compile::Reg]) {
        let pp = self.paused_p();
        let p = self.fb.ins().load(I8, tfs(), pp, 0);
        let z8 = self.fb.ins().iconst(I8, 0);
        let pn = self.fb.ins().icmp(IntCC::NotEqual, p, z8);
        let cont = self.fb.create_block();
        self.fb.ins().brif(pn, self.etrip, &[], cont, &[]);
        self.fb.switch_to_block(cont);
        let d = self.depth;
        let d2 = self.fb.ins().iadd_imm_s(d, 1);
        let lim = self.iconst(NATIVE_DEPTH);
        let over = self.fb.ins().icmp(IntCC::SignedGreaterThanOrEqual, d2, lim);
        let cont2 = self.fb.create_block();
        self.fb.ins().brif(over, self.etrip, &[], cont2, &[]);
        self.fb.switch_to_block(cont2);
        let envp = self.envp;
        let mut cargs = Vec::with_capacity(2 + args.len());
        cargs.push(envp);
        cargs.push(d2);
        for a in args {
            cargs.push(self.rv(*a));
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
        self.settle();
        let zz = self.iconst(0);
        self.fb.ins().return_(&[st, zz]);
        self.fb.switch_to_block(okb);
        self.arm();
        self.dv(dst, v);
        let nb = self.next(i);
        self.fb.ins().jump(nb, &[]);
    }

    fn emit_op(&mut self, i: usize, op: &Op) {
        let op = op.clone();
        self.fb.switch_to_block(self.blocks[i]);
        self.gate();
        match op {
            Op::Move { dst, src } => {
                let v = self.rv(src);
                self.dv(dst, v);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::LoadConst { dst, constant } => {
                match constant {
                    Constant::Int(v) => {
                        let c = self.iconst(v);
                        self.dv(dst, c);
                    }
                    Constant::Float(v) => {
                        let c = self.fb.ins().f64const(v);
                        self.dv(dst, c);
                    }
                    Constant::Bool(v) => {
                        let c = self.iconst(v as i64);
                        self.dv(dst, c);
                    }
                    Constant::Null => {
                        let c = self.iconst(0);
                        self.dv(dst, c);
                    }
                    _ => unreachable!("non-scalar const in native body"),
                }
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::AddInt { dst, left, right } => {
                let (a, b) = (self.rv(left), self.rv(right));
                self.checked(i, dst, a, b, false);
            }
            Op::SubInt { dst, left, right } => {
                let (a, b) = (self.rv(left), self.rv(right));
                self.checked(i, dst, a, b, true);
            }
            Op::MultInt { dst, left, right } => {
                let (a, b) = (self.rv(left), self.rv(right));
                let (v, of) = self.fb.ins().smul_overflow(a, b);
                let okb = self.fb.create_block();
                let t = self.err_tramp(ERR_OVFW);
                self.fb.ins().brif(of, t, &[], okb, &[]);
                self.fb.switch_to_block(okb);
                self.dv(dst, v);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::ModInt { dst, left, right } => {
                let (a, b) = (self.rv(left), self.rv(right));
                self.modint(i, dst, a, b);
            }
            Op::AddIntImm { dst, left, val } => {
                let (a, b) = (self.rv(left), self.iconst(val));
                self.checked(i, dst, a, b, false);
            }
            Op::SubIntImm { dst, left, val } => {
                let (a, b) = (self.rv(left), self.iconst(val));
                self.checked(i, dst, a, b, true);
            }
            Op::MultIntImm { dst, left, val } => {
                let (a, b) = (self.rv(left), self.iconst(val));
                let (v, of) = self.fb.ins().smul_overflow(a, b);
                let okb = self.fb.create_block();
                let t = self.err_tramp(ERR_OVFW);
                self.fb.ins().brif(of, t, &[], okb, &[]);
                self.fb.switch_to_block(okb);
                self.dv(dst, v);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::ModIntImm { dst, left, val } => {
                let (a, b) = (self.rv(left), self.iconst(val));
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
                let a = self.rv(left);
                let b = self.rv(right);
                let v = match op {
                    Op::AddFloat { .. } => self.fb.ins().fadd(a, b),
                    Op::SubFloat { .. } => self.fb.ins().fsub(a, b),
                    Op::MultFloat { .. } => self.fb.ins().fmul(a, b),
                    _ => self.fb.ins().fdiv(a, b),
                };
                self.dv(dst, v);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::AddFloatImm { dst, left, val }
            | Op::SubFloatImm { dst, left, val }
            | Op::MultFloatImm { dst, left, val }
            | Op::ModFloatImm { dst, left, val } => {
                let a = self.rv(left);
                let b = self.fb.ins().f64const(f64::from_bits(val as u64));
                let v = match &op {
                    Op::AddFloatImm { .. } => self.fb.ins().fadd(a, b),
                    Op::SubFloatImm { .. } => self.fb.ins().fsub(a, b),
                    Op::MultFloatImm { .. } => self.fb.ins().fmul(a, b),
                    _ => {
                        // ModFloatImm — no frem in clif; unwind to framed
                        let t = self.etrip;
                        self.fb.ins().jump(t, &[]);
                        return;
                    }
                };
                self.dv(dst, v);
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
                    let a = self.rv(left);
                    let b = self.rv(right);
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
                    let a = self.rv(left);
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
                    let a = self.rv(left);
                    let b = self.rv(right);
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
                    let a = self.rv(left);
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
                let c = self.rv(cond);
                let k = self.iconst(is_true as i64);
                let hit = self.fb.ins().icmp(IntCC::Equal, c, k);
                self.bri(i, hit, target, true);
            }
            Op::ForNext { idx, bound, ref target } => {
                let iv = self.rv(idx);
                let bv = self.rv(bound);
                let i2 = self.fb.ins().iadd_imm_s(iv, 1);
                self.dv(idx, i2);
                let hit = self.fb.ins().icmp(IntCC::SignedLessThan, i2, bv);
                self.bri(i, hit, target, true);
            }
            Op::ToFloat { dst, src } => {
                let iv = self.rv(src);
                let f = self.fb.ins().fcvt_from_sint(F64, iv);
                self.dv(dst, f);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::Sqrt { dst, src } => {
                let f = self.rv(src);
                let r = self.fb.ins().sqrt(f);
                self.dv(dst, r);
                let nb = self.next(i);
                self.fb.ins().jump(nb, &[]);
            }
            Op::CallDirect { dst, body, args } => {
                self.ncall(i, dst, body.index() as u32, &args);
            }
            Op::Return { val } => {
                let v = self.rv(val);
                self.settle();
                let s0 = self.fb.ins().iconst(I8, ST_OK);
                self.fb.ins().return_(&[s0, v]);
            }
            Op::Panic {} => {
                let t = self.err_tramp(ERR_PANIC);
                self.fb.ins().jump(t, &[]);
            }
            _ => unreachable!("non-whitelist op in native body"),
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
    kinds: &[Option<NK>],
    lyt: &Layout,
    fbc: &mut FunctionBuilderContext,
    ctx: &mut Context,
) -> Result<(), String> {
    module.clear_context(ctx);
    let chunk = &prog.chunks[compile::BodyId::from(body as u32)];
    ctx.func.signature = native_sig(module, chunk.params.len());

    // same alias-region indices as emit.rs (state=2 used here)
    for (id, desc) in [(0u32, "reg window"), (1, "gc heap"), (2, "vm state")] {
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

    let ops = prog.ops(compile::BodyId::from(body as u32));
    let off2idx: HashMap<usize, usize> =
        ops.iter().enumerate().map(|(i, (o, _))| (*o, i)).collect();
    let blocks: Vec<Block> = (0..ops.len()).map(|_| fb.create_block()).collect();

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

    let etrip = fb.create_block();
    fb.set_cold_block(etrip);
    let mut em = Ne {
        fb,
        lyt,
        hrefs: &hrefs,
        nrefs: &nrefs,
        vars,
        blocks,
        off2idx,
        ops,
        envp,
        depth,
        bcn,
        bcn0,
        etrip,
        errs: Vec::new(),
    };
    em.arm();
    let first = em.blocks[0];
    em.fb.ins().jump(first, &[]);
    em.fb.seal_block(entry);

    let nops = em.ops.len();
    for i in 0..nops {
        let op = em.ops[i].1.clone();
        em.emit_op(i, &op);
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
    module
        .define_function(native_ids[body].unwrap(), ctx)
        .map_err(|e| format!("native body {body}: {e}"))?;
    Ok(())
}

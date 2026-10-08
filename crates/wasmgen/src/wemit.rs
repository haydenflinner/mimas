//! Spike: translate mimas `compile::Ir` bodies into waffle `FunctionBody`s and
//! let waffle's reducify/stackify/localify backend emit structured wasm.
//!
//! The goal is to prove that the fragile machinery in `irgen` — DFS layout,
//! scope construction, repair, `topo_relax` — can be replaced wholesale by
//! waffle's fuzzed backend. Mimas locals become SSA by threading one i64
//! blockparam per local through every CFG edge (max-SSA; `localify` prunes
//! it); `Phi` insts become real blockparams; `JumpIfFalse`/`ForNext` +
//! companion `Jump` pairs become `CondBr`s.
//!
//! All values travel as `i64` words — analysis' class bits only pick operator
//! semantics (F64 vs I64 vs bool compares). Heap/dict/str/format/closure insts
//! bail for now: this spike is about control flow and SSA, not repr coverage.

use std::collections::{HashMap, HashSet};

use compile::{
    BinOp, BlockId, Body as IrBody, Constant, Inst, InstId, Ir, OperandKind, UnaryOp,
};
use waffle::{
    Block as WBlock, BlockTarget, Export, ExportKind, Func, FuncDecl, FunctionBody, Import,
    ImportKind, MemoryData, Module, Operator as WOp, Signature, SignatureData, Terminator,
    Type as WTy, Value,
    entity::EntityRef,
};

use crate::K;
use crate::irgen::analyze_body;

macro_rules! bail {
    ($($a:tt)*) => { return Err(format!($($a)*)) };
}

/// memory base for `GetEntry`/`SetEntry` slots — page 1, 8 bytes per local.
const EBASE: u64 = 65536;

pub struct WEmit {
    pub bytes: Vec<u8>,
    /// body idx -> wasm func index
    pub emitted: Vec<usize>,
    pub skipped: Vec<(usize, String)>,
}

/// body-block successors mirroring codegen's convention: `Jump` is the
/// fallthrough (unconditional edge), `JumpIfFalse`/`ForNext`/`Switch` are the
/// branch edges.
pub(crate) fn succs(body: &IrBody, bid: BlockId) -> (Option<BlockId>, Vec<BlockId>) {
    let mut ft = None;
    let mut brs = Vec::new();
    for &iid in &body.blocks[bid].stream {
        match &body.instructions[iid] {
            Inst::Jump { target } => ft = Some(*target),
            Inst::JumpIfFalse { target, .. } | Inst::ForNext { target, .. } => brs.push(*target),
            Inst::Switch {
                table, default, ..
            } => {
                brs.extend(table.iter().copied());
                brs.push(*default);
            }
            _ => {}
        }
    }
    (ft, brs)
}

/// fallthrough-first DFS from entry — matches codegen's serialization order.
pub(crate) fn dfs_order(body: &IrBody) -> Vec<BlockId> {
    let mut order = Vec::new();
    let mut placed = vec![false; body.blocks.len()];
    let mut stack = vec![BlockId::ZERO];
    while let Some(b) = stack.pop() {
        if placed[b.index()] {
            continue;
        }
        placed[b.index()] = true;
        order.push(b);
        let (ft, brs) = succs(body, b);
        for t in brs {
            stack.push(t);
        }
        if let Some(f) = ft {
            stack.push(f);
        }
    }
    order
}

fn sig_i64(module: &mut Module, sigs: &mut HashMap<usize, Signature>, n: usize) -> Signature {
    *sigs.entry(n).or_insert_with(|| {
        module.signatures.push(SignatureData {
            params: vec![WTy::I64; n],
            returns: vec![WTy::I64],
        })
    })
}

pub fn emit_waffle(ir: &Ir) -> WEmit {
    emit_waffle_stub(ir, false)
}

/// `stub`: uncovered value insts emit `i64.const 0` instead of skipping the
/// body — used to prove the *control-flow* question (does waffle's backend
/// emit the bodies our layout machinery can't?) independent of inst coverage.
/// Modules built this way validate and run but their heap/field/str reads are
/// wrong — a layout probe, not a real backend.
pub fn emit_waffle_stub(ir: &Ir, stub: bool) -> WEmit {
    let mut module = Module::empty();
    let mut sigs: HashMap<usize, Signature> = HashMap::new();
    let mut skipped = Vec::new();

    // natives -> imports, one per (id, arity) — name matches the host ABI
    // `n{id}_{param-classes}_{ret}` with all-word channels for the spike.
    let mut natives: HashMap<(u32, usize), Func> = HashMap::new();
    for (_, body) in ir.bodies.iter() {
        for (_, block) in body.blocks.iter() {
            for &iid in &block.stream {
                if let Inst::CallNative { id, args } = &body.instructions[iid] {
                    natives.entry((id.index() as u32, args.len())).or_insert_with(|| {
                        let sig = sig_i64(&mut module, &mut sigs, args.len());
                        let name = format!("n{}_{}_w", id.index(), "w".repeat(args.len()));
                        let f = Func::new(module.funcs.len());
                        module.funcs.push(FuncDecl::Import(sig, name.clone()));
                        module.imports.push(Import {
                            module: "env".into(),
                            name,
                            kind: ImportKind::Func(f),
                        });
                        f
                    });
                }
            }
        }
    }

    // body idx -> wasm func index (after imports). A stub FunctionBody holds
    // the slot so call targets stay valid even when a body skips; stubs that
    // remain trap at runtime (Unreachable) rather than silently miscompiling.
    let mut funcs: HashMap<usize, Func> = HashMap::new();
    for (bid, body) in ir.bodies.iter() {
        let sig = sig_i64(&mut module, &mut sigs, body.params.len());
        let mut stub = FunctionBody::new(&module, sig);
        stub.set_terminator(stub.entry, Terminator::Unreachable);
        let f = Func::new(module.funcs.len());
        module
            .funcs
            .push(FuncDecl::Body(sig, format!("b{}", bid.index()), stub));
        funcs.insert(bid.index(), f);
    }

    // entry-shared locals: anything a non-entry body reaches via {Get,Set}Entry
    let mut shared: HashSet<u32> = HashSet::new();
    for (_, body) in ir.bodies.iter() {
        for (_, block) in body.blocks.iter() {
            for &iid in &block.stream {
                match &body.instructions[iid] {
                    Inst::GetEntry(l) | Inst::SetEntry(l, _) => {
                        shared.insert(l.index() as u32);
                    }
                    _ => {}
                }
            }
        }
    }

    module.memories.push(MemoryData {
        initial_pages: 4,
        maximum_pages: None,
        segments: vec![],
    });

    // ret[] is indexed by callee body — one slot per body, all unknown: call
    // results come back Word, which is fine in an all-i64 pipeline.
    let no_ret: Vec<Option<K>> = vec![None; ir.bodies.len()];
    let mut emitted = Vec::new();
    for (bid, body) in ir.bodies.iter() {
        let sig = sig_i64(&mut module, &mut sigs, body.params.len());
        match wbody(
            &module,
            sig,
            ir,
            bid.index(),
            &funcs,
            &natives,
            &shared,
            &no_ret,
            stub,
        ) {
            Ok(fb) => {
                let f = funcs[&bid.index()];
                module.funcs[f] = FuncDecl::Body(sig, format!("b{}", bid.index()), fb);
                module.exports.push(Export {
                    name: format!("b{}", bid.index()),
                    kind: ExportKind::Func(f),
                });
                emitted.push(bid.index());
            }
            Err(e) => skipped.push((bid.index(), e)),
        }
    }
    module.exports.push(Export {
        name: "mem".into(),
        kind: ExportKind::Memory(waffle::Memory::new(0)),
    });

    match module.to_wasm_bytes() {
        Ok(bytes) => WEmit {
            bytes,
            emitted,
            skipped,
        },
        Err(e) => WEmit {
            bytes: vec![],
            emitted,
            skipped: vec![(usize::MAX, format!("waffle backend: {e:?}"))],
        },
    }
}

struct WB<'a> {
    body: &'a IrBody,
    fb: FunctionBody,
    /// mimas BlockId -> waffle Block
    wb: Vec<WBlock>,
    /// Phi inst -> its blockparam value
    phi_val: HashMap<u32, Value>,
    /// block -> local blockparam values (indexed by local)
    local_bp: Vec<Vec<Value>>,
    /// inst -> produced value
    iv: HashMap<u32, Value>,
    /// local -> current SSA value at this point
    cur: Vec<Value>,
    /// phi insts per block, in stream order
    phis: Vec<Vec<InstId>>,
    ana: crate::irgen::AnaI,
    funcs: &'a HashMap<usize, Func>,
    natives: &'a HashMap<(u32, usize), Func>,
    shared: &'a HashSet<u32>,
    is_entry: bool,
    stub: bool,
    order_next: HashMap<usize, usize>,
}

impl<'a> WB<'a> {
    fn op(&mut self, w: WBlock, op: WOp, args: &[Value], tys: &[WTy]) -> Value {
        self.fb.add_op(w, op, args, tys)
    }

    fn get(&self, iid: InstId) -> Result<Value, String> {
        self.iv
            .get(&(iid.index() as u32))
            .copied()
            .ok_or_else(|| format!("inst {} unemitted", iid.index()))
    }

    /// i64 const
    fn konst(&mut self, w: WBlock, v: u64) -> Value {
        self.op(w, WOp::I64Const { value: v }, &[], &[WTy::I64])
    }

    /// i32 const — memory addresses are i32 on wasm32
    fn konst32(&mut self, w: WBlock, v: u32) -> Value {
        self.op(w, WOp::I32Const { value: v }, &[], &[WTy::I32])
    }

    fn eaddr(&mut self, w: WBlock, l: u32) -> Value {
        self.konst32(w, (EBASE + 8 * l as u64) as u32)
    }

    /// edge args for `target`: one value per phi (from this pred's branch
    /// entries) then one per local.
    fn targs(&mut self, w: WBlock, pred: usize, target: usize) -> Result<Vec<Value>, String> {
        let mut args = Vec::new();
        for &piid in &self.phis[target].clone() {
            let Inst::Phi(branches) = &self.body.instructions[piid] else {
                unreachable!()
            };
            let v = match branches.iter().find(|(b, _)| b.index() == pred) {
                Some((_, i)) => self.get(*i)?,
                None => self.konst(w, 0),
            };
            args.push(v);
        }
        args.extend_from_slice(&self.cur);
        Ok(args)
    }

    fn tgt(&mut self, w: WBlock, pred: usize, target: usize) -> Result<BlockTarget, String> {
        Ok(BlockTarget {
            block: self.wb[target],
            args: self.targs(w, pred, target)?,
        })
    }

    /// Emit the current local values as an entry-slot memory write for
    /// entry-shared locals (mirrors `emit_ir`'s spill-on-return).
    fn store_entry(&mut self, w: WBlock, l: u32, v: Value) {
        let a = self.eaddr(w, l);
        self.op(
            w,
            WOp::I64Store {
                memory: waffle::MemoryArg {
                    align: 3,
                    offset: 0,
                    memory: waffle::Memory::new(0),
                },
            },
            &[a, v],
            &[],
        );
    }

    fn mirror_shared(&mut self, w: WBlock, l: u32, v: Value) {
        if self.is_entry && self.shared.contains(&l) {
            self.store_entry(w, l, v);
        }
    }

    fn emit_inst(&mut self, w: WBlock, iid: InstId) -> Result<(), String> {
        let i = iid.index() as u32;
        match &self.body.instructions[iid] {
            Inst::Constant(c) => {
                let v = match c {
                    Constant::Int(n) => self.konst(w, *n as u64),
                    Constant::Float(f) => self.konst(w, f.to_bits()),
                    Constant::Bool(b) => self.konst(w, *b as u64),
                    Constant::Null => self.konst(w, 0),
                    Constant::Str(_) | Constant::Array(_) => {
                        if self.stub {
                            self.konst(w, 0)
                        } else {
                            bail!("const {c:?} needs heap objects")
                        }
                    }
                };
                self.iv.insert(i, v);
            }
            Inst::GetLocal(l) => {
                self.iv.insert(i, self.cur[l.index()]);
            }
            Inst::SetLocal(l, v) => {
                let val = self.get(*v)?;
                self.cur[l.index()] = val;
                self.mirror_shared(w, l.index() as u32, val);
            }
            Inst::GetEntry(l) => {
                let a = self.eaddr(w, l.index() as u32);
                let v = self.op(
                    w,
                    WOp::I64Load {
                        memory: waffle::MemoryArg {
                            align: 3,
                            offset: 0,
                            memory: waffle::Memory::new(0),
                        },
                    },
                    &[a],
                    &[WTy::I64],
                );
                self.iv.insert(i, v);
            }
            Inst::SetEntry(l, v) => {
                let val = self.get(*v)?;
                self.store_entry(w, l.index() as u32, val);
            }
            Inst::Phi(_) => {
                let v = *self.phi_val.get(&i).ok_or("phi without blockparam")?;
                self.iv.insert(i, v);
            }
            Inst::BinOp {
                left,
                op,
                right,
                kind,
            } => {
                let (l, r) = (self.get(*left)?, self.get(*right)?);
                let v = self.binop(w, *op, *kind, l, r)?;
                self.iv.insert(i, v);
            }
            Inst::UnaryOp { op, right } => {
                let r = self.get(*right)?;
                let v = match op {
                    UnaryOp::Negative => {
                        if self.ana.class.get(&i) == Some(&K::Float) {
                            let f = self.op(w, WOp::F64ReinterpretI64, &[r], &[WTy::F64]);
                            let n = self.op(w, WOp::F64Neg, &[f], &[WTy::F64]);
                            self.op(w, WOp::I64ReinterpretF64, &[n], &[WTy::I64])
                        } else {
                            let z = self.konst(w, 0);
                            self.op(w, WOp::I64Sub, &[z, r], &[WTy::I64])
                        }
                    }
                    UnaryOp::Positive => r,
                    UnaryOp::Not => {
                        let c = self.op(w, WOp::I64Eqz, &[r], &[WTy::I32]);
                        self.op(w, WOp::I64ExtendI32U, &[c], &[WTy::I64])
                    }
                    UnaryOp::BitwiseNot => {
                        let m = self.konst(w, u64::MAX);
                        self.op(w, WOp::I64Xor, &[r, m], &[WTy::I64])
                    }
                };
                self.iv.insert(i, v);
            }
            Inst::ToFloat(v) => {
                let x = self.get(*v)?;
                let f = self.op(w, WOp::F64ConvertI64S, &[x], &[WTy::F64]);
                let b = self.op(w, WOp::I64ReinterpretF64, &[f], &[WTy::I64]);
                self.iv.insert(i, b);
            }
            Inst::Sqrt(v) => {
                let x = self.get(*v)?;
                let f = self.op(w, WOp::F64ReinterpretI64, &[x], &[WTy::F64]);
                let s = self.op(w, WOp::F64Sqrt, &[f], &[WTy::F64]);
                let b = self.op(w, WOp::I64ReinterpretF64, &[s], &[WTy::I64]);
                self.iv.insert(i, b);
            }
            Inst::CallDirect { body: b2, args } => {
                let f = *self
                    .funcs
                    .get(&b2.index())
                    .ok_or_else(|| format!("callee body {} unemitted", b2.index()))?;
                let args = args
                    .iter()
                    .map(|a| self.get(*a))
                    .collect::<Result<Vec<_>, _>>()?;
                let v = self.op(w, WOp::Call { function_index: f }, &args, &[WTy::I64]);
                self.iv.insert(i, v);
            }
            Inst::Call { callee, args } => {
                let bx = self.ana.callee.get(&i).copied().or_else(|| {
                    if let Inst::RefBody(b2) = &self.body.instructions[*callee] {
                        Some(b2.index())
                    } else {
                        None
                    }
                });
                let Some(bx) = bx else {
                    if self.stub {
                        let z = self.konst(w, 0);
                        self.iv.insert(i, z);
                        return Ok(());
                    }
                    bail!("dynamic call unresolved")
                };
                let f = *self
                    .funcs
                    .get(&bx)
                    .ok_or_else(|| format!("callee body {bx} unemitted"))?;
                let args = args
                    .iter()
                    .map(|a| self.get(*a))
                    .collect::<Result<Vec<_>, _>>()?;
                let v = self.op(w, WOp::Call { function_index: f }, &args, &[WTy::I64]);
                self.iv.insert(i, v);
            }
            Inst::CallNative { id, args } => {
                let f = *self
                    .natives
                    .get(&(id.index() as u32, args.len()))
                    .ok_or("native import missing")?;
                let args = args
                    .iter()
                    .map(|a| self.get(*a))
                    .collect::<Result<Vec<_>, _>>()?;
                let v = self.op(w, WOp::Call { function_index: f }, &args, &[WTy::I64]);
                self.iv.insert(i, v);
            }
            // spike: Unwrap is a passthrough — the `v <= 0` trap needs a
            // conditional branch, which means splitting blocks; deferred.
            Inst::Unwrap(v) | Inst::UnwrapUnit(v) => {
                let x = self.get(*v)?;
                self.iv.insert(i, x);
            }
            other => {
                if self.stub {
                    let z = self.konst(w, 0);
                    self.iv.insert(i, z);
                } else {
                    bail!("unsupported inst: {other:?}")
                }
            }
        }
        Ok(())
    }

    fn binop(
        &mut self,
        w: WBlock,
        op: BinOp,
        kind: OperandKind,
        l: Value,
        r: Value,
    ) -> Result<Value, String> {
        use BinOp::*;
        let cmp = |me: &mut Self, op: WOp| {
            let c = me.op(w, op, &[l, r], &[WTy::I32]);
            me.op(w, WOp::I64ExtendI32U, &[c], &[WTy::I64])
        };
        Ok(match kind {
            OperandKind::Int | OperandKind::Bool => match op {
                Add => self.op(w, WOp::I64Add, &[l, r], &[WTy::I64]),
                Sub => self.op(w, WOp::I64Sub, &[l, r], &[WTy::I64]),
                Mult => self.op(w, WOp::I64Mul, &[l, r], &[WTy::I64]),
                // `/` on ints is float division in mimas — numeric convert,
                // not bit reinterpret
                Div => {
                    let fl = self.op(w, WOp::F64ConvertI64S, &[l], &[WTy::F64]);
                    let fr = self.op(w, WOp::F64ConvertI64S, &[r], &[WTy::F64]);
                    let d = self.op(w, WOp::F64Div, &[fl, fr], &[WTy::F64]);
                    self.op(w, WOp::I64ReinterpretF64, &[d], &[WTy::I64])
                }
                IDiv => self.op(w, WOp::I64DivS, &[l, r], &[WTy::I64]),
                Mod => self.op(w, WOp::I64RemS, &[l, r], &[WTy::I64]),
                And | BitAnd => self.op(w, WOp::I64And, &[l, r], &[WTy::I64]),
                Or | BitOr => self.op(w, WOp::I64Or, &[l, r], &[WTy::I64]),
                Xor | BitXor => self.op(w, WOp::I64Xor, &[l, r], &[WTy::I64]),
                BitShiftLeft => self.op(w, WOp::I64Shl, &[l, r], &[WTy::I64]),
                BitShiftRight => self.op(w, WOp::I64ShrS, &[l, r], &[WTy::I64]),
                Identity => cmp(self, WOp::I64Eq),
                NotEqual => cmp(self, WOp::I64Ne),
                LessThan => cmp(self, WOp::I64LtS),
                LessEqual => cmp(self, WOp::I64LeS),
                GreaterThan => cmp(self, WOp::I64GtS),
                GreaterEqual => cmp(self, WOp::I64GeS),
                Coalesce => bail!("coalesce needs word null semantics"),
            },
            OperandKind::Float => {
                let fl = self.op(w, WOp::F64ReinterpretI64, &[l], &[WTy::F64]);
                let fr = self.op(w, WOp::F64ReinterpretI64, &[r], &[WTy::F64]);
                match op {
                    Add => {
                        let v = self.op(w, WOp::F64Add, &[fl, fr], &[WTy::F64]);
                        self.op(w, WOp::I64ReinterpretF64, &[v], &[WTy::I64])
                    }
                    Sub => {
                        let v = self.op(w, WOp::F64Sub, &[fl, fr], &[WTy::F64]);
                        self.op(w, WOp::I64ReinterpretF64, &[v], &[WTy::I64])
                    }
                    Mult => {
                        let v = self.op(w, WOp::F64Mul, &[fl, fr], &[WTy::F64]);
                        self.op(w, WOp::I64ReinterpretF64, &[v], &[WTy::I64])
                    }
                    Div => {
                        let v = self.op(w, WOp::F64Div, &[fl, fr], &[WTy::F64]);
                        self.op(w, WOp::I64ReinterpretF64, &[v], &[WTy::I64])
                    }
                    Identity => {
                        let c = self.op(w, WOp::F64Eq, &[fl, fr], &[WTy::I32]);
                        self.op(w, WOp::I64ExtendI32U, &[c], &[WTy::I64])
                    }
                    NotEqual => {
                        let c = self.op(w, WOp::F64Ne, &[fl, fr], &[WTy::I32]);
                        self.op(w, WOp::I64ExtendI32U, &[c], &[WTy::I64])
                    }
                    LessThan => {
                        let c = self.op(w, WOp::F64Lt, &[fl, fr], &[WTy::I32]);
                        self.op(w, WOp::I64ExtendI32U, &[c], &[WTy::I64])
                    }
                    LessEqual => {
                        let c = self.op(w, WOp::F64Le, &[fl, fr], &[WTy::I32]);
                        self.op(w, WOp::I64ExtendI32U, &[c], &[WTy::I64])
                    }
                    GreaterThan => {
                        let c = self.op(w, WOp::F64Gt, &[fl, fr], &[WTy::I32]);
                        self.op(w, WOp::I64ExtendI32U, &[c], &[WTy::I64])
                    }
                    GreaterEqual => {
                        let c = self.op(w, WOp::F64Ge, &[fl, fr], &[WTy::I32]);
                        self.op(w, WOp::I64ExtendI32U, &[c], &[WTy::I64])
                    }
                    _ => bail!("float binop {op:?} unsupported in spike"),
                }
            }
            _ => {
                if self.stub {
                    self.konst(w, 0)
                } else {
                    bail!("binop kind {kind:?} needs tagged-value ops")
                }
            }
        })
    }

}

fn wbody(
    module: &Module,
    sig: Signature,
    ir: &Ir,
    bid: usize,
    funcs: &HashMap<usize, Func>,
    natives: &HashMap<(u32, usize), Func>,
    shared: &HashSet<u32>,
    ret: &[Option<K>],
    stub: bool,
) -> Result<FunctionBody, String> {
    let body = &ir.bodies[compile::BodyId::from(bid as u32)];
    let ana = analyze_body(body, ret)?;
    let nl = body.locals.len();
    let mut fb = FunctionBody::new(module, sig);
    let entry = fb.entry;

    let mut wb: Vec<WBlock> = Vec::with_capacity(body.blocks.len());
    for (b, _) in body.blocks.iter() {
        wb.push(if b.index() == 0 { entry } else { fb.add_block() });
    }

    // blockparams: phis first (stream order), then one i64 per local
    let mut phi_val: HashMap<u32, Value> = HashMap::new();
    let mut phis: Vec<Vec<InstId>> = vec![vec![]; body.blocks.len()];
    let mut local_bp: Vec<Vec<Value>> = vec![vec![]; body.blocks.len()];
    for (b, blk) in body.blocks.iter() {
        if b.index() == 0 {
            continue;
        }
        for &iid in &blk.stream {
            if let Inst::Phi(_) = &body.instructions[iid] {
                phis[b.index()].push(iid);
                phi_val.insert(
                    iid.index() as u32,
                    fb.add_blockparam(wb[b.index()], WTy::I64),
                );
            }
        }
        for _ in 0..nl {
            local_bp[b.index()].push(fb.add_blockparam(wb[b.index()], WTy::I64));
        }
    }

    // "next block" fallback map for empty/dead blocks
    let order = dfs_order(body);
    let mut order_next: HashMap<usize, usize> = HashMap::new();
    for wpair in order.windows(2) {
        order_next.insert(wpair[0].index(), wpair[1].index());
    }

    let mut cx = WB {
        body,
        fb,
        wb,
        phi_val,
        local_bp,
        iv: HashMap::new(),
        cur: vec![],
        phis,
        ana,
        funcs,
        natives,
        shared,
        is_entry: bid == 0,
        stub,
        order_next,
    };

    // process blocks in DFS order (preds before succs where the CFG allows)
    for &b in &order {
        let bx = b.index();
        let w = cx.wb[bx];
        let blk = &body.blocks[b];
        cx.cur = if bx == 0 {
            let mut c = vec![];
            for (l, _) in body.locals.iter() {
                if let Some(pi) = body.params.iter().position(|&p| p == l) {
                    c.push(cx.fb.blocks[entry].params[pi].1);
                } else {
                    let z = cx.konst(w, 0);
                    c.push(z);
                }
            }
            c
        } else {
            cx.local_bp[bx].clone()
        };

        // terminator bookkeeping: mimas blocks end with an explicit
        // Jump/Return/Switch, or a JumpIfFalse/ForNext + companion Jump.
        let mut jump: Option<BlockId> = None;
        let mut condbr: Option<(InstId, BlockId)> = None;
        let mut fornext: Option<(InstId, InstId, BlockId)> = None;
        let mut switch: Option<(InstId, u32, Vec<BlockId>, BlockId)> = None;
        let mut ret: Option<InstId> = None;
        let mut panicked = false;

        for &iid in &blk.stream {
            match &body.instructions[iid] {
                Inst::Jump { target } => jump = Some(*target),
                Inst::JumpIfFalse { condition, target } => {
                    condbr = Some((*condition, *target))
                }
                Inst::ForNext { idx, bound, target } => {
                    fornext = Some((*idx, *bound, *target))
                }
                Inst::Switch {
                    scrut,
                    base,
                    table,
                    default,
                } => switch = Some((*scrut, *base, table.clone(), *default)),
                Inst::Return(v) => {
                    ret = Some(*v);
                    break;
                }
                Inst::Panic => {
                    cx.fb.set_terminator(w, Terminator::Unreachable);
                    panicked = true;
                    break;
                }
                _ => cx.emit_inst(w, iid)?,
            }
        }

        if panicked {
            continue;
        }
        if let Some(v) = ret {
            let val = cx.get(v)?;
            cx.fb.set_terminator(w, Terminator::Return { values: vec![val] });
            continue;
        }
        if let Some((scrut, base, table, default)) = switch {
            let mut s = cx.get(scrut)?;
            if base != 0 {
                let k = cx.konst(w, base as u64);
                s = cx.op(w, WOp::I64Sub, &[s, k], &[WTy::I64]);
            }
            let s32 = cx.op(w, WOp::I32WrapI64, &[s], &[WTy::I32]);
            let targets = table
                .iter()
                .map(|t| cx.tgt(w, bx, t.index()))
                .collect::<Result<Vec<_>, _>>()?;
            let d = cx.tgt(w, bx, default.index())?;
            cx.fb.set_terminator(
                w,
                Terminator::Select {
                    value: s32,
                    targets,
                    default: d,
                },
            );
            continue;
        }
        if let Some((idx_iid, bound_iid, t)) = fornext {
            let Some(jt) = jump else {
                bail!("for_next without companion jump")
            };
            let Inst::GetLocal(l) = &body.instructions[idx_iid] else {
                bail!("for_next idx not a local read")
            };
            let one = cx.konst(w, 1);
            let i1 = cx.op(w, WOp::I64Add, &[cx.cur[l.index()], one], &[WTy::I64]);
            cx.cur[l.index()] = i1;
            let bv = cx.get(bound_iid)?;
            let c = cx.op(w, WOp::I64LtS, &[i1, bv], &[WTy::I32]);
            let tt = cx.tgt(w, bx, t.index())?;
            let ft = cx.tgt(w, bx, jt.index())?;
            cx.fb.set_terminator(
                w,
                Terminator::CondBr {
                    cond: c,
                    if_true: tt,
                    if_false: ft,
                },
            );
            continue;
        }
        if let Some((cond, t)) = condbr {
            let Some(jt) = jump else {
                bail!("jump_if_false without companion jump")
            };
            let cv = cx.get(cond)?;
            let c32 = cx.op(w, WOp::I32WrapI64, &[cv], &[WTy::I32]);
            let tt = cx.tgt(w, bx, jt.index())?;
            let ft = cx.tgt(w, bx, t.index())?;
            cx.fb.set_terminator(
                w,
                Terminator::CondBr {
                    cond: c32,
                    if_true: tt,
                    if_false: ft,
                },
            );
            continue;
        }
        if let Some(t) = jump {
            let tt = cx.tgt(w, bx, t.index())?;
            cx.fb.set_terminator(w, Terminator::Br { target: tt });
            continue;
        }
        // no terminator inst — dead tail or DCE gap
        if let Some(&nx) = cx.order_next.get(&bx) {
            let tt = cx.tgt(w, bx, nx)?;
            cx.fb.set_terminator(w, Terminator::Br { target: tt });
        } else {
            let z = cx.konst(w, 0);
            cx.fb.set_terminator(w, Terminator::Return { values: vec![z] });
        }
    }

    // blocks unreachable from entry still need a valid terminator
    let done: HashSet<usize> = order.iter().map(|b| b.index()).collect();
    for (b, _) in body.blocks.iter() {
        if done.contains(&b.index()) {
            continue;
        }
        let w = cx.wb[b.index()];
        cx.fb.set_terminator(w, Terminator::Unreachable);
    }

    cx.fb.recompute_edges();
    Ok(cx.fb)
}

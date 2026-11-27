use shared::IdVec;

use colored::Colorize;
use shared::{FileId, Location, Ty};
use solve::{
    ResolvedDeclKind, ResolvedModule,
    components::{AdtId, DecId, Vis},
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    BinOp, BlockId, BlockTarget, BodyId, Constant, Function, Inst, InstId, Ir, Local, Module, Op,
    OperandKind, UnaryOp,
    codegen::{
        bodycache,
        clean::{self, uses},
        opt,
        program::{Chunk, Encoder, Program},
    },
    ir::Body,
};

shared::id!(pub Reg);

#[derive(Debug)]
pub struct Compiler {
    pub(crate) ops: Vec<Op>,
    pub(crate) chunks: IdVec<BodyId, Chunk>,
    disasm: bool,
    /// Keep the debugger's view of dissolved struct locals (`Inst::Shadow`
    /// copies, see `codegen::opt`): on for a VM a step debugger will read,
    /// off for a program that only runs -- the copies cost the interpreter.
    shadows: bool,
    srcs: HashMap<FileId, Arc<str>>,
    /// Natives that read the live call stack: a body calling one is never
    /// inlined, so its frame stays real. See `opt`.
    frame_natives: std::collections::HashSet<api::NativeId>,
    /// Natives that mutate their receiver: an adt that reaches one is not a
    /// value type. See `opt::value_adts`.
    mutating_natives: std::collections::HashSet<api::NativeId>,
}

// fuse a comparison op + JumpIfFalse into its `B`-prefixed branch form. reg-reg comparisons carry
// `left, right`; immediate comparisons carry `left, val` -- list them in the two groups.
macro_rules! fuse_branch {
    ($cmp:expr, $target:expr;
     $($rfrom:ident => $rto:ident),* ;
     $($ifrom:ident => $ito:ident),* $(,)?) => {
        match $cmp {
            $(Op::$rfrom { left, right, .. } => Op::$rto { target: $target, left, right, is_true: false },)*
            $(Op::$ifrom { left, val, .. } => Op::$ito { target: $target, left, val, is_true: false },)*
            _ => unreachable!("incorrectly marked an op as a branchable comp"),
        }
    };
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            chunks: IdVec::new(),
            disasm: false,
            shadows: false,
            srcs: HashMap::new(),
            frame_natives: Default::default(),
            mutating_natives: Default::default(),
        }
    }
    /// Natives that mutate their receiver (`mutates_recv`).
    pub fn with_mutating_natives(mut self, ids: impl IntoIterator<Item = api::NativeId>) -> Self {
        self.mutating_natives = ids.into_iter().collect();
        self
    }
    /// Natives that inspect the call stack (`std::debug::*`): bodies that
    /// call them keep their frames.
    pub fn with_frame_natives(mut self, ids: impl IntoIterator<Item = api::NativeId>) -> Self {
        self.frame_natives = ids.into_iter().collect();
        self
    }

    /// Emit the debugger's shadow copies of dissolved struct locals.
    pub fn with_shadows(mut self, on: bool) -> Self {
        self.shadows = on;
        self
    }

    pub fn with_disasm(mut self, on: bool) -> Self {
        self.disasm = on;
        self
    }

    /// Source text per file, used only to interleave source lines into the `--dump-bytes` disasm.
    pub fn with_sources(mut self, srcs: HashMap<FileId, Arc<str>>) -> Self {
        self.srcs = srcs;
        self
    }

    pub fn compile(&mut self, ir: &mut Ir) -> Program {
        let mut bytes = Encoder::new();

        // body -> source name, for the --disasm dump only
        let names: HashMap<BodyId, String> = if self.disasm {
            ir.item_bodies
                .iter()
                .map(|(&dec, &body)| (body, ir.resolutions.decs[dec].name.clone()))
                .collect()
        } else {
            HashMap::new()
        };

        /// A dec the host can call: everything but a native, which has no body of its own. Hands
        /// back the `Function` already built for `signatures` instead of building a second one, so
        /// the two views of the same dec can't drift apart.
        fn exported_fn(
            ir: &Ir,
            signatures: &IdVec<BodyId, Option<Function>>,
            dec: DecId,
        ) -> Option<Function> {
            let body = *ir.item_bodies.get(&dec)?;
            signatures.get(body)?.clone()
        }

        fn export(
            ir: &mut Ir,
            signatures: &IdVec<BodyId, Option<Function>>,
            module: ResolvedModule,
        ) -> Module {
            let mut out = Module::default();
            for (name, dec) in module.items {
                // a law's generic function has no body (see `is_generic_law`)
                // a page region's item also exports as `region::name` (C2)
                let base = name.rsplit_once("::").map_or(name.as_str(), |(_, b)| b);
                if crate::ir::is_generic_law(base) {
                    continue;
                }
                // a prelude decl resolves in the root scope like the file's
                // own but is not the program's to export: the host's `root`
                // lists the file's items only.
                if ir.resolutions.decs[dec].location.file_id == shared::PRELUDE_FILE {
                    continue;
                }
                match ir.resolutions.decs[dec].kind.clone() {
                    ResolvedDeclKind::Item { .. } => {
                        let f = exported_fn(ir, signatures, dec)
                            .unwrap_or_else(|| unreachable!("item `{name}` is not a callable fn"));
                        out.functions.insert(name, f);
                    }
                    // dicts and structs (and arrays holding them) have no constant
                    // form, see `Constant::emit_const_dec`
                    ResolvedDeclKind::Constant(lit) if !Constant::needs_alloc(ir, &lit) => {
                        out.constants.insert(name, Constant::from_literal(ir, lit));
                    }
                    ResolvedDeclKind::Adt(adt) => {
                        let resolved = &ir.resolutions.adts[adt];
                        let fields = resolved.fields.clone();
                        // the solver keeps impls in a `HashMap`, whose order is randomized per
                        // process -- sort so the export reads the same on every run
                        let mut methods: Vec<(String, DecId)> = resolved
                            .methods
                            .iter()
                            .map(|(name, &dec)| (name.clone(), dec))
                            .collect();
                        methods.sort_by(|(a, _), (b, _)| a.cmp(b));
                        let methods = methods
                            .into_iter()
                            .filter_map(|(name, dec)| {
                                Some((name, exported_fn(ir, signatures, dec)?))
                            })
                            .collect();
                        let implements = resolved
                            .implements
                            .iter()
                            .map(|p| ir.resolutions.pact_names[*p].clone())
                            .collect();
                        out.types.insert(
                            name,
                            crate::Type {
                                vis: ir.resolutions.decs[dec].vis,
                                fields,
                                methods,
                                implements,
                            },
                        );
                    }
                    _ => {}
                }
            }
            for (name, module) in module.modules {
                out.modules.insert(name, export(ir, signatures, module));
            }
            out
        }
        // every body's own signature keyed by body rather than by dec so a function value can be
        // called from a `BodyId` alone. a closure's defaults are never reachable from a call site
        // that only has the value, so it reports none -- same as what a dynamic call lowers to.
        let mut signatures: IdVec<BodyId, Option<Function>> = vec![None; ir.bodies.len()].into();
        // read rather than drained -- `export` still needs `item_bodies` to find each dec's body
        let item_bodies: Vec<(DecId, BodyId)> = ir
            .item_bodies
            .iter()
            .map(|(&dec, &body)| (dec, body))
            .collect();
        for (dec, body) in item_bodies {
            let ResolvedDeclKind::Item { defaults, .. } = ir.resolutions.decs[dec].kind.clone()
            else {
                continue;
            };
            let Ty::Fn(header) = ir.resolutions.decs[dec].ty.clone() else {
                continue;
            };
            let vis = ir.resolutions.decs[dec].vis;
            let defaults = defaults
                .into_iter()
                .map(|d| d.map(|lit| Constant::from_literal(ir, lit)))
                .collect();
            signatures[body] = Some(Function {
                body,
                vis,
                header,
                defaults,
            });
        }
        for (body, node) in std::mem::take(&mut ir.closure_bodies) {
            let Some(Ty::Fn(header)) = ir.resolutions.node_tys.get(&node).cloned() else {
                continue;
            };
            let defaults = vec![None; header.parameters.len()];
            signatures[body] = Some(Function {
                body,
                vis: Vis::Private,
                header,
                defaults,
            });
        }
        // `multi fn` trampolines aren't item bodies -- they're synthesized for
        // `ref_body`, but a `Val::Fn` still needs a signature to be called.
        // r[impl mimas.multi]
        for (dec, body) in std::mem::take(&mut ir.multi_tramps) {
            let Ty::Fn(header) = ir.resolutions.decs[dec].ty.clone() else {
                continue;
            };
            let defaults = vec![None; header.parameters.len()];
            signatures[body] = Some(Function {
                body,
                vis: Vis::Private,
                header,
                defaults,
            });
        }

        let root = std::mem::take(&mut ir.resolutions.root);
        let root = export(ir, &signatures, root);

        // the entry body's local->reg map, captured when body 0 compiles -- non-entry bodies
        // need it to turn a global's `Local` into its absolute (base-0) entry-frame register.
        let mut entry_to_reg: IdVec<Local, Reg> = IdVec::new();

        // the optimizer's whole-program half here; its per-body half runs
        // body by body in the loop below, where `bodycache` can skip it
        let optimizing = opt::enabled();
        let mut opt_times = opt::OptTimes::new();
        let global = if optimizing {
            opt::optimize_global(ir, &self.frame_natives, &self.mutating_natives, &mut opt_times)
        } else {
            opt::Global {
                values: opt::value_adts(ir, &self.mutating_natives),
                entry_locals: ir.bodies[BodyId::ZERO].locals.values().copied().collect(),
            }
        };
        // a dump of the optimized IR wants every body optimized before any
        // is compiled, and the disasm dump wants every body compiled
        let dump_opt = std::env::var("MIMAS_DUMP_OPT").is_ok_and(|v| v == "1");
        let cache = match bodycache::mode() {
            _ if dump_opt || self.disasm => bodycache::Mode::Off,
            m => m,
        };
        if dump_opt {
            if optimizing {
                for (bid, body) in ir.bodies.iter_mut() {
                    if bid != BodyId::ZERO {
                        opt::optimize_body(body, &global, &ir.resolutions, self.shadows, &mut opt_times);
                    }
                }
            }
            eprintln!("{ir}");
        }
        let per_body_opt = optimizing && !dump_opt;
        // what changes the passes, for the cache key
        let config = u8::from(optimizing) | u8::from(opt::ssa_enabled()) << 1 | u8::from(self.shadows) << 2;
        let (mut hits, mut misses) = (0usize, 0usize);
        if cache != bodycache::Mode::Off {
            bodycache::begin();
        }
        let value_adts: IdVec<AdtId, bool> = ir
            .resolutions
            .adts
            .iter()
            .map(|(id, _)| global.values.contains(&id))
            .collect::<Vec<_>>()
            .into();
        for (body_id, body) in ir.bodies.iter_mut() {
            // the entry body changes with nearly every edit, and its
            // registers are every other body's globals: always compiled
            let key = (cache != bodycache::Mode::Off && body_id != BodyId::ZERO)
                .then(|| bodycache::key(body, &ir.resolutions, &global, config));
            let cached = key.as_ref().and_then(bodycache::get);
            if let Some(hit) = &cached {
                hits += 1;
                if cache == bodycache::Mode::On {
                    let (ops, chunk) = hit.at(bytes.len());
                    self.chunks.push(chunk);
                    for op in ops.iter().cloned() {
                        op.encode(&mut bytes);
                    }
                    self.ops.extend(ops);
                    continue;
                }
            } else if key.is_some() {
                misses += 1;
                if let Ok(dir) = std::env::var("MIMAS_BODYCACHE_DUMP") {
                    let (_, text) = bodycache::key_text(body, &ir.resolutions, &global, config, true);
                    let _ = std::fs::write(
                        format!("{dir}/miss{}.g{}.key", body_id.index(), bodycache::generation()),
                        text,
                    );
                }
            }
            if per_body_opt && body_id != BodyId::ZERO {
                opt::optimize_body(body, &global, &ir.resolutions, self.shadows, &mut opt_times);
            }
            clean::clean(body);

            let local_kinds = local_kinds(&body, &ir.resolutions);
            let kinds = inst_kinds(&body, &local_kinds);
            let mut regs: IdVec<Reg, OperandKind> = IdVec::new();
            let mut local_to_reg: IdVec<Local, Reg> = IdVec::new();
            let cross = clean::cross_block_iids(body);
            let last_use_by_block: HashMap<BlockId, HashMap<InstId, InstId>> = body
                .blocks
                .iter()
                .map(|(bid, _)| (bid, clean::last_uses(body, bid)))
                .collect();

            for kind in local_kinds.values() {
                local_to_reg.push(regs.push(*kind));
            }
            if body_id == BodyId::ZERO {
                entry_to_reg = local_to_reg.clone();
            }

            let mut inst_to_reg: IdVec<InstId, Option<Reg>> =
                vec![None; body.instructions.len()].into();

            let mut phi_copies: HashMap<BlockId, Vec<(Reg, InstId)>> = HashMap::new();

            for (_, block) in body.blocks.iter() {
                for &iid in &block.stream {
                    if let Inst::Phi(branches) = &body.instructions[iid] {
                        let dst = regs.push(kinds[iid]);
                        inst_to_reg[iid] = Some(dst);
                        for &(pred, value) in branches {
                            phi_copies.entry(pred).or_default().push((dst, value));
                        }
                    }
                }
            }

            let chunk_offset = bytes.len();
            let mut byte_offset = chunk_offset;
            let mut body_ops = Vec::new();
            let mut block_offset: IdVec<BlockId, usize> =
                IdVec::from(vec![0usize; body.blocks.len()]);
            let mut locs: Vec<(u32, Location)> = Vec::new();
            let push_loc = |off: usize, loc: Location, locs: &mut Vec<(u32, Location)>| {
                let rel = u32::try_from(off - chunk_offset).unwrap();
                if locs.last().is_none_or(|(_, prev)| *prev != loc) {
                    locs.push((rel, loc));
                }
            };

            // a constant whose every use is an arithmetic immediate is never materialized -- the
            // value is inlined into each AddIntImm/etc. rather than loaded into a register.
            let absorbed: HashSet<InstId> = {
                let mut total: HashMap<InstId, u32> = HashMap::new();
                let mut as_imm: HashMap<InstId, u32> = HashMap::new();
                for (_, block) in body.blocks.iter() {
                    for &iid in &block.stream {
                        let inst = &body.instructions[iid];
                        let absorbs = imm_binop(body, inst).map(|(_, con, _, _, _)| con);
                        for u in uses(inst) {
                            if matches!(
                                body.instructions[u],
                                Inst::Constant(Constant::Int(_) | Constant::Float(_))
                            ) {
                                *total.entry(u).or_default() += 1;
                                if absorbs == Some(u) {
                                    *as_imm.entry(u).or_default() += 1;
                                }
                            }
                        }
                    }
                }
                total
                    .iter()
                    .filter(|(c, n)| as_imm.get(c) == Some(n))
                    .map(|(&c, _)| c)
                    .collect()
            };

            // pre-scan: give each DISTINCT cache-friendly constant a pinned register, so
            // loop-resident loads can be hoisted to the prologue and every use just aliases the
            // reg.
            //
            // this is currently limited to 12, as in theory, one could have a hot function with
            // a body like this...
            // ```
            // if foo {
            //     // many many constants
            // } else {
            //     // many many OTHER constants
            // }
            // ```
            //
            // ... where lifting the wrong branches constants would be a wasteful expense. as time
            // goes on though, this continues to not come up whereas a reason to lift the cap does.
            // so, this might raise in the future, perhaps to a much higher number.
            let mut constants = HashMap::new();
            'outer: for (_, block) in &body.blocks {
                for iid in block.stream.iter().copied() {
                    if let Inst::Constant(con) = &body.instructions[iid]
                        && let Some(con) = con.cached()
                        && !absorbed.contains(&iid)
                    {
                        constants
                            .entry(con)
                            .or_insert_with(|| regs.push(kinds[iid]));
                        if constants.len() >= 12 {
                            break 'outer;
                        }
                    }
                }
            }

            // get the emit order of each block so that we can identify which jumps are
            // fallthroughs. we can't just look at block index + 1 because dce could have made gaps.

            // successors of a block: (fall-through, branches)
            let successors = |bid: BlockId| {
                let mut fallthrough = None;
                let mut branches = Vec::new();
                for &iid in &body.blocks[bid].stream {
                    match &body.instructions[iid] {
                        Inst::Jump { target } => fallthrough = Some(*target),
                        Inst::JumpIfFalse { target, .. } | Inst::ForNext { target, .. } => {
                            branches.push(*target)
                        }
                        Inst::Switch { table, default, .. } => {
                            branches.extend(table.iter().copied());
                            branches.push(*default);
                        }
                        _ => {}
                    }
                }
                (fallthrough, branches)
            };

            let mut order = Vec::new();
            let mut placed = vec![false; body.blocks.len()];
            let mut stack = vec![BlockId::ZERO];
            while let Some(b) = stack.pop() {
                // give me take!!!! why did they reject the rfc!!!
                if std::mem::replace(&mut placed[b.index()], true) {
                    continue;
                }

                order.push(b);
                let (fallthrough, branches) = successors(b);
                for b in branches {
                    if !placed[b.index()] {
                        stack.push(b);
                    }
                }

                if let Some(f) = fallthrough
                    && !placed[f.index()]
                {
                    stack.push(f);
                }
            }

            let non_empty: Vec<BlockId> = order
                .iter()
                .copied()
                .filter(|&id| !body.blocks[id].stream.is_empty())
                .collect();
            let fallthrough: HashMap<BlockId, BlockId> =
                non_empty.windows(2).map(|w| (w[0], w[1])).collect();

            for &bid in &order {
                let block = &body.blocks[bid];
                let last_use = &last_use_by_block[&bid];
                let mut free: Vec<Reg> = Vec::new();

                let clean_is_safe = |iid: InstId, u: InstId| -> bool {
                    last_use.get(&u) == Some(&iid)
                        && !cross.contains(&u)
                        && !matches!(body.instructions[u], Inst::GetLocal(_) | Inst::Phi(_))
                };

                block_offset[bid] = byte_offset;

                // prologue: materialize the cached constants once at the top of the entry block.
                // entry dominates every block, so this is live at every aliased use; runs once per
                // call. sorted by reg for deterministic bytecode (HashMap order is randomized).
                if bid == BlockId::ZERO {
                    let mut loads: Vec<(Reg, Constant)> = constants
                        .iter()
                        .map(|(c, &r)| (r, c.to_constant()))
                        .collect();
                    loads.sort_by_key(|(r, _)| r.index());
                    for (dst, constant) in loads {
                        let op = Op::LoadConst { dst, constant };
                        push_loc(byte_offset, Location::SYNTHETIC, &mut locs);
                        byte_offset += op.encoded_len();
                        body_ops.push(op);
                    }
                }

                for iid in block.stream.iter().copied() {
                    let inst = &body.instructions[iid];
                    let loc = body.locs[iid];

                    // some insts have optimization shortcuts
                    match inst {
                        Inst::Phi(_) => continue,
                        Inst::GetLocal(l) => {
                            // alias it straight away -- no need for a move
                            inst_to_reg[iid] = Some(local_to_reg[l]);
                            continue;
                        }
                        Inst::SetLocal(l, v) => {
                            let dst = local_to_reg[l];
                            let src = inst_to_reg[v].unwrap();
                            if clean_is_safe(iid, *v)
                                && !constants.values().any(|&r| r == src)
                                && body_ops.last().and_then(Op::reg) == Some(src)
                                && same_class(regs[src], regs[dst])
                            {
                                body_ops.last_mut().unwrap().set_reg(dst);
                                regs[dst] = widen(regs[dst], regs[src]);
                                inst_to_reg[v] = Some(dst);
                                free.push(src);
                                continue;
                            }
                        }
                        Inst::JumpIfFalse { condition, target } => {
                            let con_reg = inst_to_reg[condition].unwrap();
                            let block_target = BlockTarget::Block(*target);
                            if clean_is_safe(iid, *condition)
                                && body_ops.last().is_some_and(|op| {
                                    Op::reg(op) == Some(con_reg) && Op::is_comparison(op)
                                })
                            {
                                let cmp = body_ops.pop().unwrap();
                                byte_offset -= cmp.encoded_len();
                                let op = fuse_branch!(cmp, block_target;
                                    IntLt => BIntLt, IntLe => BIntLe, IntGt => BIntGt,
                                    IntGe => BIntGe, IntEq => BIntEq, IntNe => BIntNe,
                                    FloatLt => BFloatLt, FloatLe => BFloatLe, FloatGt => BFloatGt,
                                    FloatGe => BFloatGe, FloatEq => BFloatEq, FloatNe => BFloatNe;
                                    IntLtImm => BIntLtImm, IntLeImm => BIntLeImm, IntGtImm => BIntGtImm,
                                    IntGeImm => BIntGeImm, IntEqImm => BIntEqImm, IntNeImm => BIntNeImm,
                                    FloatLtImm => BFloatLtImm, FloatLeImm => BFloatLeImm, FloatGtImm => BFloatGtImm,
                                    FloatGeImm => BFloatGeImm, FloatEqImm => BFloatEqImm, FloatNeImm => BFloatNeImm
                                );

                                byte_offset += op.encoded_len();
                                body_ops.push(op);
                                continue;
                            }
                        }
                        Inst::Jump { target, .. } => {
                            if let Some(copies) = phi_copies.get(&bid) {
                                // The copies are one parallel assignment: with several
                                // phis in the target block a destination may be another
                                // copy's source (a loop that swaps two values), so order
                                // them so no source is overwritten before it is read, and
                                // break a cycle through a scratch register.
                                let mut pending: Vec<(Reg, Reg)> = copies
                                    .iter()
                                    .map(|&(dst, value)| (dst, inst_to_reg[value].unwrap()))
                                    .filter(|(d, s)| d != s)
                                    .collect();
                                let mut moves: Vec<(Reg, Reg)> = Vec::new();
                                while !pending.is_empty() {
                                    if let Some(i) = pending.iter().position(|&(d, _)| {
                                        !pending.iter().any(|&(d2, s2)| d2 != d && s2 == d)
                                    }) {
                                        moves.push(pending.remove(i));
                                    } else {
                                        // every destination is still needed: park one source
                                        let (_, s) = pending[0];
                                        let tmp = regs.push(regs[s]);
                                        moves.push((tmp, s));
                                        for p in pending.iter_mut() {
                                            if p.1 == s {
                                                p.1 = tmp;
                                            }
                                        }
                                    }
                                }
                                for (dst, src) in moves {
                                    let op = Op::Move { dst, src };
                                    // phi-copy moves sit at the predecessor's tail -- borrow the
                                    // jump's loc so a fault here attributes to the same source
                                    // range.
                                    push_loc(byte_offset, loc, &mut locs);
                                    byte_offset += op.encoded_len();
                                    body_ops.push(op);
                                }
                            }

                            if fallthrough.get(&bid) == Some(target) {
                                // this is a no-op, skip it!
                                continue;
                            }
                        }
                        Inst::Constant(_) if absorbed.contains(&iid) => continue,
                        Inst::Constant(con)
                            if let Some(reg) = con.cached().and_then(|v| constants.get(&v)) =>
                        {
                            inst_to_reg[iid] = Some(*reg);
                            continue;
                        }
                        _ => {}
                    };

                    // fuse an int arith/comparison op with a constant operand into its immediate
                    // form, inlining the value instead of reading it from a register.
                    if let Some((non_const, _con, val, op, kind)) = imm_binop(body, inst) {
                        let (left, dst) = {
                            let mut ctx = Ctx {
                                inst_to_reg: &inst_to_reg,
                                local_to_reg: &local_to_reg,
                                entry_to_reg: &entry_to_reg,
                                regs: &mut regs,
                                free: &mut free,
                                kind: kinds[iid],
                            };
                            (ctx.i2r(&non_const), ctx.reg())
                        };
                        use OperandKind::{Float, Int};
                        let imm = match (kind, op) {
                            (Int, BinOp::Add) => Op::AddIntImm { dst, left, val },
                            (Int, BinOp::Sub) => Op::SubIntImm { dst, left, val },
                            (Int, BinOp::Mult) => Op::MultIntImm { dst, left, val },
                            (Int, BinOp::Mod) => Op::ModIntImm { dst, left, val },
                            (Int, BinOp::LessThan) => Op::IntLtImm { dst, left, val },
                            (Int, BinOp::LessEqual) => Op::IntLeImm { dst, left, val },
                            (Int, BinOp::GreaterThan) => Op::IntGtImm { dst, left, val },
                            (Int, BinOp::GreaterEqual) => Op::IntGeImm { dst, left, val },
                            (Int, BinOp::Identity) => Op::IntEqImm { dst, left, val },
                            (Int, BinOp::NotEqual) => Op::IntNeImm { dst, left, val },
                            (Float, BinOp::Add) => Op::AddFloatImm { dst, left, val },
                            (Float, BinOp::Sub) => Op::SubFloatImm { dst, left, val },
                            (Float, BinOp::Mult) => Op::MultFloatImm { dst, left, val },
                            (Float, BinOp::Mod) => Op::ModFloatImm { dst, left, val },
                            (Float, BinOp::LessThan) => Op::FloatLtImm { dst, left, val },
                            (Float, BinOp::LessEqual) => Op::FloatLeImm { dst, left, val },
                            (Float, BinOp::GreaterThan) => Op::FloatGtImm { dst, left, val },
                            (Float, BinOp::GreaterEqual) => Op::FloatGeImm { dst, left, val },
                            (Float, BinOp::Identity) => Op::FloatEqImm { dst, left, val },
                            (Float, BinOp::NotEqual) => Op::FloatNeImm { dst, left, val },
                            _ => unreachable!(
                                "imm_binop only returns int/float arith + comparison ops"
                            ),
                        };
                        inst_to_reg[iid] = Some(dst);
                        if clean_is_safe(iid, non_const) && !constants.values().any(|&r| r == left)
                        {
                            free.push(left);
                        }
                        push_loc(byte_offset, loc, &mut locs);
                        byte_offset += imm.encoded_len();
                        body_ops.push(imm);
                        continue;
                    }

                    let op = {
                        let ctx = Ctx {
                            inst_to_reg: &inst_to_reg,
                            local_to_reg: &local_to_reg,
                            entry_to_reg: &entry_to_reg,
                            regs: &mut regs,
                            free: &mut free,
                            kind: kinds[iid],
                        };

                        Op::from_inst(inst, ctx)
                    };

                    inst_to_reg[iid] = op.reg();

                    // free each operand whose last use is here (uses() is already deduped, so a
                    // value filling two slots like `x * x` won't get freed twice). a pinned
                    // constant register is never recycled -- its value must outlive every use.
                    for u in uses(&body.instructions[iid]) {
                        let reg = inst_to_reg[u].unwrap();
                        if clean_is_safe(iid, u) && !constants.values().any(|&r| r == reg) {
                            free.push(reg);
                        }
                    }

                    push_loc(byte_offset, loc, &mut locs);
                    byte_offset += op.encoded_len();
                    body_ops.push(op);
                }
            }

            for op in &mut body_ops {
                if let Op::Jump { target }
                | Op::JumpIf { target, .. }
                | Op::BIntLt { target, .. }
                | Op::BIntLe { target, .. }
                | Op::BIntGt { target, .. }
                | Op::BIntGe { target, .. }
                | Op::BIntEq { target, .. }
                | Op::BIntNe { target, .. }
                | Op::BIntLtImm { target, .. }
                | Op::BIntLeImm { target, .. }
                | Op::BIntGtImm { target, .. }
                | Op::BIntGeImm { target, .. }
                | Op::BIntEqImm { target, .. }
                | Op::BIntNeImm { target, .. }
                | Op::BFloatLt { target, .. }
                | Op::BFloatLe { target, .. }
                | Op::BFloatGt { target, .. }
                | Op::BFloatGe { target, .. }
                | Op::BFloatEq { target, .. }
                | Op::BFloatNe { target, .. }
                | Op::BFloatLtImm { target, .. }
                | Op::BFloatLeImm { target, .. }
                | Op::BFloatGtImm { target, .. }
                | Op::BFloatGeImm { target, .. }
                | Op::BFloatEqImm { target, .. }
                | Op::BFloatNeImm { target, .. }
                | Op::ForNext { target, .. } = op
                    && let BlockTarget::Block(b) = target
                {
                    *target = BlockTarget::ByteOffset(block_offset[*b]);
                }
                if let Op::Switch { default, table, .. } = op {
                    for t in std::iter::once(default).chain(table.iter_mut()) {
                        if let BlockTarget::Block(b) = t {
                            *t = BlockTarget::ByteOffset(block_offset[*b]);
                        }
                    }
                }
            }

            let params = body
                .params
                .iter()
                .map(|local| local_to_reg[*local])
                .collect();
            let captures = body
                .captures
                .iter()
                .map(|local| local_to_reg[*local])
                .collect();
            // r[impl mimas.inspect.by-name]
            // One name, one register: when a top-level `let` and a block local share a
            // name, the `let` is the one a host reading by name means; between two
            // locals, the later declaration (the shadowing one) wins.
            let named = named_locals(&body, &ir.resolutions);
            let locals = named
                .iter()
                .map(|(name, local)| (name.clone(), local_to_reg[*local]))
                .collect();
            let shadows = body
                .shadows
                .iter()
                .filter_map(|(local, sh)| {
                    let name = body.artifacts.get(local)?;
                    (named.get(name) == Some(local))
                        .then(|| (name.clone(), sh.map(&|l| local_to_reg[*l])))
                })
                .collect();
            let args = u16::try_from(body.params.len()).unwrap();
            let reg_kinds: Vec<OperandKind> = regs.values().copied().collect();
            let regs = u16::try_from(regs.len()).unwrap();

            if self.disasm {
                let name = names.get(&body_id).map(String::as_str).unwrap_or("<anon>");
                let moves = body_ops
                    .iter()
                    .filter(|op| matches!(op, Op::Move { .. }))
                    .count();
                println!(
                    "\n── body @{} {name:?}  ops {} · regs {regs} · moves {moves} ──",
                    body_id.index(),
                    body_ops.len(),
                );
                let mut rel = 0;
                let mut shown_line: Option<usize> = None;
                for op in body_ops.iter() {
                    // print the source line above the ops it lowered from, when it changes.
                    let loc = locs
                        .partition_point(|(b, _)| (*b as usize) <= rel)
                        .checked_sub(1)
                        .map(|i| locs[i].1);
                    if let Some(loc) = loc
                        && !loc.is_synthetic()
                        && let Some(src) = self.srcs.get(&loc.file_id)
                    {
                        let upto = loc.span.start.min(src.len());
                        let line = src[..upto].bytes().filter(|&b| b == b'\n').count();
                        if shown_line != Some(line) {
                            shown_line = Some(line);
                            let text = src.lines().nth(line).unwrap_or("").trim();
                            println!("  {}", format!("{} │ {text}", line + 1).dimmed());
                        }
                    }
                    // byte-offset column, colored like the existing address column so jump
                    // targets are cross-referenceable.
                    let addr = format!("{:04}", chunk_offset + rel).bright_black().bold();
                    println!("  {addr}  {op}");
                    rel += op.encoded_len();
                }
            }

            let chunk = Chunk {
                offset: chunk_offset,
                args,
                regs,
                params,
                captures,
                locals,
                shadows,
                locs,
                kinds: reg_kinds,
            };
            if let Some(k) = key {
                let fresh = bodycache::Entry::new(&body_ops, &chunk);
                match &cached {
                    Some(hit) if !hit.same(&fresh) => {
                        if let Ok(dir) = std::env::var("MIMAS_BODYCACHE_DUMP") {
                            hit.dump(&format!("{dir}/body{}.cached", body_id.index()));
                            fresh.dump(&format!("{dir}/body{}.fresh", body_id.index()));
                        }
                        panic!(
                        "MIMAS_BODYCACHE=check: body {} compiled differently than its cached form \
                         (a read the key misses)",
                        body_id.index()
                    )
                    }
                    Some(_) => {}
                    None => bodycache::put(k, fresh),
                }
            }
            self.chunks.push(chunk);

            for op in body_ops.iter().cloned() {
                op.encode(&mut bytes);
            }
            self.ops.extend(body_ops);
        }
        opt_times.report();
        bodycache::record(hits, misses);
        if cache != bodycache::Mode::Off {
            bodycache::end();
            if std::env::var_os("MIMAS_COMPILE_TIMING").is_some() {
                eprintln!("BODYCACHE hits {hits} misses {misses}");
            }
        }

        let struct_names: IdVec<_, String> = ir
            .resolutions
            .adts
            .values()
            .map(|adt| adt.name.clone())
            .collect::<Vec<_>>()
            .into();

        let mut by_adt: Vec<HashMap<String, BodyId>> = ir
            .resolutions
            .adts
            .values()
            .map(|adt| {
                adt.methods
                    .iter()
                    .filter_map(|(name, &dec)| {
                        ir.item_bodies.get(&dec).map(|&body| (name.clone(), body))
                    })
                    .collect()
            })
            .collect();
        // A value's runtime struct id is its variant's for an enum, while the
        // impl sits on the enum: register the enum's methods under each
        // variant id too, so a host asking `Sky::Moon` for `inv` (or `draw`)
        // finds it. A variant's own method of the same name wins.
        for (i, adt) in ir.resolutions.adts.values().enumerate() {
            for did in &adt.dispatch_ids {
                let d = did.index();
                if d == i || d >= by_adt.len() {
                    continue;
                }
                let inherited: Vec<(String, BodyId)> =
                    by_adt[i].iter().map(|(k, v)| (k.clone(), *v)).collect();
                for (name, body) in inherited {
                    by_adt[d].entry(name).or_insert(body);
                }
            }
        }
        // r[impl mimas.show.runtime] each adt that implements `Show`, and each
        // of its variant ids, names the body of its `show`
        let shows = |i: usize, by_adt: &Vec<HashMap<String, BodyId>>| -> Option<BodyId> {
            let adt = ir.resolutions.adts.values().nth(i)?;
            let implements = adt
                .implements
                .iter()
                .any(|p| ir.resolutions.pact_names.get(*p).is_some_and(|n| n == "Show"));
            if !implements {
                return None;
            }
            let table = by_adt.get(i)?;
            table.get("show").copied().or_else(|| {
                table.iter().find(|(k, _)| k.ends_with("::show")).map(|(_, b)| *b)
            })
        };
        let mut show_bodies: Vec<Option<BodyId>> =
            (0..by_adt.len()).map(|i| shows(i, &by_adt)).collect();
        for (i, adt) in ir.resolutions.adts.values().enumerate() {
            let Some(body) = show_bodies[i] else { continue };
            for did in &adt.dispatch_ids {
                if let Some(slot) = show_bodies.get_mut(did.index())
                    && slot.is_none()
                {
                    *slot = Some(body);
                }
            }
        }
        let show_bodies: IdVec<AdtId, Option<BodyId>> = show_bodies.into();
        let methods: IdVec<_, HashMap<String, BodyId>> = by_adt.into();

        let field_names: IdVec<_, Vec<String>> = ir
            .resolutions
            .adts
            .values()
            .map(|adt| adt.fields.clone())
            .collect::<Vec<_>>()
            .into();

        // A variant's own adt id is what an instance carries as `struct_id`;
        // map it to the variant's position in its enum so a host can read a
        // variant as its ordinal. Structs list only themselves and stay `None`.
        let mut variant_ordinals: IdVec<_, Option<u32>> =
            vec![None; ir.resolutions.adts.len()].into();
        for (i, adt) in ir.resolutions.adts.values().enumerate() {
            for (ord, did) in adt.dispatch_ids.iter().enumerate() {
                let d = did.index();
                if d != i && d < variant_ordinals.len() {
                    variant_ordinals[AdtId::from(d as u32)] = Some(ord as u32);
                }
            }
        }

        let field_dims: IdVec<_, Vec<Option<shared::units::Dim>>> = ir
            .resolutions
            .adts
            .values()
            .map(|adt| adt.member_dims.clone())
            .collect::<Vec<_>>()
            .into();

        let field_tys: IdVec<_, Vec<Ty>> = ir
            .resolutions
            .adts
            .values()
            .map(|adt| adt.member_tys.clone())
            .collect::<Vec<_>>()
            .into();

        let field_shapes: IdVec<_, Vec<shared::units::Shape>> = ir
            .resolutions
            .adts
            .values()
            .map(|adt| {
                adt.member_tys
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| match adt.member_shapes.get(i).cloned().flatten() {
                        Some(shape) => shape,
                        None => shared::units::Shape::of_field(ty, adt.member_dims.get(i).copied().flatten()),
                    })
                    .collect()
            })
            .collect::<Vec<_>>()
            .into();
        let print_shapes = std::mem::take(&mut ir.resolutions.print_shapes).into_iter().collect();
        let shown_units = std::mem::take(&mut ir.resolutions.shown_units).into_iter().collect();
        let hole_shapes = std::mem::take(&mut ir.hole_shapes);

        let tests = std::mem::take(&mut ir.tests);

        let print_dims = std::mem::take(&mut ir.resolutions.print_dims)
            .into_iter()
            .collect();
        let echo_units = std::mem::take(&mut ir.resolutions.echo_units)
            .into_iter()
            .collect();

        Program {
            entry: BodyId::ZERO,
            root,
            chunks: std::mem::replace(&mut self.chunks, IdVec::new()),
            signatures,
            strs: std::mem::take(&mut ir.str_interner),
            bytes: bytes.finish(),
            struct_names,
            methods,
            field_names,
            variant_ordinals,
            field_dims,
            print_dims,
            echo_units,
            print_shapes,
            hole_shapes,
            field_shapes,
            shown_units,
            show_bodies,
            field_tys,
            value_adts,
            tests,
            units: shared::units::snapshot(),
        }
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) struct Ctx<'a> {
    pub inst_to_reg: &'a IdVec<InstId, Option<Reg>>,
    pub local_to_reg: &'a IdVec<Local, Reg>,
    /// The entry body's local->reg map. Its registers are the global slots (`t.regs[0..]`,
    /// base 0) that `GetEntry`/`SetEntry` insts address.
    pub entry_to_reg: &'a IdVec<Local, Reg>,
    pub regs: &'a mut IdVec<Reg, OperandKind>,
    pub free: &'a mut Vec<Reg>,
    /// Value class of the inst being lowered -- `reg` only recycles a register of this class.
    pub kind: OperandKind,
}

impl Ctx<'_> {
    pub fn reg(&mut self) -> Reg {
        let kind = self.kind;
        match self
            .free
            .iter()
            .rposition(|&r| same_class(self.regs[r], kind))
        {
            Some(at) => {
                let r = self.free.remove(at);
                self.regs[r] = widen(self.regs[r], kind);
                r
            }
            None => self.regs.push(kind),
        }
    }

    pub fn i2r(&self, iid: &InstId) -> Reg {
        self.inst_to_reg[iid].unwrap()
    }
}

// if `inst` is an int arith or comparison op with a constant-int operand, return (non-constant
// operand, the constant operand, value, effective op). The constant is canonicalized to the right:
// Sub/Mod only fuse a right constant; Add/Mult take either side; comparisons take either side and
// flip the operator when the constant was on the left (`2 < n` == `n > 2`).
fn imm_binop(body: &Body, inst: &Inst) -> Option<(InstId, InstId, i64, BinOp, OperandKind)> {
    use BinOp::*;
    let Inst::BinOp {
        left,
        op,
        right,
        kind,
    } = inst
    else {
        return None;
    };
    let kind = *kind;
    // int constants pass their value through; float constants are bit-cast into the i64 slot
    let cst = |i: InstId| match (kind, &body.instructions[i]) {
        (OperandKind::Int, Inst::Constant(Constant::Int(v))) => Some(*v),
        (OperandKind::Float, Inst::Constant(Constant::Float(f))) => Some(f.to_bits() as i64),
        _ => None,
    };
    let (non_const, con, val, op) = match op {
        Add | Mult => match (cst(*left), cst(*right)) {
            (_, Some(v)) => (*left, *right, v, *op),
            (Some(v), _) => (*right, *left, v, *op),
            _ => return None,
        },
        Sub | Mod => (*left, *right, cst(*right)?, *op),
        LessThan | LessEqual | GreaterThan | GreaterEqual | Identity | NotEqual => {
            match (cst(*left), cst(*right)) {
                (_, Some(v)) => (*left, *right, v, *op),
                (Some(v), _) => {
                    // gotta flip the operator
                    let op = match op {
                        BinOp::LessThan => BinOp::GreaterThan,
                        BinOp::LessEqual => BinOp::GreaterEqual,
                        BinOp::GreaterThan => BinOp::LessThan,
                        BinOp::GreaterEqual => BinOp::LessEqual,
                        other => *other, // equality is symetric
                    };
                    (*right, *left, v, op)
                }
                _ => return None,
            }
        }
        _ => return None,
    };
    Some((non_const, con, val, op, kind))
}

/// Registers are recycled only within one class: `Str` and `Generic` values are both opaque
/// words and share one; each scalar kind is its own.
fn same_class(a: OperandKind, b: OperandKind) -> bool {
    use OperandKind::{Generic, Str};
    a == b || matches!((a, b), (Str | Generic, Str | Generic))
}

fn widen(a: OperandKind, b: OperandKind) -> OperandKind {
    if a == b { a } else { OperandKind::Generic }
}

fn ty_kind(ty: &Ty) -> OperandKind {
    match ty {
        Ty::Bool => OperandKind::Bool,
        Ty::Int => OperandKind::Int,
        Ty::Float => OperandKind::Float,
        Ty::Str => OperandKind::Str,
        _ => OperandKind::Generic,
    }
}

/// Each local's class: its declared type, or for synthetic locals (no dec) the class every
/// write to it agrees on.
fn local_kinds(body: &Body, res: &solve::Resolutions) -> IdVec<Local, OperandKind> {
    let mut kinds: IdVec<Local, OperandKind> = body
        .locals
        .values()
        .map(|&dec| {
            if dec == DecId::DANGLING {
                OperandKind::Generic
            } else {
                ty_kind(&res.decs[dec].ty)
            }
        })
        .collect::<Vec<_>>()
        .into();
    let mut written: HashMap<Local, Option<OperandKind>> = HashMap::new();
    for (_, block) in body.blocks.iter() {
        for &iid in &block.stream {
            if let Inst::SetLocal(l, v) | Inst::Shadow { local: l, value: v } =
                &body.instructions[iid]
                && body.locals[*l] == DecId::DANGLING
            {
                let k = fixed_kind(body, *v, &kinds);
                let slot = written.entry(*l).or_insert(k);
                if *slot != k {
                    *slot = None;
                }
            }
        }
    }
    for (l, k) in written {
        if let Some(k) = k {
            kinds[l] = k;
        }
    }
    kinds
}

/// The class an inst's result has without looking through phis: fixed by the op where the op
/// decides it, else the solved type of the expr it lowered from.
/// Each source name in `body` mapped to the one local a by-name read resolves to:
/// a top-level `let` over a block local of the same name, then the later declaration.
fn named_locals(body: &Body, res: &solve::Resolutions) -> HashMap<String, Local> {
    let rank = |l: Local| {
        let dec = body.locals[l];
        let global =
            dec != DecId::DANGLING && matches!(res.decs[dec].kind, ResolvedDeclKind::Global);
        (global, l.index())
    };
    let mut out: HashMap<String, Local> = HashMap::new();
    for (&local, name) in &body.artifacts {
        match out.get(name) {
            Some(&have) if rank(have) >= rank(local) => {}
            _ => {
                out.insert(name.clone(), local);
            }
        }
    }
    out
}

fn fixed_kind(body: &Body, iid: InstId, locals: &IdVec<Local, OperandKind>) -> Option<OperandKind> {
    use BinOp::*;
    use OperandKind as K;
    let fixed = match &body.instructions[iid] {
        Inst::Constant(c) => Some(match c {
            Constant::Bool(_) => K::Bool,
            Constant::Int(_) => K::Int,
            Constant::Float(_) => K::Float,
            Constant::Str(_) => K::Str,
            Constant::Array(_) | Constant::Null | Constant::Dec(_) => K::Generic,
        }),
        Inst::BinOp { op, kind, .. } => match op {
            Identity | NotEqual | LessThan | LessEqual | GreaterThan | GreaterEqual => {
                Some(K::Bool)
            }
            Add | Sub | Mult | Mod if matches!(kind, K::Int | K::Float) => Some(*kind),
            Div if *kind == K::Float => Some(K::Float),
            _ => None,
        },
        Inst::UnaryOp {
            op: UnaryOp::Not, ..
        } => Some(K::Bool),
        Inst::ToFloat(_) | Inst::Sqrt(_) => Some(K::Float),
        Inst::Len(_) => Some(K::Int),
        Inst::In(..) | Inst::IsInstance { .. } | Inst::IsRaised(_) => Some(K::Bool),
        Inst::GetLocal(l) => Some(locals[*l]),
        Inst::Phi(_) => None,
        _ => Some(K::Generic),
    };
    match fixed {
        Some(K::Generic) | None => body.kinds.get(&iid).copied().or(fixed),
        k => k,
    }
}

/// Every inst's result class. A phi without a solved type takes the class its incoming values
/// agree on.
fn inst_kinds(body: &Body, locals: &IdVec<Local, OperandKind>) -> IdVec<InstId, OperandKind> {
    let mut known: Vec<Option<OperandKind>> = (0..body.instructions.len())
        .map(|i| fixed_kind(body, InstId::from(i as u32), locals))
        .collect();
    loop {
        let mut changed = false;
        for (iid, inst) in body.instructions.iter() {
            let Inst::Phi(branches) = inst else { continue };
            if known[iid.index()].is_some() {
                continue;
            }
            let mut agreed: Option<OperandKind> = None;
            for &(_, v) in branches {
                if let Some(k) = known[v.index()] {
                    agreed = Some(agreed.map_or(k, |a| widen(a, k)));
                }
            }
            if agreed.is_some() {
                known[iid.index()] = agreed;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    known
        .into_iter()
        .map(|k| k.unwrap_or(OperandKind::Generic))
        .collect::<Vec<_>>()
        .into()
}

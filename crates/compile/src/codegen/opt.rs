//! IR-level optimizations that run before `clean`: inlining of small bodies and
//! scalar replacement of struct locals. Together they turn `a.sub(b).dot(c)`
//! -- two calls and a heap `V2` per step of a numeric loop -- into float
//! arithmetic on registers, which is where every lane (interpreter, jit,
//! wasmgen) gets its speed from. `MIMAS_OPT=0` disables both.
//!
//! Scope, deliberately: neither pass touches the entry body. Its locals are
//! the page's top-level bindings -- the debugger, `#m` prose and `envDraw`
//! read them off the heap by declaration -- so a struct bound at top level
//! must stay a struct.
use std::collections::{HashMap, HashSet};

use api::NativeId;
use parse::AccessKind;
use shared::Location;
use solve::Resolutions;
use solve::components::{AdtId, DecId, Ty};

use crate::{
    BlockId, Inst, InstId, Ir, Local, Shadow,
    ir::{Block, Body, BodyId, FormatPart},
};

use super::clean::uses;

/// Callee bodies at most this many instructions long are inlined.
const MAX_INLINE_INSTS: usize = 160;
/// A caller stops inlining once it has grown past this many instructions.
const MAX_CALLER_INSTS: usize = 4096;

pub(crate) fn enabled() -> bool {
    std::env::var("MIMAS_OPT").map_or(true, |v| v != "0")
}

/// Time per pass, for `MIMAS_COMPILE_TIMING=1` (stderr). Never reads the
/// clock otherwise: `Instant::now` panics on wasm32.
#[derive(Default)]
pub(crate) struct OptTimes {
    on: bool,
    tail: std::time::Duration,
    inline: std::time::Duration,
    values: std::time::Duration,
    fwd: std::time::Duration,
    scalar: std::time::Duration,
    promote: std::time::Duration,
    split: std::time::Duration,
}

impl OptTimes {
    pub(crate) fn new() -> Self {
        OptTimes {
            on: std::env::var_os("MIMAS_COMPILE_TIMING").is_some(),
            ..Default::default()
        }
    }
    fn now(&self) -> Option<std::time::Instant> {
        self.on.then(std::time::Instant::now)
    }
    fn add(acc: &mut std::time::Duration, t: Option<std::time::Instant>) {
        if let Some(t) = t {
            *acc += t.elapsed();
        }
    }
    pub(crate) fn report(&self) {
        if self.on {
            eprintln!(
                "OPT tail calls {:?} | inline {:?} | value adts {:?} | forward fields {:?} | scalar replace {:?} | promote locals {:?} | split phis {:?}",
                self.tail, self.inline, self.values, self.fwd, self.scalar, self.promote, self.split
            );
        }
    }
}

/// What the per-body passes read besides the body: the value-type adts and
/// the entry body's local decs (a `GetEntry`'s type).
pub(crate) struct Global {
    pub(crate) values: HashSet<AdtId>,
    pub(crate) entry_locals: Vec<DecId>,
}

/// `MIMAS_SSA=0` keeps the slot form (an A/B for the lanes' cost model).
pub(crate) fn ssa_enabled() -> bool {
    !std::env::var("MIMAS_SSA").is_ok_and(|v| v == "0")
}

/// The whole-program half of the optimizer: self tail calls, inlining
/// (callers in postorder, so a body is inlined with its own calls already
/// inlined) and the value-type analysis. After this each body is optimized
/// on its own by `optimize_body`, reading only itself and `Global`.
///
/// `frame_natives` are natives that read the live call stack
/// (`std::debug::caller_line` and kin): a body that calls one keeps its
/// frame, so what they report stays true.
/// `mutating_natives` are natives that mutate their receiver
/// (`mutates_recv`): an adt that reaches one is not a value type.
pub(crate) fn optimize_global(
    ir: &mut Ir,
    frame_natives: &HashSet<NativeId>,
    mutating_natives: &HashSet<NativeId>,
    times: &mut OptTimes,
) -> Global {
    let t = times.now();
    self_tail_calls(ir);
    OptTimes::add(&mut times.tail, t);
    let t = times.now();
    inline_small_calls(ir, frame_natives);
    OptTimes::add(&mut times.inline, t);
    let t = times.now();
    let values = value_adts(ir, mutating_natives);
    OptTimes::add(&mut times.values, t);
    let entry_locals: Vec<DecId> = ir.bodies[BodyId::ZERO].locals.values().copied().collect();
    Global {
        values,
        entry_locals,
    }
}

/// The per-body half: field forwarding, scalar replacement, locals to SSA
/// and struct phi splitting, to a fixpoint. Reads `body`, `g`, `res` (decl
/// types, adt member types and dispatch ids) and `shadows`, and writes only
/// `body`, which is what lets `bodycache` key a body's compiled form on
/// exactly those. The entry body is never optimized (see the module doc).
/// `shadows` keeps the debugger's copies of dissolved struct locals (see
/// the shadows section below) -- a cost only a VM a debugger reads pays.
pub(crate) fn optimize_body(
    body: &mut Body,
    g: &Global,
    res: &Resolutions,
    shadows: bool,
    times: &mut OptTimes,
) {
    let t = times.now();
    forward_value_fields(body, &g.values);
    OptTimes::add(&mut times.fwd, t);
    // a round can expose the next: a field that held a struct becomes a
    // local holding a struct
    let t = times.now();
    for _ in 0..4 {
        if !scalar_replace(body) {
            break;
        }
    }
    OptTimes::add(&mut times.scalar, t);
    // locals to SSA, then structs chosen by control flow to field phis;
    // a split can expose a local whose only writes are now structs.
    if !ssa_enabled() {
        return;
    }
    for _ in 0..3 {
        let t = times.now();
        let a = promote_locals(body, res, &g.entry_locals, shadows);
        OptTimes::add(&mut times.promote, t);
        let t = times.now();
        let b = split_struct_phis(body, &g.values, res, &g.entry_locals);
        OptTimes::add(&mut times.split, t);
        let t = times.now();
        let c = forward_value_fields(body, &g.values);
        OptTimes::add(&mut times.fwd, t);
        if !a && !b && !c {
            break;
        }
    }
}

// ------------------------------------------------------------ debugger shadows

/// A struct local the optimizer dissolves still has a name the debugger
/// shows, so the passes keep a `Body::shadows` tree per such local:
/// `scalar_replace` points it at the field slots it made (always live, even
/// under mutation), `promote_locals` gives a promoted leaf a shadow local
/// that `Inst::Shadow` copies every store into, and `split_struct_phis`
/// splits a shadow of a dissolved struct into one per field.
///
/// `l` is leaving the slot form: a named local, or a leaf of a tree. Returns
/// the shadow local its stores copy into, if the debugger can see it.
fn shadow_leaf_for(body: &mut Body, l: Local) -> Option<Local> {
    if body.artifacts.contains_key(&l) {
        let s = body.locals.push(DecId::DANGLING);
        body.shadows.insert(l, Shadow::Slot(s));
        return Some(s);
    }
    let root = body.shadow_leaf_root(l)?;
    let s = body.locals.push(DecId::DANGLING);
    let tree = body.shadows.get_mut(&root).unwrap();
    tree.replace_leaf(&l, Shadow::Slot(s));
    Some(s)
}

/// Leaf `s` of some tree becomes a struct of `adt` with fresh leaves, which
/// are returned (`None` when `s` is in no tree).
fn shadow_split_leaf(body: &mut Body, s: Local, adt: AdtId, n: usize) -> Option<Vec<Local>> {
    let root = body.shadow_leaf_root(s)?;
    let leaves: Vec<Local> = (0..n).map(|_| body.locals.push(DecId::DANGLING)).collect();
    let tree = body.shadows.get_mut(&root).unwrap();
    tree.replace_leaf(
        &s,
        Shadow::Struct {
            adt,
            fields: leaves.iter().map(|&l| Shadow::Slot(l)).collect(),
        },
    );
    Some(leaves)
}

// ------------------------------------------------------------ inlining

/// Can `bid` be copied into its callers: a small, capture-free body that
/// never reaches itself through direct calls, never `raise`s (`Op::Raise`
/// returns from the *frame*, which a copy no longer has), and calls no
/// native that reads the frame stack.
fn inlinable(
    ir: &Ir,
    bid: BodyId,
    calls: &HashMap<BodyId, Vec<BodyId>>,
    frame_natives: &HashSet<NativeId>,
) -> bool {
    if bid == BodyId::ZERO {
        return false;
    }
    let body = &ir.bodies[bid];
    if !body.captures.is_empty() {
        return false;
    }
    let n: usize = body.blocks.values().map(|b| b.stream.len()).sum();
    if n > MAX_INLINE_INSTS {
        return false;
    }
    for block in body.blocks.values() {
        for &iid in &block.stream {
            match &body.instructions[iid] {
                Inst::Raise(_) => return false,
                Inst::CallNative { id, .. } if frame_natives.contains(id)
                    || ir.resolutions.cleanup_natives.is_some_and(|(mark, take)| *id == mark || *id == take) => return false,
                _ => {}
            }
        }
    }
    // every param is read as a plain local; the entry checks may Panic, fine
    !reaches(calls, bid, bid)
}

fn reaches(calls: &HashMap<BodyId, Vec<BodyId>>, from: BodyId, to: BodyId) -> bool {
    let mut seen = HashSet::new();
    let mut work = vec![from];
    while let Some(b) = work.pop() {
        for &c in calls.get(&b).map(|v| v.as_slice()).unwrap_or(&[]) {
            if c == to {
                return true;
            }
            if seen.insert(c) {
                work.push(c);
            }
        }
    }
    false
}

fn direct_calls(body: &Body) -> Vec<BodyId> {
    let mut out = Vec::new();
    for block in body.blocks.values() {
        for &iid in &block.stream {
            if let Inst::CallDirect { body, .. } = &body_inst(body, iid) {
                out.push(*body);
            }
        }
    }
    out
}

fn body_inst(body: &Body, iid: InstId) -> &Inst {
    &body.instructions[iid]
}

fn inline_small_calls(ir: &mut Ir, frame_natives: &HashSet<NativeId>) {
    let calls: HashMap<BodyId, Vec<BodyId>> = ir
        .bodies
        .iter()
        .map(|(bid, body)| (bid, direct_calls(body)))
        .collect();
    let small: HashSet<BodyId> = ir
        .bodies
        .iter()
        .map(|(bid, _)| bid)
        .filter(|&bid| inlinable(ir, bid, &calls, frame_natives))
        .collect();
    if small.is_empty() {
        return;
    }
    // callees before callers, so a body is copied after its own calls were
    // inlined and the copy carries no further small calls
    let order = postorder(&calls, ir.bodies.len());
    for caller in order {
        if caller == BodyId::ZERO {
            continue;
        }
        loop {
            let size: usize = ir.bodies[caller]
                .blocks
                .values()
                .map(|b| b.stream.len())
                .sum();
            if size > MAX_CALLER_INSTS {
                break;
            }
            let Some((block, iid, callee)) = find_call(&ir.bodies[caller], &small, caller) else {
                break;
            };
            let snap = Snap::of(&ir.bodies[callee]);
            inline_at(&mut ir.bodies[caller], block, iid, callee, &snap);
        }
    }
}

fn postorder(calls: &HashMap<BodyId, Vec<BodyId>>, n: usize) -> Vec<BodyId> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    fn visit(
        b: BodyId,
        calls: &HashMap<BodyId, Vec<BodyId>>,
        seen: &mut HashSet<BodyId>,
        out: &mut Vec<BodyId>,
    ) {
        if !seen.insert(b) {
            return;
        }
        for &c in calls.get(&b).map(|v| v.as_slice()).unwrap_or(&[]) {
            visit(c, calls, seen, out);
        }
        out.push(b);
    }
    for i in 0..n {
        visit(BodyId::from(i as u32), calls, &mut seen, &mut out);
    }
    out
}

/// The first direct call in `body` to a small callee other than itself.
fn find_call(
    body: &Body,
    small: &HashSet<BodyId>,
    me: BodyId,
) -> Option<(BlockId, InstId, BodyId)> {
    for (bid, block) in body.blocks.iter() {
        for &iid in &block.stream {
            if let Inst::CallDirect { body: callee, .. } = &body.instructions[iid]
                && *callee != me
                && small.contains(callee)
            {
                return Some((bid, iid, *callee));
            }
        }
    }
    None
}

/// A callee's shape, copied out so the caller can be mutated while it's read.
struct Snap {
    blocks: Vec<Vec<InstId>>,
    insts: Vec<Inst>,
    locs: Vec<Location>,
    kinds: HashMap<InstId, crate::OperandKind>,
    locals: Vec<DecId>,
    params: Vec<Local>,
}

impl Snap {
    fn of(body: &Body) -> Self {
        Self {
            blocks: body.blocks.values().map(|b| b.stream.clone()).collect(),
            insts: body.instructions.values().cloned().collect(),
            locs: body.locs.values().cloned().collect(),
            kinds: body.kinds.clone(),
            locals: body.locals.values().cloned().collect(),
            params: body.params.clone(),
        }
    }
}

fn push_inst(body: &mut Body, inst: Inst, loc: Location) -> InstId {
    let id = body.instructions.push(inst);
    body.locs.push(loc);
    id
}

/// Replace the call at `iid` in `block` with a copy of the callee: the call's
/// block ends by binding the params and jumping into the copy; every copied
/// `return` stores into one fresh local and jumps to a continuation block
/// that starts with the rest of the original block, the call itself turned
/// into a read of that local -- so nothing that referenced the call's id
/// has to change.
fn inline_at(body: &mut Body, block: BlockId, iid: InstId, callee: BodyId, snap: &Snap) {
    let Inst::CallDirect { args, .. } = body.instructions[iid].clone() else {
        unreachable!("inline_at on a non-call");
    };
    let call_loc = body.locs[iid].clone();
    // fresh ids for everything the callee owns
    let local_map: Vec<Local> = snap
        .locals
        .iter()
        .map(|&dec| body.locals.push(dec))
        .collect();
    let ret = body.locals.push(DecId::DANGLING);
    let block_map: Vec<BlockId> = snap
        .blocks
        .iter()
        .map(|_| body.blocks.push(Block::new()))
        .collect();
    let cont = body.blocks.push(Block::new());
    let inst_map: Vec<InstId> = snap
        .insts
        .iter()
        .zip(&snap.locs)
        .map(|(_, loc)| push_inst(body, Inst::Panic, loc.clone()))
        .collect();
    for (i, inst) in snap.insts.iter().enumerate() {
        let mapped = match inst {
            Inst::Return(v) => Inst::SetLocal(ret, inst_map[v.index()]),
            other => remap(other, &inst_map, &local_map, &block_map),
        };
        body.instructions[inst_map[i]] = mapped;
        if let Some(k) = snap.kinds.get(&InstId::from(i as u32)) {
            body.kinds.insert(inst_map[i], *k);
        }
    }
    // the callee's blocks, a `return` becoming store + jump to the continuation
    for (bi, stream) in snap.blocks.iter().enumerate() {
        let mut out = Vec::with_capacity(stream.len() + 1);
        for &old in stream {
            out.push(inst_map[old.index()]);
            if matches!(snap.insts[old.index()], Inst::Return(_)) {
                let j = push_inst(
                    body,
                    Inst::Jump { target: cont },
                    snap.locs[old.index()].clone(),
                );
                out.push(j);
            }
        }
        body.blocks[block_map[bi]].stream = out;
    }
    // split the call's block: params bound, jump in; the rest moves to `cont`
    let stream = std::mem::take(&mut body.blocks[block].stream);
    let at = stream
        .iter()
        .position(|&x| x == iid)
        .expect("call in its block");
    let (pre, post) = stream.split_at(at);
    let mut head = pre.to_vec();
    // the call's footprint for the trace's call graph
    let te = push_inst(body, Inst::TraceEdge { callee }, call_loc);
    head.push(te);
    for (p, arg) in snap.params.iter().zip(&args) {
        let s = push_inst(
            body,
            Inst::SetLocal(local_map[p.index()], *arg),
            call_loc.clone(),
        );
        head.push(s);
    }
    let j = push_inst(
        body,
        Inst::Jump {
            target: block_map[0],
        },
        call_loc.clone(),
    );
    head.push(j);
    body.blocks[block].stream = head;
    body.instructions[iid] = Inst::GetLocal(ret);
    body.blocks[cont].stream = post.to_vec();
    // phis that named the split block as a predecessor now see `cont`, where
    // its terminator went
    for inst in body.instructions.values_mut() {
        if let Inst::Phi(branches) = inst {
            for (b, _) in branches.iter_mut() {
                if *b == block {
                    *b = cont;
                }
            }
        }
    }
}

/// `inst` with every instruction, local and block reference mapped. Entry
/// locals (`GetEntry`/`SetEntry`) index the entry body and stay as they are.
fn remap(inst: &Inst, im: &[InstId], lm: &[Local], bm: &[BlockId]) -> Inst {
    let i = |x: &InstId| im[x.index()];
    let l = |x: &Local| lm[x.index()];
    let b = |x: &BlockId| bm[x.index()];
    match inst {
        Inst::Constant(c) => Inst::Constant(c.clone()),
        Inst::SetLocal(x, v) => Inst::SetLocal(l(x), i(v)),
        Inst::Shadow { local, value } => Inst::Shadow {
            local: l(local),
            value: i(value),
        },
        Inst::GetLocal(x) => Inst::GetLocal(l(x)),
        Inst::SetEntry(x, v) => Inst::SetEntry(*x, i(v)),
        Inst::GetEntry(x) => Inst::GetEntry(*x),
        Inst::BinOp {
            left,
            op,
            right,
            kind,
        } => Inst::BinOp {
            left: i(left),
            op: *op,
            right: i(right),
            kind: *kind,
        },
        Inst::UnaryOp { op, right } => Inst::UnaryOp {
            op: *op,
            right: i(right),
        },
        Inst::JumpIfFalse { condition, target } => Inst::JumpIfFalse {
            condition: i(condition),
            target: b(target),
        },
        Inst::Jump { target } => Inst::Jump { target: b(target) },
        Inst::Switch {
            scrut,
            base,
            table,
            default,
        } => Inst::Switch {
            scrut: i(scrut),
            base: *base,
            table: table.iter().map(b).collect(),
            default: b(default),
        },
        Inst::Phi(branches) => Inst::Phi(branches.iter().map(|(bb, v)| (b(bb), i(v))).collect()),
        Inst::ForNext { idx, bound, target } => Inst::ForNext {
            idx: i(idx),
            bound: i(bound),
            target: b(target),
        },
        Inst::NewArray => Inst::NewArray,
        Inst::NewDict => Inst::NewDict,
        Inst::SetIndex { set, index, value } => Inst::SetIndex {
            set: i(set),
            index: i(index),
            value: i(value),
        },
        Inst::GetIndex { set, index, kind } => Inst::GetIndex {
            set: i(set),
            index: i(index),
            kind: *kind,
        },
        Inst::GetField { src, slot, kind } => Inst::GetField {
            src: i(src),
            slot: *slot,
            kind: *kind,
        },
        Inst::SetField {
            receiver,
            slot,
            value,
        } => Inst::SetField {
            receiver: i(receiver),
            slot: *slot,
            value: i(value),
        },
        Inst::Push { array, value } => Inst::Push {
            array: i(array),
            value: i(value),
        },
        Inst::Insert { dict, key, value } => Inst::Insert {
            dict: i(dict),
            key: *key,
            value: i(value),
        },
        Inst::Len(v) => Inst::Len(i(v)),
        Inst::ToFloat(v) => Inst::ToFloat(i(v)),
        Inst::Sqrt(v) => Inst::Sqrt(i(v)),
        Inst::Unwrap(v) => Inst::Unwrap(i(v)),
        Inst::UnwrapUnit(v) => Inst::UnwrapUnit(i(v)),
        Inst::In(a, c, f) => Inst::In(i(a), i(c), *f),
        Inst::Format(parts) => Inst::Format(
            parts
                .iter()
                .map(|p| match p {
                    FormatPart::Value(v) => FormatPart::Value(i(v)),
                    lit => *lit,
                })
                .collect(),
        ),
        Inst::RefBody(x) => Inst::RefBody(*x),
        Inst::MakeClosure { body, captures } => Inst::MakeClosure {
            body: *body,
            captures: captures.iter().map(i).collect(),
        },
        Inst::Call { callee, args } => Inst::Call {
            callee: i(callee),
            args: args.iter().map(i).collect(),
        },
        Inst::CallDirect { body, args } => Inst::CallDirect {
            body: *body,
            args: args.iter().map(i).collect(),
        },
        Inst::CallNative { id, args } => Inst::CallNative {
            id: *id,
            args: args.iter().map(i).collect(),
        },
        Inst::Return(v) => Inst::Return(i(v)),
        Inst::NewInstance { adt, fields } => Inst::NewInstance {
            adt: *adt,
            fields: fields.iter().map(i).collect(),
        },
        Inst::IsInstance { src, adt } => Inst::IsInstance {
            src: i(src),
            adt: *adt,
        },
        Inst::Panic => Inst::Panic,
        Inst::Raise(v) => Inst::Raise(i(v)),
        Inst::IsRaised(v) => Inst::IsRaised(i(v)),
        Inst::UnwrapRaised(v) => Inst::UnwrapRaised(i(v)),
        Inst::TraceEdge { callee } => Inst::TraceEdge { callee: *callee },
    }
}

// ---------------------------------------------------- scalar replacement

/// The users of every instruction, by use site.
fn users_of(body: &Body) -> HashMap<InstId, Vec<InstId>> {
    let mut out: HashMap<InstId, Vec<InstId>> = HashMap::new();
    for block in body.blocks.values() {
        for &iid in &block.stream {
            for u in uses(&body.instructions[iid]) {
                out.entry(u).or_default().push(iid);
            }
        }
    }
    out
}

/// Block successors, from each block's terminator(s).
fn successors(body: &Body) -> Vec<Vec<BlockId>> {
    body.blocks
        .values()
        .map(|b| {
            let mut out = Vec::new();
            for &iid in &b.stream {
                match &body.instructions[iid] {
                    Inst::Jump { target }
                    | Inst::JumpIfFalse { target, .. }
                    | Inst::ForNext { target, .. } => out.push(*target),
                    Inst::Switch { table, default, .. } => {
                        out.extend(table.iter().copied());
                        out.push(*default);
                    }
                    _ => {}
                }
            }
            out
        })
        .collect()
}

/// Dominance over a body's blocks: the dominator tree (`idom`) and a
/// preorder/postorder numbering of it, so "does `d` dominate `b`" is two
/// compares. Unreachable blocks have no number, so nothing in them counts
/// as dominated (not even by themselves).
struct Dom {
    /// Immediate dominator of each reachable non-entry block.
    idom: Vec<Option<BlockId>>,
    pre: Vec<u32>,
    post: Vec<u32>,
}

impl Dom {
    /// Does `d` dominate `b`?
    fn contains(&self, b: BlockId, d: BlockId) -> bool {
        let (b, d) = (b.index(), d.index());
        self.reachable(b)
            && self.reachable(d)
            && self.pre[d] <= self.pre[b]
            && self.post[b] <= self.post[d]
    }
    fn reachable(&self, b: usize) -> bool {
        self.pre[b] != u32::MAX
    }
}

/// Immediate dominators by Cooper, Harvey and Kennedy's iterative scheme ("A
/// Simple, Fast Dominance Algorithm"): sweep the reachable blocks in reverse
/// postorder, each block's idom the intersection (a walk up the tree by
/// postorder rank) of its processed predecessors'. Converges in a couple of
/// sweeps on structured code; each step touches only tree paths, where the
/// bitset fixpoint it replaced did a word-AND per block per pred and then
/// found each idom by scanning and counting every strict dominator's row
/// (quadratic in blocks: 48 ms of florilegium's compile).
fn dominators(body: &Body) -> Dom {
    let succ = successors(body);
    let preds = predecessors(&succ);
    dominators_of(&succ, &preds)
}

fn dominators_of(succ: &[Vec<BlockId>], preds: &[Vec<BlockId>]) -> Dom {
    let n = succ.len();

    // postorder DFS over the reachable region; `rpo` is its reverse and
    // `rank` the postorder number (higher is earlier in rpo)
    let mut rank = vec![u32::MAX; n];
    let mut rpo: Vec<u32> = Vec::new();
    if n > 0 {
        let mut seen = vec![false; n];
        let mut post: Vec<u32> = Vec::with_capacity(n);
        let mut stack: Vec<(u32, usize)> = vec![(0, 0)];
        seen[0] = true;
        while let Some(top) = stack.last_mut() {
            let (b, ci) = (top.0, top.1);
            if ci < succ[b as usize].len() {
                let t = succ[b as usize][ci].index() as u32;
                top.1 += 1;
                if !seen[t as usize] {
                    seen[t as usize] = true;
                    stack.push((t, 0));
                }
            } else {
                rank[b as usize] = post.len() as u32;
                post.push(b);
                stack.pop();
            }
        }
        rpo = post.into_iter().rev().collect();
    }

    let mut idom = vec![u32::MAX; n];
    if n > 0 {
        idom[0] = 0;
    }
    let intersect = |idom: &[u32], mut a: u32, mut b: u32| -> u32 {
        while a != b {
            while rank[a as usize] < rank[b as usize] {
                a = idom[a as usize];
            }
            while rank[b as usize] < rank[a as usize] {
                b = idom[b as usize];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let mut new: Option<u32> = None;
            for &p in &preds[b as usize] {
                let p = p.index() as u32;
                if idom[p as usize] == u32::MAX {
                    continue; // unreachable, or not yet processed
                }
                new = Some(match new {
                    None => p,
                    Some(x) => intersect(&idom, p, x),
                });
            }
            if let Some(d) = new {
                if idom[b as usize] != d {
                    idom[b as usize] = d;
                    changed = true;
                }
            }
        }
    }

    // the tree, then its pre/post numbering by an explicit DFS
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &b in rpo.iter().skip(1) {
        children[idom[b as usize] as usize].push(b);
    }
    let mut pre = vec![u32::MAX; n];
    let mut postn = vec![u32::MAX; n];
    if n > 0 {
        let (mut tp, mut tq) = (0u32, 0u32);
        let mut stack: Vec<(u32, usize)> = vec![(0, 0)];
        pre[0] = tp;
        tp += 1;
        while let Some(top) = stack.last_mut() {
            let (b, ci) = (top.0, top.1);
            if ci < children[b as usize].len() {
                top.1 += 1;
                let c = children[b as usize][ci];
                pre[c as usize] = tp;
                tp += 1;
                stack.push((c, 0));
            } else {
                postn[b as usize] = tq;
                tq += 1;
                stack.pop();
            }
        }
    }
    Dom {
        idom: (0..n)
            .map(|i| (i != 0 && idom[i] != u32::MAX).then(|| BlockId::from(idom[i])))
            .collect(),
        pre,
        post: postn,
    }
}

/// Where each instruction sits: its block and position in the stream, by
/// instruction index (a dense table: the passes ask it once per read and
/// write, and a hashed map was most of promote_locals' scan).
struct Positions(Vec<Option<(BlockId, usize)>>);

impl Positions {
    fn get(&self, iid: &InstId) -> Option<&(BlockId, usize)> {
        self.0.get(iid.index()).and_then(Option::as_ref)
    }
}

impl std::ops::Index<&InstId> for Positions {
    type Output = (BlockId, usize);
    fn index(&self, iid: &InstId) -> &(BlockId, usize) {
        self.get(iid).expect("instruction not in a block")
    }
}

fn positions(body: &Body) -> Positions {
    let mut out = vec![None; body.instructions.len()];
    for (bid, block) in body.blocks.iter() {
        for (i, &iid) in block.stream.iter().enumerate() {
            out[iid.index()] = Some((bid, i));
        }
    }
    Positions(out)
}

/// Replace struct locals with one local per field, so the struct is never
/// allocated. The unit of replacement is an *object class*: a root local
/// whose single store is a `make_instance` nothing else uses, plus every
/// local that is a plain copy of a class member (`set l2: get l1`), as
/// inlining leaves behind for returns and parameters. Members share the
/// field locals, which is exactly what aliasing means. The rule that keeps
/// it sound: every member is written exactly once, that write dominates all
/// of its reads, and every read is either a field access on the object or
/// the copy into another member. A local a parameter, a capture, an entry
/// slot, or anything else can see keeps its object and its identity.
/// Returns whether anything changed.
fn scalar_replace(body: &mut Body) -> bool {
    let users = users_of(body);
    let pos = positions(body);
    let dom = dominators(body);
    let dominated = |w: InstId, r: InstId| -> bool {
        match (pos.get(&w), pos.get(&r)) {
            (Some(&(bw, pw)), Some(&(br, pr))) => {
                (bw == br && pw < pr) || (bw != br && dom.contains(br, bw))
            }
            _ => false,
        }
    };
    let mut writes: HashMap<Local, Vec<InstId>> = HashMap::new();
    let mut reads: HashMap<Local, Vec<InstId>> = HashMap::new();
    let mut bad: HashSet<Local> = body.params.iter().chain(&body.captures).copied().collect();
    for (bid, block) in body.blocks.iter() {
        let _ = bid;
        for &iid in &block.stream {
            match &body.instructions[iid] {
                Inst::SetLocal(l, _) => writes.entry(*l).or_default().push(iid),
                Inst::GetLocal(l) => reads.entry(*l).or_default().push(iid),
                Inst::SetEntry(l, _) | Inst::GetEntry(l) => {
                    bad.insert(*l);
                }
                _ => {}
            }
        }
    }
    // single-write locals whose write dominates every read
    let mut single: HashMap<Local, InstId> = HashMap::new();
    for (&l, ws) in &writes {
        if bad.contains(&l) || ws.len() != 1 {
            continue;
        }
        let w = ws[0];
        if reads
            .get(&l)
            .map_or(true, |rs| rs.iter().all(|&r| dominated(w, r)))
        {
            single.insert(l, w);
        }
    }
    let single_use = |v: InstId| users.get(&v).map_or(0, |u| u.len()) == 1;
    // roots and copies
    let mut root_of: HashMap<Local, Local> = HashMap::new(); // member -> root
    let mut root_adt: HashMap<Local, (usize, InstId)> = HashMap::new(); // root -> (nfields, the make_instance)
    for (&l, &w) in &single {
        let Inst::SetLocal(_, v) = body.instructions[w] else {
            unreachable!()
        };
        if let Inst::NewInstance { fields, .. } = &body.instructions[v]
            && single_use(v)
        {
            root_of.insert(l, l);
            root_adt.insert(l, (fields.len(), v));
        }
    }
    // copies chain to a root through other copies; iterate until stable
    let mut changed = true;
    while changed {
        changed = false;
        for (&l, &w) in &single {
            if root_of.contains_key(&l) {
                continue;
            }
            let Inst::SetLocal(_, v) = body.instructions[w] else {
                unreachable!()
            };
            if let Inst::GetLocal(src) = body.instructions[v]
                && single_use(v)
                && let Some(&r) = root_of.get(&src)
            {
                root_of.insert(l, r);
                changed = true;
            }
        }
    }
    // every read of every member must be a field op on it or a copy into a member of the same class
    let mut members: HashMap<Local, Vec<Local>> = HashMap::new();
    for (&m, &r) in &root_of {
        members.entry(r).or_default().push(m);
    }
    let mut ok_roots: Vec<Local> = Vec::new();
    'root: for (&r, ms) in &members {
        let nfields = root_adt[&r].0;
        for &m in ms {
            for &g in reads.get(&m).map(|v| v.as_slice()).unwrap_or(&[]) {
                for &u in users.get(&g).map(|v| v.as_slice()).unwrap_or(&[]) {
                    match &body.instructions[u] {
                        Inst::GetField { src, slot, .. }
                            if *src == g && (*slot as usize) < nfields => {}
                        Inst::SetField {
                            receiver,
                            slot,
                            value,
                        } if *receiver == g && *value != g && (*slot as usize) < nfields => {}
                        Inst::SetLocal(dst, v) if *v == g && root_of.get(dst) == Some(&r) => {}
                        _ => continue 'root,
                    }
                }
            }
        }
        ok_roots.push(r);
    }
    if ok_roots.is_empty() {
        return false;
    }
    ok_roots.sort_by_key(|l| l.index());
    for r in ok_roots {
        let (nfields, ni) = root_adt[&r];
        let ms = &members[&r];
        let fields: Vec<Local> = (0..nfields)
            .map(|_| body.locals.push(DecId::DANGLING))
            .collect();
        let Inst::NewInstance { fields: vals, adt } = body.instructions[ni].clone() else {
            unreachable!()
        };
        // the debugger keeps seeing each named member: as the struct of the
        // field slots themselves, live under every later store
        for &m in ms {
            let tree = Shadow::Struct {
                adt,
                fields: fields.iter().map(|&f| Shadow::Slot(f)).collect(),
            };
            if body.artifacts.contains_key(&m) {
                body.shadows.insert(m, tree);
            } else if let Some(root) = body.shadow_leaf_root(m) {
                body.shadows.get_mut(&root).unwrap().replace_leaf(&m, tree);
            }
        }
        let root_write = single[&r];
        let copy_writes: HashSet<InstId> =
            ms.iter().filter(|&&m| m != r).map(|m| single[m]).collect();
        for bi in 0..body.blocks.len() {
            let bid = BlockId::from(bi as u32);
            let stream = std::mem::take(&mut body.blocks[bid].stream);
            let mut out = Vec::with_capacity(stream.len() + nfields);
            for iid in stream {
                if iid == root_write {
                    // the aggregate store becomes one store per field
                    let loc = body.locs[iid];
                    for (k, &fv) in vals.iter().enumerate() {
                        if k == 0 {
                            body.instructions[iid] = Inst::SetLocal(fields[0], fv);
                            out.push(iid);
                        } else {
                            let s = push_inst(body, Inst::SetLocal(fields[k], fv), loc);
                            out.push(s);
                        }
                    }
                    continue;
                }
                if copy_writes.contains(&iid) {
                    // a copy between members is the identity: drop it
                    continue;
                }
                out.push(iid);
            }
            body.blocks[bid].stream = out;
        }
        for &m in ms {
            for &g in reads.get(&m).map(|v| v.as_slice()).unwrap_or(&[]) {
                for &u in users.get(&g).map(|v| v.as_slice()).unwrap_or(&[]) {
                    body.instructions[u] = match &body.instructions[u] {
                        Inst::GetField { slot, .. } => Inst::GetLocal(fields[*slot as usize]),
                        Inst::SetField { slot, value, .. } => {
                            Inst::SetLocal(fields[*slot as usize], *value)
                        }
                        // the copy was dropped from its stream; leave a harmless
                        // dead read in its place. a fieldless aggregate has no
                        // field slots to read, so it reads the member's own local
                        Inst::SetLocal(..) => Inst::GetLocal(fields.first().copied().unwrap_or(m)),
                        _ => unreachable!("checked above"),
                    };
                }
            }
        }
    }
    true
}

// ------------------------------------------------------------ SSA for locals

/// Block predecessors, from the successor lists.
fn predecessors(succ: &[Vec<BlockId>]) -> Vec<Vec<BlockId>> {
    let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); succ.len()];
    for (i, ss) in succ.iter().enumerate() {
        for &t in ss {
            let p = BlockId::from(i as u32);
            if !preds[t.index()].contains(&p) {
                preds[t.index()].push(p);
            }
        }
    }
    preds
}

/// Codegen hosts a phi's copies only before an unconditional `jump`, so a
/// block that will carry phis must be entered by `jump`s alone. Every edge
/// into a block with two or more predecessors that comes from a
/// `jump_if_false`, `for_next` or `switch` gets a forwarder block in
/// between. `clean` leaves such forwarders alone (its "dangerous target"
/// rule); the ones it can fold, it does.
fn split_critical_edges(body: &mut Body) -> bool {
    let succ = successors(body);
    let preds = predecessors(&succ);
    let mut changed = false;
    for (bi, ps) in preds.iter().enumerate() {
        if ps.len() < 2 {
            continue;
        }
        let b = BlockId::from(bi as u32);
        for &p in ps {
            let stream = body.blocks[p].stream.clone();
            for iid in stream {
                let loc = body.locs[iid];
                let conditional = match &body.instructions[iid] {
                    Inst::JumpIfFalse { target, .. } | Inst::ForNext { target, .. } => *target == b,
                    Inst::Switch { table, default, .. } => *default == b || table.contains(&b),
                    _ => false,
                };
                if !conditional {
                    continue;
                }
                let f = body.blocks.push(Block::new());
                let j = push_inst(body, Inst::Jump { target: b }, loc);
                body.blocks[f].stream = vec![j];
                match &mut body.instructions[iid] {
                    Inst::JumpIfFalse { target, .. } | Inst::ForNext { target, .. } => *target = f,
                    Inst::Switch { table, default, .. } => {
                        if *default == b {
                            *default = f;
                        }
                        for t in table.iter_mut() {
                            if *t == b {
                                *t = f;
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                // a phi already in `b` that named `p` now sees `f`
                for &pi in &body.blocks[b].stream.clone() {
                    if let Inst::Phi(branches) = &mut body.instructions[pi] {
                        for (pb, _) in branches.iter_mut() {
                            if *pb == p {
                                *pb = f;
                            }
                        }
                    }
                }
                changed = true;
            }
        }
    }
    changed
}

struct Cfg {
    succ: Vec<Vec<BlockId>>,
    dom: Dom,
    children: Vec<Vec<BlockId>>,
    frontier: Vec<HashSet<BlockId>>,
}

fn cfg_of(body: &Body) -> Cfg {
    let succ = successors(body);
    let preds = predecessors(&succ);
    let dom = dominators_of(&succ, &preds);
    let n = body.blocks.len();
    let idom = dom.idom.clone();
    let mut children: Vec<Vec<BlockId>> = vec![Vec::new(); n];
    for (i, d) in idom.iter().enumerate() {
        if let Some(d) = d {
            children[d.index()].push(BlockId::from(i as u32));
        }
    }
    let mut frontier: Vec<HashSet<BlockId>> = vec![HashSet::new(); n];
    for i in 0..n {
        let b = BlockId::from(i as u32);
        if preds[i].len() < 2 || !dom.reachable(i) {
            continue;
        }
        for &p in &preds[i] {
            if !dom.reachable(p.index()) {
                continue;
            }
            let mut runner = Some(p);
            while let Some(r) = runner {
                if Some(r) == idom[i] {
                    break;
                }
                frontier[r.index()].insert(b);
                runner = idom[r.index()];
            }
        }
    }
    Cfg {
        succ,
        dom,
        children,
        frontier,
    }
}

/// Promote locals to SSA values: every read of a promoted local becomes
/// the definition that reaches it, with phis at dominance frontiers, and
/// the slot disappears. Not promoted: params and captures (written by the
/// caller), entry slots, a `for_next` loop variable (the op increments the
/// slot's register in place), and a local with a read no write dominates.
/// A *named* local that holds scalars keeps its stores (reads are still
/// forwarded) so the step debugger's frame view still shows it; a named
/// local that only ever holds a struct drops them, because the struct is
/// about to stop existing (see `split_struct_phis`). Returns whether
/// anything changed.
fn promote_locals(
    body: &mut Body,
    res: &Resolutions,
    entry_locals: &[DecId],
    shadows: bool,
) -> bool {
    split_critical_edges(body);
    let cfg = cfg_of(body);
    let pos = positions(body);
    let dominated = |w: InstId, r: InstId| -> bool {
        match (pos.get(&w), pos.get(&r)) {
            (Some(&(bw, pw)), Some(&(br, pr))) => {
                (bw == br && pw < pr) || (bw != br && cfg.dom.contains(br, bw))
            }
            _ => false,
        }
    };
    let mut writes: HashMap<Local, Vec<InstId>> = HashMap::new();
    let mut reads: HashMap<Local, Vec<InstId>> = HashMap::new();
    let mut bad: HashSet<Local> = body.params.iter().chain(&body.captures).copied().collect();
    for block in body.blocks.values() {
        for &iid in &block.stream {
            match &body.instructions[iid] {
                Inst::SetLocal(l, _) => writes.entry(*l).or_default().push(iid),
                Inst::GetLocal(l) => reads.entry(*l).or_default().push(iid),
                Inst::SetEntry(l, _) | Inst::GetEntry(l) => {
                    bad.insert(*l);
                }
                Inst::ForNext { idx, .. } => {
                    if let Inst::GetLocal(l) = body.instructions[*idx] {
                        bad.insert(l);
                    }
                }
                _ => {}
            }
        }
    }
    // a named local that holds scalars stays a slot: its reads are already
    // free (codegen aliases a read to the slot's register) and SSA would
    // only add a phi copy per merge on top of the store the debugger's frame
    // view needs. A named local whose every write is a struct -- built here,
    // copied from another such local, or chosen by an `if` -- is promoted,
    // because that struct is what `split_struct_phis` can then dissolve.
    // Unnamed locals (inlining's and scalar replacement's) always promote.
    let mut struct_local: HashSet<Local> = HashSet::new();
    loop {
        let mut grew = false;
        for (&l, ws) in &writes {
            if struct_local.contains(&l) {
                continue;
            }
            let all = ws.iter().all(|&w| {
                let Inst::SetLocal(_, v) = body.instructions[w] else {
                    return false;
                };
                match &body.instructions[v] {
                    Inst::NewInstance { .. } => true,
                    Inst::GetLocal(m) if struct_local.contains(m) => true,
                    Inst::Phi(br)
                        if br.iter().all(|(_, x)| {
                            matches!(body.instructions[*x], Inst::NewInstance { .. })
                        }) =>
                    {
                        true
                    }
                    // a typed struct from anywhere: a param, a field, an element
                    _ => ty_of(res, entry_locals, body, v, 0)
                        .as_ref()
                        .and_then(adt_of_ty)
                        .is_some(),
                }
            });
            if all {
                struct_local.insert(l);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    let mut promote: Vec<Local> = Vec::new();
    for (&l, ws) in &writes {
        if bad.contains(&l) {
            continue;
        }
        let named = body.locals[l] != DecId::DANGLING;
        if named && !struct_local.contains(&l) {
            continue;
        }
        let ok = reads.get(&l).map_or(true, |rs| {
            rs.iter().all(|&r| ws.iter().any(|&w| dominated(w, r)))
        });
        if ok {
            promote.push(l);
        }
    }
    // a local whose store reads a slot that is written and stays a slot
    // cannot be promoted: its reads would alias the slot, not snapshot it
    loop {
        let set: HashSet<Local> = promote.iter().copied().collect();
        let before = promote.len();
        promote.retain(|&l| {
            writes[&l].iter().all(|&w| {
                let Inst::SetLocal(_, v) = body.instructions[w] else {
                    return false;
                };
                match body.instructions[v] {
                    Inst::GetLocal(src) => set.contains(&src) || !writes.contains_key(&src),
                    _ => true,
                }
            })
        });
        if promote.len() == before {
            break;
        }
    }
    if promote.is_empty() {
        return false;
    }
    promote.sort_by_key(|l| l.index());
    let mut promote_set = vec![false; body.locals.len()];
    for &l in &promote {
        promote_set[l.index()] = true;
    }
    // the debugger's copy: every store to a local it can see also lands in a
    // shadow slot, as an `Inst::Shadow` right after the store (the value
    // operand follows the renaming below like any other). Without shadows a
    // tree whose leaf is leaving its slot is dropped: the view shows the
    // local's own unwritten slot, null, rather than a stale part
    for &l in &promote {
        if !shadows {
            if let Some(root) = body.shadow_leaf_root(l) {
                body.shadows.remove(&root);
            }
            continue;
        }
        let Some(s) = shadow_leaf_for(body, l) else {
            continue;
        };
        for &w in &writes[&l] {
            let Inst::SetLocal(_, v) = body.instructions[w] else {
                continue;
            };
            let (blk, _) = pos[&w];
            let sh = push_inst(body, Inst::Shadow { local: s, value: v }, body.locs[w]);
            let st = &mut body.blocks[blk].stream;
            let at = st.iter().position(|&x| x == w).map_or(st.len(), |i| i + 1);
            st.insert(at, sh);
        }
    }
    // phis at the iterated dominance frontier of each local's definition
    // blocks, listed per block (in `promote` order) so renaming visits a
    // block's own phis, not every promoted local at every block
    let mut phis_at: Vec<Vec<(Local, InstId)>> = vec![Vec::new(); body.blocks.len()];
    for &l in &promote {
        let mut work: Vec<BlockId> = writes[&l].iter().map(|w| pos[w].0).collect();
        let mut placed: HashSet<BlockId> = HashSet::new();
        while let Some(b) = work.pop() {
            for &f in &cfg.frontier[b.index()] {
                if placed.insert(f) {
                    let loc = body.blocks[f]
                        .stream
                        .first()
                        .map(|&i| body.locs[i])
                        .unwrap_or_default();
                    let phi = push_inst(body, Inst::Phi(Vec::new()), loc);
                    body.blocks[f].stream.insert(0, phi);
                    phis_at[f.index()].push((l, phi));
                    work.push(f);
                }
            }
        }
    }
    // renaming, down the dominator tree
    let mut undef: Option<InstId> = None;
    // promoted GetLocal -> its def, by instruction index (u32::MAX: none);
    // every operand of every instruction is resolved through it below
    let mut rewrite: Vec<u32> = vec![u32::MAX; body.instructions.len()];
    let mut stores: Vec<(InstId, Local, InstId)> = Vec::new(); // promoted SetLocal, its local, resolved value
    let mut stack: Vec<Vec<InstId>> = vec![Vec::new(); body.locals.len()];
    let resolve = |v: InstId, rewrite: &[u32]| -> InstId {
        let mut v = v;
        while let Some(&n) = rewrite.get(v.index()) {
            if n == u32::MAX {
                break;
            }
            v = InstId::from(n);
        }
        v
    };
    // explicit DFS: (block, number of defs pushed per local to pop on exit)
    let mut dfs: Vec<(BlockId, bool)> = vec![(BlockId::ZERO, false)];
    let mut pushed: Vec<Vec<Local>> = Vec::new();
    while let Some((b, leaving)) = dfs.pop() {
        if leaving {
            for l in pushed.pop().unwrap() {
                stack[l.index()].pop();
            }
            continue;
        }
        let mut mine: Vec<Local> = Vec::new();
        for &(l, phi) in &phis_at[b.index()] {
            stack[l.index()].push(phi);
            mine.push(l);
        }
        for &iid in &body.blocks[b].stream.clone() {
            match body.instructions[iid].clone() {
                Inst::GetLocal(l) if promote_set.get(l.index()).copied().unwrap_or(false) => {
                    let def = match stack[l.index()].last() {
                        Some(&d) => d,
                        None => *undef.get_or_insert_with(|| {
                            let u = push_inst(
                                body,
                                Inst::Constant(crate::Constant::Null),
                                Location::default(),
                            );
                            body.blocks[BlockId::ZERO].stream.insert(0, u);
                            u
                        }),
                    };
                    rewrite[iid.index()] = def.index() as u32;
                }
                Inst::SetLocal(l, v) if promote_set.get(l.index()).copied().unwrap_or(false) => {
                    let v = resolve(v, &rewrite);
                    stack[l.index()].push(v);
                    mine.push(l);
                    stores.push((iid, l, v));
                }
                _ => {}
            }
        }
        for &s in &cfg.succ[b.index()] {
            for &(l, phi) in &phis_at[s.index()] {
                {
                    let def = match stack[l.index()].last() {
                        Some(&d) => d,
                        None => *undef.get_or_insert_with(|| {
                            let u = push_inst(
                                body,
                                Inst::Constant(crate::Constant::Null),
                                Location::default(),
                            );
                            body.blocks[BlockId::ZERO].stream.insert(0, u);
                            u
                        }),
                    };
                    if let Inst::Phi(branches) = &mut body.instructions[phi] {
                        branches.push((b, def));
                    }
                }
            }
        }
        pushed.push(mine);
        dfs.push((b, true));
        for &c in cfg.children[b.index()].iter().rev() {
            dfs.push((c, false));
        }
    }
    // apply: operands through the rewrite map; the slot's reads and stores go
    let mut drop = rewrite.iter().map(|&r| r != u32::MAX).collect::<Vec<bool>>();
    drop.resize(body.instructions.len(), false);
    for (iid, _, _) in &stores {
        drop[iid.index()] = true;
    }
    for inst in body.instructions.values_mut() {
        map_operands(inst, &|x| resolve(x, &rewrite));
    }
    for block in body.blocks.values_mut() {
        block.stream.retain(|iid| !drop[iid.index()]);
    }
    true
}

/// `inst` with every instruction operand mapped (locals and blocks untouched).
fn map_operands(inst: &mut Inst, f: &dyn Fn(InstId) -> InstId) {
    match inst {
        Inst::Constant(_)
        | Inst::GetLocal(_)
        | Inst::GetEntry(_)
        | Inst::Jump { .. }
        | Inst::NewArray
        | Inst::NewDict
        | Inst::RefBody(_)
        | Inst::Panic
        | Inst::TraceEdge { .. } => {}
        Inst::SetLocal(_, v) | Inst::SetEntry(_, v) | Inst::Shadow { value: v, .. } => *v = f(*v),
        Inst::BinOp { left, right, .. } => {
            *left = f(*left);
            *right = f(*right);
        }
        Inst::UnaryOp { right, .. } => *right = f(*right),
        Inst::JumpIfFalse { condition, .. } => *condition = f(*condition),
        Inst::Switch { scrut, .. } => *scrut = f(*scrut),
        Inst::Phi(branches) => {
            for (_, v) in branches.iter_mut() {
                *v = f(*v);
            }
        }
        Inst::ForNext { idx, bound, .. } => {
            *idx = f(*idx);
            *bound = f(*bound);
        }
        Inst::SetIndex { set, index, value } => {
            *set = f(*set);
            *index = f(*index);
            *value = f(*value);
        }
        Inst::GetIndex { set, index, .. } => {
            *set = f(*set);
            *index = f(*index);
        }
        Inst::GetField { src, .. } => *src = f(*src),
        Inst::SetField {
            receiver, value, ..
        } => {
            *receiver = f(*receiver);
            *value = f(*value);
        }
        Inst::Push { array, value } => {
            *array = f(*array);
            *value = f(*value);
        }
        Inst::Insert { value, .. } => *value = f(*value),
        Inst::Len(v)
        | Inst::ToFloat(v)
        | Inst::Sqrt(v)
        | Inst::Unwrap(v)
        | Inst::UnwrapUnit(v)
        | Inst::Return(v)
        | Inst::Raise(v)
        | Inst::IsRaised(v)
        | Inst::UnwrapRaised(v) => *v = f(*v),
        Inst::In(a, b, _) => {
            *a = f(*a);
            *b = f(*b);
        }
        Inst::Format(parts) => {
            for p in parts.iter_mut() {
                if let FormatPart::Value(v) = p {
                    *v = f(*v);
                }
            }
        }
        Inst::MakeClosure { captures, .. } => {
            for c in captures.iter_mut() {
                *c = f(*c);
            }
        }
        Inst::Call { callee, args } => {
            *callee = f(*callee);
            for a in args.iter_mut() {
                *a = f(*a);
            }
        }
        Inst::CallDirect { args, .. } | Inst::CallNative { args, .. } => {
            for a in args.iter_mut() {
                *a = f(*a);
            }
        }
        Inst::NewInstance { fields, .. } => {
            for x in fields.iter_mut() {
                *x = f(*x);
            }
        }
        Inst::IsInstance { src, .. } => *src = f(*src),
    }
}

// ----------------------------------------------- phis of immutable structs

/// A struct that is only ever *read* field by field, even when which struct
/// it is depends on control flow (`x = if c { V2 {…} } else { V2 {…} }`, a
/// loop-carried `x = hit.p`), needs no allocation: a field read off the phi
/// is the phi of the field reads, and a field read off a `make_instance` is
/// its operand. The class is the phi, its inputs, their inputs; it qualifies
/// when every use of every member is a `get_field` on it or membership in
/// another member phi -- no `set_field` (a write through one alias would
/// have to reach the others), nothing stored, passed or compared. Returns
/// whether anything changed.
// How many times `split_struct_phis` rebuilt its whole-body scans: the
// scale test reads it (tests/opt_tests.rs).
#[cfg(test)]
thread_local! {
    pub(crate) static SPLIT_PHI_SCANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn split_struct_phis(
    body: &mut Body,
    values: &HashSet<AdtId>,
    res: &Resolutions,
    entry_locals: &[DecId],
) -> bool {
    // candidate classes, seeded from phis whose inputs are all struct values.
    // For a value type an input may also be a *leaf*: a typed struct from
    // outside the class -- a param, a call result, an element -- whose
    // fields are read once, right where it is defined (immutable, so that
    // is as good as any later point), and which keeps existing for its
    // other uses.
    let leaf_adt = |v: InstId, body: &Body| -> Option<AdtId> {
        let id = adt_of_ty(&ty_of(res, entry_locals, body, v, 0)?)?;
        // a struct, one layout: an enum's value could be any variant, and
        // splitting it into its (zero) direct fields rebuilt a bare enum
        let one_layout = res.adts[id].dispatch_ids.as_slice() == [id];
        (values.contains(&id) && one_layout).then_some(id)
    };
    let mut done_any = false;
    let phis: Vec<InstId> = body
        .instructions
        .iter()
        .filter(|(_, i)| matches!(i, Inst::Phi(_)))
        .map(|(id, _)| id)
        .collect();
    // The use map, the in-stream set and the volatile reads are whole-body
    // scans. A class rewrite changes them, so they are rebuilt after one;
    // a root that does not qualify (most of them) changes nothing, and the
    // next root reuses them. Rebuilding them for every phi made this pass
    // O(phis × instructions): 10.6 s of a 10.8 s compile on wost-kit.
    // r[impl mimas.opt.linear]
    let mut scans: Option<(HashMap<InstId, Vec<InstId>>, HashSet<InstId>)> = None;
    let mut volatile_scan: Option<HashSet<InstId>> = None;
    let mut dirty = false;
    'phi: for &root in &phis {
        // a class rewrite changes the uses the next class sees: recount
        if dirty {
            scans = None;
            volatile_scan = None;
            dirty = false;
        }
        let (users, in_stream) = scans.get_or_insert_with(|| {
            #[cfg(test)]
            SPLIT_PHI_SCANS.with(|c| c.set(c.get() + 1));
            let in_stream: HashSet<InstId> = body
                .blocks
                .values()
                .flat_map(|b| b.stream.iter().copied())
                .collect();
            (users_of(body), in_stream)
        });
        let (users, in_stream) = (&*users, &*in_stream);
        if !in_stream.contains(&root) {
            continue;
        }
        // collect the class: a phi's inputs, and every phi a member feeds --
        // the struct built before a loop is an input of the loop-head phi and
        // of the loop-exit phi alike, and both must come along
        let mut class: Vec<InstId> = Vec::new();
        let mut leaves: Vec<InstId> = Vec::new();
        let mut work = vec![root];
        let mut nfields: Option<usize> = None;
        let mut class_adt: Option<AdtId> = None;
        while let Some(v) = work.pop() {
            if class.contains(&v) || leaves.contains(&v) {
                continue;
            }
            match &body.instructions[v] {
                Inst::NewInstance { adt, fields } => {
                    if nfields.is_some_and(|n| n != fields.len())
                        || class_adt.is_some_and(|a| a != *adt)
                    {
                        continue 'phi;
                    }
                    nfields = Some(fields.len());
                    class_adt = Some(*adt);
                }
                Inst::Phi(br) => {
                    if br.is_empty() {
                        continue 'phi;
                    }
                    for (_, x) in br {
                        work.push(*x);
                    }
                }
                _ => {
                    let Some(id) = leaf_adt(v, body) else {
                        continue 'phi;
                    };
                    if class_adt.is_some_and(|a| a != id) {
                        continue 'phi;
                    }
                    class_adt = Some(id);
                    let n = res.adts[id].member_tys.len();
                    if nfields.is_some_and(|m| m != n) {
                        continue 'phi;
                    }
                    nfields = Some(n);
                    leaves.push(v);
                    continue;
                }
            }
            for &u in users.get(&v).map(|x| x.as_slice()).unwrap_or(&[]) {
                if matches!(body.instructions[u], Inst::Phi(_)) && in_stream.contains(&u) {
                    work.push(u);
                }
            }
            class.push(v);
        }
        let Some(nf) = nfields else { continue };
        // a field that is a read of a slot written later cannot be forwarded
        let volatile = volatile_scan.get_or_insert_with(|| volatile_reads(body));
        if class.iter().any(|&m| matches!(&body.instructions[m], Inst::NewInstance { fields, .. } if fields.iter().any(|f| volatile.contains(f)))) {
            continue 'phi;
        }
        let adt = class_adt;
        let is_value = adt.is_some_and(|a| values.contains(&a));
        if !leaves.is_empty() && !is_value {
            continue 'phi;
        }
        if !class
            .iter()
            .any(|&m| matches!(body.instructions[m], Inst::Phi(_)))
        {
            continue 'phi;
        }
        // every use of every member: a field read on it, or feeding a member
        // phi. A value type may also escape -- be returned, passed, stored,
        // compared, join a phi outside the class -- and gets a fresh struct
        // built from the field values there: no one can tell the copy apart.
        let mut escapes: Vec<(InstId, InstId)> = Vec::new(); // (member, use)
        // the debugger's copies of members: split into one per field below
        let mut shadow_users: Vec<(InstId, InstId, Local)> = Vec::new(); // (member, shadow inst, its leaf)
        for &m in &class {
            for &u in users.get(&m).map(|v| v.as_slice()).unwrap_or(&[]) {
                match &body.instructions[u] {
                    Inst::GetField { src, slot, .. } if *src == m && (*slot as usize) < nf => {}
                    Inst::Phi(_) if class.contains(&u) => {}
                    Inst::Shadow { local, value } if *value == m => {
                        shadow_users.push((m, u, *local))
                    }
                    _ if is_value => escapes.push((m, u)),
                    _ => continue 'phi,
                }
            }
        }
        // one field-phi per (member phi, slot) and one field read per (leaf,
        // slot), built on demand
        let mut field_phi: HashMap<(InstId, u32), InstId> = HashMap::new();
        fn field_of(
            body: &mut Body,
            v: InstId,
            slot: u32,
            field_phi: &mut HashMap<(InstId, u32), InstId>,
            pos: &Positions,
        ) -> InstId {
            match body.instructions[v].clone() {
                Inst::NewInstance { fields, .. } => fields[slot as usize],
                Inst::Phi(br) => {
                    if let Some(&p) = field_phi.get(&(v, slot)) {
                        return p;
                    }
                    let blk = pos[&v].0;
                    let p = push_inst(body, Inst::Phi(Vec::new()), body.locs[v]);
                    field_phi.insert((v, slot), p);
                    // placed at the top of the phi's block, before any non-phi
                    body.blocks[blk].stream.insert(0, p);
                    let inputs: Vec<(BlockId, InstId)> = br
                        .iter()
                        .map(|(b, x)| (*b, field_of(body, *x, slot, field_phi, pos)))
                        .collect();
                    body.instructions[p] = Inst::Phi(inputs);
                    p
                }
                _ => {
                    // a leaf: read the field right after the leaf is defined
                    if let Some(&g) = field_phi.get(&(v, slot)) {
                        return g;
                    }
                    let blk = pos[&v].0;
                    let g = push_inst(
                        body,
                        Inst::GetField {
                            src: v,
                            slot,
                            kind: AccessKind::Direct,
                        },
                        body.locs[v],
                    );
                    field_phi.insert((v, slot), g);
                    let st = &mut body.blocks[blk].stream;
                    let at = st.iter().position(|&x| x == v).map_or(st.len(), |i| i + 1);
                    st.insert(at, g);
                    g
                }
            }
        }
        let pos = positions(body);
        // from here the class is rewritten: the next root rescans
        dirty = true;
        let mut rewrites: Vec<(InstId, InstId)> = Vec::new();
        for &m in &class {
            for &u in users.get(&m).map(|v| v.as_slice()).unwrap_or(&[]) {
                if let Inst::GetField { slot, .. } = body.instructions[u] {
                    let fv = field_of(body, m, slot, &mut field_phi, &pos);
                    rewrites.push((u, fv));
                }
            }
        }
        if rewrites.is_empty() && escapes.is_empty() && shadow_users.is_empty() {
            continue;
        }
        // shadows: the leaf becomes a struct of fresh leaves, and *every*
        // shadow into that leaf -- of a member, split into the field values;
        // of anything else (a leaf of the class, a struct from elsewhere),
        // into reads of its fields -- so the tree is the same at each store
        let mut split_leaves: HashMap<Local, Vec<Local>> = HashMap::new();
        for &(_, _, s) in &shadow_users {
            if split_leaves.contains_key(&s) {
                continue;
            }
            if let Some(leaves) = shadow_split_leaf(body, s, adt.unwrap(), nf) {
                split_leaves.insert(s, leaves);
            }
        }
        if !split_leaves.is_empty() {
            let all_shadows: Vec<(InstId, Local, InstId)> = body
                .blocks
                .values()
                .flat_map(|b| b.stream.iter().copied())
                .filter_map(|i| match body.instructions[i] {
                    Inst::Shadow { local, value } if split_leaves.contains_key(&local) => {
                        Some((i, local, value))
                    }
                    _ => None,
                })
                .collect();
            for (u, s, v) in all_shadows {
                let leaves = split_leaves[&s].clone();
                let (ublk, _) = pos[&u];
                let loc = body.locs[u];
                let mut insts: Vec<InstId> = Vec::new();
                for (k, &leaf) in leaves.iter().enumerate() {
                    let fv = if class.contains(&v) {
                        field_of(body, v, k as u32, &mut field_phi, &pos)
                    } else {
                        let g = push_inst(
                            body,
                            Inst::GetField {
                                src: v,
                                slot: k as u32,
                                kind: AccessKind::Direct,
                            },
                            loc,
                        );
                        insts.push(g);
                        g
                    };
                    insts.push(push_inst(
                        body,
                        Inst::Shadow {
                            local: leaf,
                            value: fv,
                        },
                        loc,
                    ));
                }
                let st = &mut body.blocks[ublk].stream;
                let at = st.iter().position(|&x| x == u).unwrap();
                st.splice(at..at + 1, insts);
            }
        }
        // escapes: a fresh struct from the field values, right before the use
        // (for a phi outside the class: at the end of the edge's block)
        for &(m, u) in &escapes {
            let adt = adt.unwrap();
            let fields: Vec<InstId> = (0..nf as u32)
                .map(|k| field_of(body, m, k, &mut field_phi, &pos))
                .collect();
            let fresh = push_inst(body, Inst::NewInstance { adt, fields }, body.locs[u]);
            let (ublk, _) = pos[&u];
            match &mut body.instructions[u] {
                Inst::Phi(br) => {
                    for (pb, v) in br.iter_mut() {
                        if *v == m {
                            let pb = *pb;
                            *v = fresh;
                            // before the predecessor's terminator
                            let st = &mut body.blocks[pb].stream;
                            let at = st.len().saturating_sub(1);
                            st.insert(at, fresh);
                        }
                    }
                }
                other => {
                    map_operands(other, &|x| if x == m { fresh } else { x });
                    let st = &mut body.blocks[ublk].stream;
                    let at = st.iter().position(|&x| x == u).unwrap();
                    st.insert(at, fresh);
                }
            }
        }
        let map: HashMap<InstId, InstId> = rewrites.iter().copied().collect();
        let resolve = |x: InstId| -> InstId {
            let mut x = x;
            while let Some(&n) = map.get(&x) {
                x = n;
            }
            x
        };
        for inst in body.instructions.values_mut() {
            map_operands(inst, &resolve);
        }
        for block in body.blocks.values_mut() {
            block.stream.retain(|iid| !map.contains_key(iid));
        }
        done_any = true;
    }
    done_any
}

/// Reads of slots that are written somewhere in the body. Codegen aliases a
/// `get l` to the slot's register rather than copying it, so such a read is
/// *not* a snapshot: forwarding another value's reads to it would see the
/// slot's later writes. Every forwarder below refuses to forward to one.
fn volatile_reads(body: &Body) -> HashSet<InstId> {
    let mut written: HashSet<Local> = HashSet::new();
    for block in body.blocks.values() {
        for &iid in &block.stream {
            if let Inst::SetLocal(l, _) = body.instructions[iid] {
                written.insert(l);
            }
        }
    }
    let mut out = HashSet::new();
    for block in body.blocks.values() {
        for &iid in &block.stream {
            if let Inst::GetLocal(l) = body.instructions[iid]
                && written.contains(&l)
            {
                out.insert(iid);
            }
        }
    }
    out
}

// ----------------------------------------------------------- value types

/// The static type of an instruction's value, as far as the IR can say:
/// construction, a typed local or param, a field of a typed struct, an
/// element of a typed array, an unwrapped option, a phi of agreeing inputs.
/// `None` is "don't know", and the callers fail closed on it.
fn ty_of(
    res: &Resolutions,
    entry_locals: &[DecId],
    body: &Body,
    iid: InstId,
    depth: u32,
) -> Option<Ty> {
    if depth > 8 {
        return None;
    }
    let dec_ty = |dec: Option<&DecId>| -> Option<Ty> {
        let dec = *dec?;
        if dec == DecId::DANGLING {
            return None;
        }
        Some(res.decs[dec].ty.clone())
    };
    match &body.instructions[iid] {
        Inst::NewInstance { adt, .. } => Some(Ty::Adt(*adt, Vec::new())),
        Inst::GetLocal(l) => dec_ty(body.locals.get(*l)),
        Inst::GetEntry(l) => dec_ty(entry_locals.get(l.index())),
        Inst::GetField { src, slot, .. } => {
            match ty_of(res, entry_locals, body, *src, depth + 1)? {
                Ty::Adt(id, _) | Ty::Identity(id, _) => {
                    res.adts[id].member_tys.get(*slot as usize).cloned()
                }
                _ => None,
            }
        }
        Inst::GetIndex { set, .. } => match ty_of(res, entry_locals, body, *set, depth + 1)? {
            Ty::Array(elem, _) => Some(*elem),
            _ => None,
        },
        Inst::Unwrap(v) => match ty_of(res, entry_locals, body, *v, depth + 1)? {
            Ty::Option(inner) => Some(*inner),
            other => Some(other),
        },
        Inst::Phi(br) => {
            let mut out: Option<Ty> = None;
            for (_, v) in br {
                let t = ty_of(res, entry_locals, body, *v, depth + 1)?;
                match &out {
                    None => out = Some(t),
                    Some(o) if *o == t => {}
                    _ => return None,
                }
            }
            out
        }
        Inst::Constant(c) => Some(match c {
            crate::Constant::Int(_) => Ty::Int,
            crate::Constant::Float(_) => Ty::Float,
            crate::Constant::Bool(_) => Ty::Bool,
            crate::Constant::Str(_) => Ty::Str,
            crate::Constant::Null => Ty::Null,
            crate::Constant::Array(_) | crate::Constant::Dec(_) => return None,
        }),
        Inst::BinOp { .. }
        | Inst::UnaryOp { .. }
        | Inst::ToFloat(_)
        | Inst::Sqrt(_)
        | Inst::Len(_)
        | Inst::In(..)
        | Inst::IsInstance { .. }
        | Inst::IsRaised(_)
        | Inst::Format(_) => Some(Ty::Any),
        Inst::NewArray | Inst::NewDict => Some(Ty::Any),
        _ => None,
    }
}

fn adt_of_ty(t: &Ty) -> Option<AdtId> {
    match t {
        Ty::Adt(id, _) | Ty::Identity(id, _) => Some(*id),
        _ => None,
    }
}

/// Whether a value of type `t` could be an instance: `Any`, a type variable
/// or an option of one could, scalars, arrays and strings cannot.
fn may_be_instance(t: &Ty) -> bool {
    match t {
        Ty::Adt(..) | Ty::Identity(..) | Ty::Any | Ty::Vid(_) => true,
        Ty::Option(inner) => may_be_instance(inner),
        _ => false,
    }
}

/// The adts no instruction in the program mutates: no `set_field` on one,
/// no `set_index` into one (a tuple struct's `t.0 = v`), no native that
/// mutates its receiver (`mutates_recv`). An instance of such an adt has
/// no identity anyone can observe -- `==` is structural -- so copying one
/// is always sound: that is what makes it a *value type*. Fails closed: a
/// receiver whose type the IR cannot recover disables value types for the
/// whole program (`MIMAS_DUMP_OPT=1` says where).
pub(crate) fn value_adts(ir: &Ir, mutating: &HashSet<NativeId>) -> HashSet<AdtId> {
    let entry_locals: Vec<DecId> = ir.bodies[BodyId::ZERO].locals.values().copied().collect();
    let mut mutable: HashSet<AdtId> = HashSet::new();
    let mut unknown: Option<String> = None;
    for (bid, body) in ir.bodies.iter() {
        for block in body.blocks.values() {
            for &iid in &block.stream {
                let receiver = match &body.instructions[iid] {
                    Inst::SetField { receiver, .. } => Some(*receiver),
                    Inst::SetIndex { set, .. } => Some(*set),
                    Inst::CallNative { id, args } if mutating.contains(id) => args.first().copied(),
                    _ => None,
                };
                let Some(r) = receiver else { continue };
                match ty_of(&ir.resolutions, &entry_locals, body, r, 0) {
                    Some(t) => {
                        if let Some(id) = adt_of_ty(&t) {
                            mutable.insert(id);
                        } else if may_be_instance(&t) {
                            unknown.get_or_insert_with(|| {
                                format!("body @{} {iid}: receiver typed {t:?}", bid.index())
                            });
                        }
                    }
                    None => {
                        unknown.get_or_insert_with(|| {
                            format!("body @{} {iid}: receiver of unknown type", bid.index())
                        });
                    }
                }
            }
        }
    }
    if let Some(why) = unknown {
        if std::env::var("MIMAS_DUMP_OPT").is_ok_and(|v| v == "1") {
            eprintln!("value types off: {why}");
        }
        return HashSet::new();
    }
    ir.resolutions
        .adts
        .iter()
        .map(|(id, _)| id)
        .filter(|id| !mutable.contains(id))
        .collect()
}

/// A field read off a freshly built value-type struct is its operand,
/// whatever else the struct is used for: `let d = a.sub(b); keep(d); d.x`
/// reads the operand, and the object lives on only for `keep`.
fn forward_value_fields(body: &mut Body, values: &HashSet<AdtId>) -> bool {
    let volatile = volatile_reads(body);
    let mut map: HashMap<InstId, InstId> = HashMap::new();
    for block in body.blocks.values() {
        for &iid in &block.stream {
            if let Inst::GetField { src, slot, .. } = body.instructions[iid]
                && let Inst::NewInstance { adt, fields } = &body.instructions[src]
                && values.contains(adt)
                && let Some(&f) = fields.get(slot as usize)
                && !volatile.contains(&f)
            {
                map.insert(iid, f);
            }
        }
    }
    if map.is_empty() {
        return false;
    }
    let resolve = |x: InstId| -> InstId {
        let mut x = x;
        while let Some(&n) = map.get(&x) {
            x = n;
        }
        x
    };
    for inst in body.instructions.values_mut() {
        map_operands(inst, &resolve);
    }
    for block in body.blocks.values_mut() {
        block.stream.retain(|iid| !map.contains_key(iid));
    }
    true
}

// ------------------------------------------------------------ tail calls

/// A call of the body to itself whose value is returned straight away --
/// `fn walk(x, k) -> … { if done { x } else { walk(next(x), k + 1) } }` --
/// becomes a jump: the arguments are assigned to the params as one
/// parallel assignment (an argument that *is* a param reads the slot, so
/// the order matters and a cycle goes through a scratch local), a
/// `trace_edge` keeps the self edge in the call graph, and control jumps
/// to a header block that took over the entry block's instructions, so
/// the entry block itself stays phi-free for SSA. Tail position means the
/// call is the last thing in its block and that block either returns the
/// call or jumps to a block holding only phis and a return of a phi the
/// call feeds. Returns whether anything changed.
fn self_tail_calls(ir: &mut Ir) -> bool {
    let mut changed = false;
    for bi in 1..ir.bodies.len() {
        let bid = BodyId::from(bi as u32);
        if !ir.bodies[bid].captures.is_empty() {
            continue;
        }
        let mut header: Option<BlockId> = None;
        loop {
            let body = &ir.bodies[bid];
            let users = users_of(body);
            // find a candidate: (call inst, its block, the phi entry (block, phi) it feeds if any).
            // The lowering puts forwarder blocks (a lone `jump`) between the
            // call's block and the phi block, and the phi block jumps on to
            // the block that returns: follow both chains.
            let follow = |mut b: BlockId, body: &Body| -> BlockId {
                for _ in 0..body.blocks.len() {
                    let st = &body.blocks[b].stream;
                    match (st.len(), st.first().map(|&i| &body.instructions[i])) {
                        (1, Some(Inst::Jump { target })) => b = *target,
                        _ => break,
                    }
                }
                b
            };
            // the block that jumps straight into `t` along the chain from `from`
            let last_before = |from: BlockId, t: BlockId, body: &Body| -> BlockId {
                let mut b = from;
                for _ in 0..body.blocks.len() {
                    let st = &body.blocks[b].stream;
                    match st.last().map(|&i| &body.instructions[i]) {
                        Some(Inst::Jump { target }) if *target == t => return b,
                        Some(Inst::Jump { target }) => b = *target,
                        _ => break,
                    }
                }
                b
            };
            let mut found: Option<(InstId, BlockId, Option<(BlockId, InstId)>)> = None;
            'scan: for (blk, block) in body.blocks.iter() {
                let n = block.stream.len();
                for (i, &iid) in block.stream.iter().enumerate() {
                    let Inst::CallDirect { body: callee, .. } = &body.instructions[iid] else {
                        continue;
                    };
                    if *callee != bid {
                        continue;
                    }
                    let us = users.get(&iid).map(|v| v.as_slice()).unwrap_or(&[]);
                    if us.len() != 1 || i + 2 != n {
                        continue;
                    }
                    let u = us[0];
                    let tail = block.stream[n - 1];
                    match &body.instructions[tail] {
                        Inst::Return(r) if *r == iid && tail == u => {
                            found = Some((iid, blk, None));
                            break 'scan;
                        }
                        Inst::Jump { target } => {
                            let t = follow(*target, body);
                            let ts = &body.blocks[t].stream;
                            // `return call` after a chain of jumps
                            if ts.len() == 1
                                && matches!(body.instructions[ts[0]], Inst::Return(r) if r == iid)
                            {
                                found = Some((iid, blk, None));
                                break 'scan;
                            }
                            // phis, then on to a block that returns the phi the call feeds
                            if !matches!(body.instructions[u], Inst::Phi(_)) || !ts.contains(&u) {
                                continue;
                            }
                            let only_phis = ts
                                .iter()
                                .take(ts.len().saturating_sub(1))
                                .all(|&x| matches!(body.instructions[x], Inst::Phi(_)));
                            let Some(&tl) = ts.last() else { continue };
                            let ret_block = match &body.instructions[tl] {
                                Inst::Jump { target } => follow(*target, body),
                                _ => t,
                            };
                            let rs = &body.blocks[ret_block].stream;
                            let returns_phi = if ret_block == t {
                                matches!(body.instructions[tl], Inst::Return(v) if v == u)
                            } else {
                                rs.len() == 1
                                    && matches!(body.instructions[rs[0]], Inst::Return(v) if v == u)
                            };
                            let pus = users.get(&u).map(|v| v.as_slice()).unwrap_or(&[]);
                            if only_phis && returns_phi && pus.len() == 1 {
                                let pred = last_before(blk, t, body);
                                found = Some((iid, blk, Some((pred, u))));
                                break 'scan;
                            }
                        }
                        _ => {}
                    }
                }
            }
            let Some((call, blk, phi)) = found else { break };
            let body = &mut ir.bodies[bid];
            let Inst::CallDirect { args, .. } = body.instructions[call].clone() else {
                unreachable!()
            };
            let loc = body.locs[call];
            // the header: the entry block's instructions move into it once
            let h = *header.get_or_insert_with(|| {
                let h = body.blocks.push(Block::new());
                let moved = std::mem::take(&mut body.blocks[BlockId::ZERO].stream);
                body.blocks[h].stream = moved;
                let j = push_inst(body, Inst::Jump { target: h }, loc);
                body.blocks[BlockId::ZERO].stream = vec![j];
                // phis that named the entry block as predecessor now mean the header
                for inst in body.instructions.values_mut() {
                    if let Inst::Phi(br) = inst {
                        for (p, _) in br.iter_mut() {
                            if *p == BlockId::ZERO {
                                *p = h;
                            }
                        }
                    }
                }
                h
            });
            // the parallel assignment params <- args: an arg that reads a param
            // slot (`get l_param`) must be consumed before that slot is written
            let params = body.params.clone();
            let reads_param = |v: InstId, body: &Body| -> Option<Local> {
                match body.instructions[v] {
                    Inst::GetLocal(l) if params.contains(&l) => Some(l),
                    _ => None,
                }
            };
            let mut pending: Vec<(Local, InstId)> =
                params.iter().copied().zip(args.iter().copied()).collect();
            let mut tail: Vec<InstId> = vec![push_inst(body, Inst::TraceEdge { callee: bid }, loc)];
            while !pending.is_empty() {
                // an assignment whose target no other pending arg still reads
                let pick = pending.iter().position(|&(dst, _)| {
                    !pending
                        .iter()
                        .any(|&(d2, a2)| d2 != dst && reads_param(a2, body) == Some(dst))
                });
                match pick {
                    Some(i) => {
                        let (dst, arg) = pending.remove(i);
                        if reads_param(arg, body) == Some(dst) {
                            continue; // `f(a, …)` passing `a` through: nothing to do
                        }
                        tail.push(push_inst(body, Inst::SetLocal(dst, arg), loc));
                    }
                    None => {
                        // a cycle: park one param's current value in a scratch local
                        let (_, arg) = pending[0];
                        let src = reads_param(arg, body).unwrap();
                        let tmp = body.locals.push(DecId::DANGLING);
                        let read = push_inst(body, Inst::GetLocal(src), loc);
                        let park = push_inst(body, Inst::SetLocal(tmp, read), loc);
                        tail.push(read);
                        tail.push(park);
                        let tmp_read = push_inst(body, Inst::GetLocal(tmp), loc);
                        tail.push(tmp_read);
                        for p in pending.iter_mut() {
                            if reads_param(p.1, body) == Some(src) {
                                p.1 = tmp_read;
                            }
                        }
                    }
                }
            }
            tail.push(push_inst(body, Inst::Jump { target: h }, loc));
            // splice: drop the call and what followed it, append the tail
            let stream = &mut body.blocks[blk].stream;
            let at = stream.iter().position(|&x| x == call).unwrap();
            stream.truncate(at);
            stream.extend(tail);
            if let Some((pred, p)) = phi
                && let Inst::Phi(br) = &mut body.instructions[p]
            {
                br.retain(|(b, v)| !(*b == pred && *v == call));
            }
            changed = true;
        }
    }
    changed
}

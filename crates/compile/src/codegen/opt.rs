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
use shared::Location;
use solve::components::DecId;

use crate::{
    BlockId, Inst, InstId, Ir, Local,
    ir::{Block, Body, BodyId, FormatPart},
};

use super::clean::uses;

/// Callee bodies at most this many instructions long are inlined.
const MAX_INLINE_INSTS: usize = 64;
/// A caller stops inlining once it has grown past this many instructions.
const MAX_CALLER_INSTS: usize = 4096;

pub(crate) fn enabled() -> bool {
    std::env::var("MIMAS_OPT").map_or(true, |v| v != "0")
}

/// `frame_natives` are natives that read the live call stack
/// (`std::debug::caller_line` and kin): a body that calls one keeps its
/// frame, so what they report stays true.
pub(crate) fn optimize(ir: &mut Ir, frame_natives: &HashSet<NativeId>) {
    inline_small_calls(ir, frame_natives);
    for (bid, body) in ir.bodies.iter_mut() {
        if bid == BodyId::ZERO {
            continue;
        }
        // a round can expose the next: a field that held a struct becomes a
        // local holding a struct
        for _ in 0..4 {
            if !scalar_replace(body) {
                break;
            }
        }
        // locals to SSA, then structs chosen by control flow to field phis;
        // a split can expose a local whose only writes are now structs.
        // `MIMAS_SSA=0` keeps the slot form (an A/B for the lanes' cost model)
        if std::env::var("MIMAS_SSA").is_ok_and(|v| v == "0") {
            continue;
        }
        for _ in 0..3 {
            let a = promote_locals(body);
            let b = split_struct_phis(body);
            if !a && !b {
                break;
            }
        }
    }
}

// ------------------------------------------------------------ inlining

/// Can `bid` be copied into its callers: a small, capture-free body that
/// never reaches itself through direct calls, never `raise`s (`Op::Raise`
/// returns from the *frame*, which a copy no longer has), and calls no
/// native that reads the frame stack.
fn inlinable(ir: &Ir, bid: BodyId, calls: &HashMap<BodyId, Vec<BodyId>>, frame_natives: &HashSet<NativeId>) -> bool {
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
                Inst::CallNative { id, .. } if frame_natives.contains(id) => return false,
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
            let size: usize = ir.bodies[caller].blocks.values().map(|b| b.stream.len()).sum();
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
fn find_call(body: &Body, small: &HashSet<BodyId>, me: BodyId) -> Option<(BlockId, InstId, BodyId)> {
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
    let local_map: Vec<Local> = snap.locals.iter().map(|&dec| body.locals.push(dec)).collect();
    let ret = body.locals.push(DecId::DANGLING);
    let block_map: Vec<BlockId> = snap.blocks.iter().map(|_| body.blocks.push(Block::new())).collect();
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
                let j = push_inst(body, Inst::Jump { target: cont }, snap.locs[old.index()].clone());
                out.push(j);
            }
        }
        body.blocks[block_map[bi]].stream = out;
    }
    // split the call's block: params bound, jump in; the rest moves to `cont`
    let stream = std::mem::take(&mut body.blocks[block].stream);
    let at = stream.iter().position(|&x| x == iid).expect("call in its block");
    let (pre, post) = stream.split_at(at);
    let mut head = pre.to_vec();
    // the call's footprint for the trace's call graph
    let te = push_inst(body, Inst::TraceEdge { callee }, call_loc);
    head.push(te);
    for (p, arg) in snap.params.iter().zip(&args) {
        let s = push_inst(body, Inst::SetLocal(local_map[p.index()], *arg), call_loc.clone());
        head.push(s);
    }
    let j = push_inst(body, Inst::Jump { target: block_map[0] }, call_loc.clone());
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
        Inst::GetLocal(x) => Inst::GetLocal(l(x)),
        Inst::SetEntry(x, v) => Inst::SetEntry(*x, i(v)),
        Inst::GetEntry(x) => Inst::GetEntry(*x),
        Inst::BinOp { left, op, right, kind } => Inst::BinOp { left: i(left), op: *op, right: i(right), kind: *kind },
        Inst::UnaryOp { op, right } => Inst::UnaryOp { op: *op, right: i(right) },
        Inst::JumpIfFalse { condition, target } => Inst::JumpIfFalse { condition: i(condition), target: b(target) },
        Inst::Jump { target } => Inst::Jump { target: b(target) },
        Inst::Switch { scrut, base, table, default } => Inst::Switch {
            scrut: i(scrut),
            base: *base,
            table: table.iter().map(b).collect(),
            default: b(default),
        },
        Inst::Phi(branches) => Inst::Phi(branches.iter().map(|(bb, v)| (b(bb), i(v))).collect()),
        Inst::ForNext { idx, bound, target } => Inst::ForNext { idx: i(idx), bound: i(bound), target: b(target) },
        Inst::NewArray => Inst::NewArray,
        Inst::NewDict => Inst::NewDict,
        Inst::SetIndex { set, index, value } => Inst::SetIndex { set: i(set), index: i(index), value: i(value) },
        Inst::GetIndex { set, index, kind } => Inst::GetIndex { set: i(set), index: i(index), kind: *kind },
        Inst::GetField { src, slot, kind } => Inst::GetField { src: i(src), slot: *slot, kind: *kind },
        Inst::SetField { receiver, slot, value } => Inst::SetField { receiver: i(receiver), slot: *slot, value: i(value) },
        Inst::Push { array, value } => Inst::Push { array: i(array), value: i(value) },
        Inst::Insert { dict, key, value } => Inst::Insert { dict: i(dict), key: *key, value: i(value) },
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
        Inst::MakeClosure { body, captures } => Inst::MakeClosure { body: *body, captures: captures.iter().map(i).collect() },
        Inst::Call { callee, args } => Inst::Call { callee: i(callee), args: args.iter().map(i).collect() },
        Inst::CallDirect { body, args } => Inst::CallDirect { body: *body, args: args.iter().map(i).collect() },
        Inst::CallNative { id, args } => Inst::CallNative { id: *id, args: args.iter().map(i).collect() },
        Inst::Return(v) => Inst::Return(i(v)),
        Inst::NewInstance { adt, fields } => Inst::NewInstance { adt: *adt, fields: fields.iter().map(i).collect() },
        Inst::IsInstance { src, adt } => Inst::IsInstance { src: i(src), adt: *adt },
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
                    Inst::Jump { target } | Inst::JumpIfFalse { target, .. } | Inst::ForNext { target, .. } => out.push(*target),
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

/// Immediate-dominator-free dominance: `dom[b]` is the set of blocks on every
/// path from the entry to `b` (itself included). Unreachable blocks get an
/// empty set, so nothing in them counts as dominated. Bodies are small; the
/// plain iterative solve is fine.
fn dominators(body: &Body) -> Vec<HashSet<BlockId>> {
    let n = body.blocks.len();
    let succ = successors(body);
    let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); n];
    for (i, ss) in succ.iter().enumerate() {
        for &t in ss {
            preds[t.index()].push(BlockId::from(i as u32));
        }
    }
    let all: HashSet<BlockId> = (0..n).map(|i| BlockId::from(i as u32)).collect();
    let mut reach = HashSet::new();
    let mut work = vec![BlockId::ZERO];
    while let Some(b) = work.pop() {
        if reach.insert(b) {
            work.extend(succ[b.index()].iter().copied());
        }
    }
    let mut dom: Vec<HashSet<BlockId>> = (0..n)
        .map(|i| {
            let b = BlockId::from(i as u32);
            if b == BlockId::ZERO {
                HashSet::from([b])
            } else if reach.contains(&b) {
                all.clone()
            } else {
                HashSet::new()
            }
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for i in 1..n {
            let b = BlockId::from(i as u32);
            if !reach.contains(&b) {
                continue;
            }
            let mut next: Option<HashSet<BlockId>> = None;
            for &p in &preds[i] {
                if !reach.contains(&p) {
                    continue;
                }
                next = Some(match next {
                    None => dom[p.index()].clone(),
                    Some(acc) => acc.intersection(&dom[p.index()]).copied().collect(),
                });
            }
            let mut next = next.unwrap_or_default();
            next.insert(b);
            if next != dom[i] {
                dom[i] = next;
                changed = true;
            }
        }
    }
    dom
}

/// Where each instruction sits: its block and position in the stream.
fn positions(body: &Body) -> HashMap<InstId, (BlockId, usize)> {
    let mut out = HashMap::new();
    for (bid, block) in body.blocks.iter() {
        for (i, &iid) in block.stream.iter().enumerate() {
            out.insert(iid, (bid, i));
        }
    }
    out
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
            (Some(&(bw, pw)), Some(&(br, pr))) => (bw == br && pw < pr) || (bw != br && dom[br.index()].contains(&bw)),
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
        if reads.get(&l).map_or(true, |rs| rs.iter().all(|&r| dominated(w, r))) {
            single.insert(l, w);
        }
    }
    let single_use = |v: InstId| users.get(&v).map_or(0, |u| u.len()) == 1;
    // roots and copies
    let mut root_of: HashMap<Local, Local> = HashMap::new(); // member -> root
    let mut root_adt: HashMap<Local, (usize, InstId)> = HashMap::new(); // root -> (nfields, the make_instance)
    for (&l, &w) in &single {
        let Inst::SetLocal(_, v) = body.instructions[w] else { unreachable!() };
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
            let Inst::SetLocal(_, v) = body.instructions[w] else { unreachable!() };
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
                        Inst::GetField { src, slot, .. } if *src == g && (*slot as usize) < nfields => {}
                        Inst::SetField { receiver, slot, value } if *receiver == g && *value != g && (*slot as usize) < nfields => {}
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
        let fields: Vec<Local> = (0..nfields).map(|_| body.locals.push(DecId::DANGLING)).collect();
        let Inst::NewInstance { fields: vals, .. } = body.instructions[ni].clone() else { unreachable!() };
        let root_write = single[&r];
        let copy_writes: HashSet<InstId> = ms.iter().filter(|&&m| m != r).map(|m| single[m]).collect();
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
                        Inst::SetField { slot, value, .. } => Inst::SetLocal(fields[*slot as usize], *value),
                        // the copy was dropped from its stream; leave a harmless
                        // dead read in its place
                        Inst::SetLocal(..) => Inst::GetLocal(fields[0]),
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
    preds: Vec<Vec<BlockId>>,
    dom: Vec<HashSet<BlockId>>,
    idom: Vec<Option<BlockId>>,
    children: Vec<Vec<BlockId>>,
    frontier: Vec<HashSet<BlockId>>,
}

fn cfg_of(body: &Body) -> Cfg {
    let succ = successors(body);
    let preds = predecessors(&succ);
    let dom = dominators(body);
    let n = body.blocks.len();
    let mut idom: Vec<Option<BlockId>> = vec![None; n];
    for i in 1..n {
        if dom[i].is_empty() {
            continue;
        }
        // the immediate dominator is the strict dominator dominated by all others
        let b = BlockId::from(i as u32);
        idom[i] = dom[i]
            .iter()
            .filter(|&&d| d != b)
            .max_by_key(|d| dom[d.index()].len())
            .copied();
    }
    let mut children: Vec<Vec<BlockId>> = vec![Vec::new(); n];
    for (i, d) in idom.iter().enumerate() {
        if let Some(d) = d {
            children[d.index()].push(BlockId::from(i as u32));
        }
    }
    let mut frontier: Vec<HashSet<BlockId>> = vec![HashSet::new(); n];
    for i in 0..n {
        let b = BlockId::from(i as u32);
        if preds[i].len() < 2 || dom[i].is_empty() {
            continue;
        }
        for &p in &preds[i] {
            if dom[p.index()].is_empty() {
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
    Cfg { succ, preds, dom, idom, children, frontier }
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
fn promote_locals(body: &mut Body) -> bool {
    split_critical_edges(body);
    let cfg = cfg_of(body);
    let pos = positions(body);
    let dominated = |w: InstId, r: InstId| -> bool {
        match (pos.get(&w), pos.get(&r)) {
            (Some(&(bw, pw)), Some(&(br, pr))) => (bw == br && pw < pr) || (bw != br && cfg.dom[br.index()].contains(&bw)),
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
                let Inst::SetLocal(_, v) = body.instructions[w] else { return false };
                match &body.instructions[v] {
                    Inst::NewInstance { .. } => true,
                    Inst::GetLocal(m) => struct_local.contains(m),
                    Inst::Phi(br) => br.iter().all(|(_, x)| matches!(body.instructions[*x], Inst::NewInstance { .. })),
                    _ => false,
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
        let ok = reads
            .get(&l)
            .map_or(true, |rs| rs.iter().all(|&r| ws.iter().any(|&w| dominated(w, r))));
        if ok {
            promote.push(l);
        }
    }
    if promote.is_empty() {
        return false;
    }
    promote.sort_by_key(|l| l.index());
    let promote_set: HashSet<Local> = promote.iter().copied().collect();
    // phis at the iterated dominance frontier of each local's definition blocks
    let mut phi_of: HashMap<(BlockId, Local), InstId> = HashMap::new();
    for &l in &promote {
        let mut work: Vec<BlockId> = writes[&l].iter().map(|w| pos[w].0).collect();
        let mut placed: HashSet<BlockId> = HashSet::new();
        while let Some(b) = work.pop() {
            for &f in &cfg.frontier[b.index()] {
                if placed.insert(f) {
                    let loc = body.blocks[f].stream.first().map(|&i| body.locs[i]).unwrap_or_default();
                    let phi = push_inst(body, Inst::Phi(Vec::new()), loc);
                    body.blocks[f].stream.insert(0, phi);
                    phi_of.insert((f, l), phi);
                    work.push(f);
                }
            }
        }
    }
    // renaming, down the dominator tree
    let mut undef: Option<InstId> = None;
    let mut rewrite: HashMap<InstId, InstId> = HashMap::new(); // promoted GetLocal -> def
    let mut stores: Vec<(InstId, Local, InstId)> = Vec::new(); // promoted SetLocal, its local, resolved value
    let mut stack: HashMap<Local, Vec<InstId>> = HashMap::new();
    let resolve = |v: InstId, rewrite: &HashMap<InstId, InstId>| -> InstId {
        let mut v = v;
        while let Some(&n) = rewrite.get(&v) {
            v = n;
        }
        v
    };
    // explicit DFS: (block, number of defs pushed per local to pop on exit)
    let mut dfs: Vec<(BlockId, bool)> = vec![(BlockId::ZERO, false)];
    let mut pushed: Vec<Vec<Local>> = Vec::new();
    while let Some((b, leaving)) = dfs.pop() {
        if leaving {
            for l in pushed.pop().unwrap() {
                stack.get_mut(&l).unwrap().pop();
            }
            continue;
        }
        let mut mine: Vec<Local> = Vec::new();
        for &l in &promote {
            if let Some(&phi) = phi_of.get(&(b, l)) {
                stack.entry(l).or_default().push(phi);
                mine.push(l);
            }
        }
        for &iid in &body.blocks[b].stream.clone() {
            match body.instructions[iid].clone() {
                Inst::GetLocal(l) if promote_set.contains(&l) => {
                    let def = match stack.get(&l).and_then(|s| s.last()) {
                        Some(&d) => d,
                        None => *undef.get_or_insert_with(|| {
                            let u = push_inst(body, Inst::Constant(crate::Constant::Null), Location::default());
                            body.blocks[BlockId::ZERO].stream.insert(0, u);
                            u
                        }),
                    };
                    rewrite.insert(iid, def);
                }
                Inst::SetLocal(l, v) if promote_set.contains(&l) => {
                    let v = resolve(v, &rewrite);
                    stack.entry(l).or_default().push(v);
                    mine.push(l);
                    stores.push((iid, l, v));
                }
                _ => {}
            }
        }
        for &s in &cfg.succ[b.index()] {
            for &l in &promote {
                if let Some(&phi) = phi_of.get(&(s, l)) {
                    let def = match stack.get(&l).and_then(|st| st.last()) {
                        Some(&d) => d,
                        None => *undef.get_or_insert_with(|| {
                            let u = push_inst(body, Inst::Constant(crate::Constant::Null), Location::default());
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
    let drop_stores: HashSet<InstId> = stores.iter().map(|(iid, _, _)| *iid).collect();
    let dropped_reads: HashSet<InstId> = rewrite.keys().copied().collect();
    for inst in body.instructions.values_mut() {
        map_operands(inst, &|x| resolve(x, &rewrite));
    }
    for block in body.blocks.values_mut() {
        block.stream.retain(|iid| !dropped_reads.contains(iid) && !drop_stores.contains(iid));
    }
    true
}

/// `inst` with every instruction operand mapped (locals and blocks untouched).
fn map_operands(inst: &mut Inst, f: &dyn Fn(InstId) -> InstId) {
    match inst {
        Inst::Constant(_) | Inst::GetLocal(_) | Inst::GetEntry(_) | Inst::Jump { .. } | Inst::NewArray | Inst::NewDict
        | Inst::RefBody(_) | Inst::Panic | Inst::TraceEdge { .. } => {}
        Inst::SetLocal(_, v) | Inst::SetEntry(_, v) => *v = f(*v),
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
        Inst::SetField { receiver, value, .. } => {
            *receiver = f(*receiver);
            *value = f(*value);
        }
        Inst::Push { array, value } => {
            *array = f(*array);
            *value = f(*value);
        }
        Inst::Insert { value, .. } => *value = f(*value),
        Inst::Len(v) | Inst::ToFloat(v) | Inst::Sqrt(v) | Inst::Unwrap(v) | Inst::UnwrapUnit(v) | Inst::Return(v)
        | Inst::Raise(v) | Inst::IsRaised(v) | Inst::UnwrapRaised(v) => *v = f(*v),
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
fn split_struct_phis(body: &mut Body) -> bool {
    // candidate classes, seeded from phis whose inputs are all struct values
    let is_struct = |v: InstId, body: &Body| matches!(body.instructions[v], Inst::NewInstance { .. } | Inst::Phi(_));
    let mut done_any = false;
    let phis: Vec<InstId> = body
        .instructions
        .iter()
        .filter(|(_, i)| matches!(i, Inst::Phi(_)))
        .map(|(id, _)| id)
        .collect();
    'phi: for &root in &phis {
        // a class rewrite changes the uses the next class sees: recount
        let users = users_of(body);
        let in_stream: HashSet<InstId> = body.blocks.values().flat_map(|b| b.stream.iter().copied()).collect();
        if !in_stream.contains(&root) {
            continue;
        }
        // collect the class: a phi's inputs, and every phi a member feeds --
        // the struct built before a loop is an input of the loop-head phi and
        // of the loop-exit phi alike, and both must come along
        let mut class: Vec<InstId> = Vec::new();
        let mut work = vec![root];
        let mut nfields: Option<usize> = None;
        while let Some(v) = work.pop() {
            if class.contains(&v) {
                continue;
            }
            match &body.instructions[v] {
                Inst::NewInstance { fields, .. } => {
                    if nfields.is_some_and(|n| n != fields.len()) {
                        continue 'phi;
                    }
                    nfields = Some(fields.len());
                }
                Inst::Phi(br) => {
                    if br.is_empty() {
                        continue 'phi;
                    }
                    for (_, x) in br {
                        if !is_struct(*x, body) {
                            continue 'phi;
                        }
                        work.push(*x);
                    }
                }
                _ => continue 'phi,
            }
            for &u in users.get(&v).map(|x| x.as_slice()).unwrap_or(&[]) {
                if matches!(body.instructions[u], Inst::Phi(_)) && in_stream.contains(&u) {
                    work.push(u);
                }
            }
            class.push(v);
        }
        let Some(nf) = nfields else { continue };
        // every use of every member: a field read on it, or feeding a member phi
        for &m in &class {
            for &u in users.get(&m).map(|v| v.as_slice()).unwrap_or(&[]) {
                match &body.instructions[u] {
                    Inst::GetField { src, slot, .. } if *src == m && (*slot as usize) < nf => {}
                    Inst::Phi(_) if class.contains(&u) => {}
                    _ => continue 'phi,
                }
            }
        }
        // one field-phi per (member phi, slot), built on demand
        let mut field_phi: HashMap<(InstId, u32), InstId> = HashMap::new();
        fn field_of(
            body: &mut Body,
            v: InstId,
            slot: u32,
            field_phi: &mut HashMap<(InstId, u32), InstId>,
            pos: &HashMap<InstId, (BlockId, usize)>,
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
                _ => unreachable!("class members are structs"),
            }
        }
        let pos = positions(body);
        let mut rewrites: Vec<(InstId, InstId)> = Vec::new();
        for &m in &class {
            for &u in users.get(&m).map(|v| v.as_slice()).unwrap_or(&[]) {
                if let Inst::GetField { slot, .. } = body.instructions[u] {
                    let fv = field_of(body, m, slot, &mut field_phi, &pos);
                    rewrites.push((u, fv));
                }
            }
        }
        if rewrites.is_empty() {
            continue;
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

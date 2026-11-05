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
            inline_at(&mut ir.bodies[caller], block, iid, &snap);
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
fn inline_at(body: &mut Body, block: BlockId, iid: InstId, snap: &Snap) {
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

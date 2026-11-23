//! A body's compiled form, kept across compiles in one process: the edit
//! loop recompiles the whole spliced program on every keystroke, but an edit
//! changes a few bodies. Spliced includes come ahead of the page's own code
//! (`include.order`), so the ids of everything an include defines (decs,
//! adts, bodies, interned strings) do not move when the page below changes,
//! and the includes' bodies come out of the whole-program passes the same.
//!
//! The key of a body is everything its compile reads, taken after the
//! whole-program half of the optimizer (`opt::optimize_global`), so the
//! bodies inlined into it are part of it:
//!
//! - the body itself: blocks, instructions, locals, locations, the solved
//!   kind of each instruction, its artifact names and debugger shadows;
//! - the decs of its locals, and of the entry locals it reads (`GetEntry`):
//!   their solved types and whether they are globals (`ty_of`,
//!   `local_kinds`, `named_locals`);
//! - every adt reachable from those types, from `NewInstance`/`IsInstance`
//!   and from the shadows, closed over member types: member types, dispatch
//!   ids, type params, and whether it is a value type (`forward_value_fields`,
//!   `split_struct_phis`);
//! - the knobs that change the passes (`MIMAS_OPT`, `MIMAS_SSA`, shadows).
//!
//! The value is the body's ops, with jump targets relative to the body's
//! first byte, and its `Chunk` (registers, params, locations: all already
//! relative). A hit skips `optimize_body`, `clean` and register allocation
//! for the body; the ops are relocated to where the body lands in this
//! program's byte stream.
//!
//! `MIMAS_BODYCACHE=0` turns it off. `MIMAS_BODYCACHE=check` compiles every
//! hit anyway and panics when the result differs from the cached one: run a
//! corpus under it after changing what a body's compile reads (a read the
//! key misses is a stale body, not a slow one). `MIMAS_BODYCACHE_MB` caps
//! the store (default 64); the least recently used bodies go first.
//! `MIMAS_COMPILE_TIMING=1` prints the hits per compile.
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Mutex;

use solve::Resolutions;
use solve::ResolvedDeclKind;
use solve::components::{AdtId, DecId, Ty};

use super::opt::Global;
use super::program::Chunk;
use crate::{BlockTarget, Inst, Op, ir::Body};

/// Bump when what a key covers changes shape.
const VERSION: u32 = 1;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    On,
    Check,
}

pub(crate) fn mode() -> Mode {
    match std::env::var("MIMAS_BODYCACHE").as_deref() {
        Ok("0") => Mode::Off,
        Ok("check") => Mode::Check,
        _ => Mode::On,
    }
}

/// A body's key: a SipHash of what its compile reads, plus its shape as a
/// cheap second check against a collision.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Key {
    hash: u64,
    insts: u32,
    blocks: u32,
    locals: u32,
}

/// Feeds `Debug` text to a hasher without building strings (and keeps it
/// when `MIMAS_BODYCACHE_DUMP` asks for the key text of each miss).
struct H(DefaultHasher, Option<String>);

impl std::fmt::Write for H {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write(s.as_bytes());
        if let Some(t) = &mut self.1 {
            t.push_str(s);
        }
        Ok(())
    }
}

impl H {
    fn sep(&mut self) {
        self.0.write_u8(0xff);
        if let Some(t) = &mut self.1 {
            t.push('\n');
        }
    }
}

/// Type parameters and type variables numbered by first appearance in the
/// key. The solver draws both from counters that run on across compiles in
/// one process, so a generic body's decs read `Param(ParamId(26163))` on one
/// compile and `Param(ParamId(45448))` on the next. What a body's compile
/// does with them depends only on which are equal (`ty_of`'s phi rule,
/// `adt_of_ty`), and one renaming covers the whole key, so renaming keeps
/// every equality the passes can see.
#[derive(Default)]
struct Canon {
    params: HashMap<shared::ParamId, u32>,
    vids: HashMap<shared::Vid, u32>,
}

impl Canon {
    fn param(&mut self, p: shared::ParamId) -> shared::ParamId {
        let n = self.params.len() as u32;
        shared::ParamId::from(*self.params.entry(p).or_insert(n))
    }
    fn ty(&mut self, t: &Ty) -> Ty {
        let mut tys = |ts: &[Ty], c: &mut Canon| ts.iter().map(|t| c.ty(t)).collect::<Vec<_>>();
        match t {
            Ty::Param(p) => Ty::Param(self.param(*p)),
            Ty::Vid(v) => {
                let n = self.vids.len() as u32;
                Ty::Vid(shared::Vid::from(*self.vids.entry(*v).or_insert(n)))
            }
            Ty::Array(e, l) => Ty::Array(Box::new(self.ty(e)), l.clone()),
            Ty::Dict(k, v) => Ty::Dict(Box::new(self.ty(k)), Box::new(self.ty(v))),
            Ty::Option(e) => Ty::Option(Box::new(self.ty(e))),
            Ty::Result(e) => Ty::Result(Box::new(self.ty(e))),
            Ty::Tuple(ts) => Ty::Tuple(tys(ts, self)),
            Ty::Adt(id, args) => Ty::Adt(*id, tys(args, self)),
            Ty::Identity(id, args) => Ty::Identity(*id, tys(args, self)),
            Ty::Fn(h) => {
                let mut h = h.clone();
                for p in &mut h.parameters {
                    p.ty = self.ty(&p.ty);
                }
                h.return_ty = Box::new(self.ty(&h.return_ty));
                Ty::Fn(h)
            }
            other => other.clone(),
        }
    }
}

/// The adts in `t`.
fn adts_of(t: &Ty, out: &mut Vec<AdtId>) {
    match t {
        Ty::Adt(id, args) | Ty::Identity(id, args) => {
            out.push(*id);
            for a in args {
                adts_of(a, out);
            }
        }
        Ty::Dict(k, v) => {
            adts_of(k, out);
            adts_of(v, out);
        }
        Ty::Array(e, _) | Ty::Option(e) | Ty::Result(e) => adts_of(e, out),
        Ty::Tuple(ts) => ts.iter().for_each(|t| adts_of(t, out)),
        Ty::Fn(h) => {
            for p in &h.parameters {
                adts_of(&p.ty, out);
            }
            adts_of(&h.return_ty, out);
        }
        _ => {}
    }
}

/// The key of `body`'s compiled form. `config` carries the knobs that change
/// the passes (see `compiler.rs`).
pub(crate) fn key(body: &Body, res: &Resolutions, g: &Global, config: u8) -> Key {
    key_text(body, res, g, config, false).0
}

/// The key, and with `text` the text it hashes (a miss's, for
/// `MIMAS_BODYCACHE_DUMP`: diff two compiles' to see what moved).
pub(crate) fn key_text(body: &Body, res: &Resolutions, g: &Global, config: u8, text: bool) -> (Key, String) {
    let mut h = H(DefaultHasher::new(), text.then(String::new));
    h.0.write_u32(VERSION);
    h.0.write_u8(config);

    // the body, hashed field by field (its `Debug` text only for a dump)
    let mut d2l: Vec<_> = body.dec_to_local.iter().map(|(d, l)| (d.index(), l.index())).collect();
    d2l.sort_unstable();
    let mut arts: Vec<_> = body.artifacts.iter().map(|(l, n)| (l.index(), n)).collect();
    arts.sort_unstable();
    let mut kinds: Vec<_> = body.kinds.iter().map(|(i, k)| (i.index(), *k)).collect();
    kinds.sort_unstable_by_key(|(i, _)| *i);
    let mut shadows: Vec<_> = body.shadows.iter().map(|(l, s)| (l.index(), s)).collect();
    shadows.sort_unstable_by_key(|(l, _)| *l);
    {
        let hs = &mut h.0;
        body.blocks.len().hash(hs);
        for block in body.blocks.values() {
            block.stream.hash(hs);
        }
        body.instructions.len().hash(hs);
        for inst in body.instructions.values() {
            inst.hash(hs);
        }
        for d in body.locals.values() {
            d.hash(hs);
        }
        body.params.hash(hs);
        body.captures.hash(hs);
        for loc in body.locs.values() {
            loc.hash(hs);
        }
        d2l.hash(hs);
        arts.hash(hs);
        kinds.hash(hs);
        shadows.hash(hs);
    }
    if let Some(t) = &mut h.1 {
        for (bid, block) in body.blocks.iter() {
            let _ = writeln!(t, "{}:{:?}", bid.index(), block.stream);
        }
        for (iid, inst) in body.instructions.iter() {
            let _ = writeln!(t, "{} {inst:?} @{:?}", iid.index(), body.locs.get(iid));
        }
        let _ = writeln!(
            t,
            "{:?}|{:?}|{:?}|{d2l:?}|{arts:?}|{kinds:?}|{shadows:?}",
            body.locals, body.params, body.captures
        );
    }
    h.sep();

    // the decs it reads
    let mut decs: Vec<DecId> = body.locals.values().copied().collect();
    let mut adts: Vec<AdtId> = Vec::new();
    for inst in body.instructions.values() {
        match inst {
            Inst::GetEntry(l) | Inst::SetEntry(l, _) => {
                if let Some(&d) = g.entry_locals.get(l.index()) {
                    decs.push(d);
                }
            }
            Inst::NewInstance { adt, .. } | Inst::IsInstance { adt, .. } => adts.push(*adt),
            _ => {}
        }
    }
    decs.sort_unstable_by_key(|d| d.index());
    decs.dedup();
    let mut canon = Canon::default();
    for d in decs {
        if d == DecId::DANGLING {
            continue;
        }
        let dec = &res.decs[d];
        let global = matches!(dec.kind, ResolvedDeclKind::Global);
        let _ = write!(h, "{}:{global}:{:?};", d.index(), canon.ty(&dec.ty));
        adts_of(&dec.ty, &mut adts);
    }
    h.sep();

    // the adts reachable from them
    fn shadow_adts(s: &crate::Shadow<crate::Local>, out: &mut Vec<AdtId>) {
        if let crate::Shadow::Struct { adt, fields } = s {
            out.push(*adt);
            fields.iter().for_each(|f| shadow_adts(f, out));
        }
    }
    for s in body.shadows.values() {
        shadow_adts(s, &mut adts);
    }
    let mut seen: HashSet<AdtId> = HashSet::new();
    let mut closure: Vec<AdtId> = Vec::new();
    while let Some(a) = adts.pop() {
        if !seen.insert(a) {
            continue;
        }
        closure.push(a);
        // a primitive's sentinel id (`impl Num for float`) has no adt
        let Some(adt) = res.adts.get(a) else { continue };
        for t in &adt.member_tys {
            adts_of(t, &mut adts);
        }
        adts.extend(adt.dispatch_ids.iter().copied());
    }
    closure.sort_unstable_by_key(|a| a.index());
    for a in closure {
        let Some(adt) = res.adts.get(a) else {
            let _ = write!(h, "{}:prim;", a.index());
            continue;
        };
        let member_tys: Vec<Ty> = adt.member_tys.iter().map(|t| canon.ty(t)).collect();
        let type_params: Vec<_> = adt.type_params.iter().map(|p| canon.param(*p)).collect();
        let _ = write!(
            h,
            "{}:{}:{member_tys:?}:{:?}:{type_params:?};",
            a.index(),
            g.values.contains(&a),
            adt.dispatch_ids,
        );
    }

    let k = Key {
        hash: h.0.finish(),
        insts: body.instructions.len() as u32,
        blocks: body.blocks.len() as u32,
        locals: body.locals.len() as u32,
    };
    (k, h.1.unwrap_or_default())
}

/// Starts the compile number, for naming dumps.
pub(crate) fn generation() -> u64 {
    STORE.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, |s| s.generation)
}

/// A body's compiled form, relocatable: jump targets relative to the body's
/// first byte, the chunk's offset 0.
#[derive(Clone)]
pub(crate) struct Entry {
    pub(crate) ops: Vec<Op>,
    pub(crate) chunk: Chunk,
    used: u64,
    size: usize,
}

fn retarget(ops: &mut [Op], f: impl Fn(usize) -> usize) {
    for op in ops {
        let one = |t: &mut BlockTarget| {
            if let BlockTarget::ByteOffset(o) = t {
                *o = f(*o);
            }
        };
        match op {
            Op::Jump { target }
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
            | Op::ForNext { target, .. } => one(target),
            Op::Switch { default, table, .. } => {
                one(default);
                table.iter_mut().for_each(one);
            }
            _ => {}
        }
    }
}

impl Entry {
    /// `ops` and `chunk` as compiled at `chunk.offset`.
    pub(crate) fn new(ops: &[Op], chunk: &Chunk) -> Self {
        let base = chunk.offset;
        let mut ops = ops.to_vec();
        retarget(&mut ops, |o| o - base);
        let mut chunk = chunk.clone();
        chunk.offset = 0;
        // a rough footprint for the cap: ops and their side tables
        let size = ops.len() * std::mem::size_of::<Op>()
            + chunk.locs.len() * 32
            + chunk.kinds.len()
            + chunk.locals.len() * 48
            + 256;
        Entry {
            ops,
            chunk,
            used: 0,
            size,
        }
    }

    /// The ops and chunk relocated to `offset`.
    pub(crate) fn at(&self, offset: usize) -> (Vec<Op>, Chunk) {
        let mut ops = self.ops.clone();
        retarget(&mut ops, |o| o + offset);
        let mut chunk = self.chunk.clone();
        chunk.offset = offset;
        (ops, chunk)
    }

    /// The compiled form as text, one op per line, the chunk's maps sorted
    /// (their `HashMap` order is per process).
    fn text(&self) -> String {
        let mut s = String::new();
        for op in &self.ops {
            let _ = writeln!(s, "{op:?}");
        }
        let c = &self.chunk;
        let mut locals: Vec<_> = c.locals.iter().collect();
        locals.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let mut shadows: Vec<_> = c.shadows.iter().collect();
        shadows.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let _ = writeln!(
            s,
            "args {} regs {} params {:?} captures {:?}\nlocals {locals:?}\nshadows {shadows:?}\nlocs {:?}\nkinds {:?}",
            c.args, c.regs, c.params, c.captures, c.locs, c.kinds
        );
        s
    }

    /// Writes `text` (`MIMAS_BODYCACHE_DUMP`, on a `Mode::Check` mismatch).
    pub(crate) fn dump(&self, path: &str) {
        let _ = std::fs::write(path, self.text());
    }

    /// Whether `other` is the same compiled form (for `Mode::Check`).
    pub(crate) fn same(&self, other: &Entry) -> bool {
        self.text() == other.text()
    }
}

struct Store {
    map: HashMap<Key, Entry>,
    /// Compiles so far: an entry's `used` is the last one that read it.
    generation: u64,
    bytes: usize,
}

static STORE: Mutex<Option<Store>> = Mutex::new(None);

thread_local! {
    static LAST: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

/// The hits and misses of the last compile on this thread (the entry body
/// counts as neither).
pub fn body_cache_stats() -> (usize, usize) {
    LAST.with(|c| c.get())
}

pub(crate) fn record(hits: usize, misses: usize) {
    LAST.with(|c| c.set((hits, misses)));
}

fn cap_bytes() -> usize {
    std::env::var("MIMAS_BODYCACHE_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64)
        << 20
}

/// Starts a compile: later `get`s mark what they read as used by it.
pub(crate) fn begin() {
    let mut s = STORE.lock().unwrap_or_else(|e| e.into_inner());
    let s = s.get_or_insert_with(|| Store {
        map: HashMap::new(),
        generation: 0,
        bytes: 0,
    });
    s.generation += 1;
}

pub(crate) fn get(k: &Key) -> Option<Entry> {
    let mut s = STORE.lock().unwrap_or_else(|e| e.into_inner());
    let s = s.as_mut()?;
    let g = s.generation;
    let e = s.map.get_mut(k)?;
    e.used = g;
    Some(e.clone())
}

pub(crate) fn put(k: Key, mut e: Entry) {
    let mut s = STORE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(s) = s.as_mut() else { return };
    e.used = s.generation;
    s.bytes += e.size;
    if let Some(old) = s.map.insert(k, e) {
        s.bytes -= old.size;
    }
}

/// Ends a compile: over the cap, drop the least recently used entries, never
/// one this compile read or wrote.
pub(crate) fn end() {
    let mut s = STORE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(s) = s.as_mut() else { return };
    let cap = cap_bytes();
    if s.bytes <= cap {
        return;
    }
    let mut by_age: Vec<(u64, Key, usize)> = s
        .map
        .iter()
        .filter(|(_, e)| e.used < s.generation)
        .map(|(k, e)| (e.used, *k, e.size))
        .collect();
    by_age.sort_unstable_by_key(|(u, _, _)| *u);
    for (_, k, size) in by_age {
        if s.bytes <= cap {
            break;
        }
        s.map.remove(&k);
        s.bytes -= size;
    }
}

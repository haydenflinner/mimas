//! mimas-wasmgen — emit a freestanding wasm module from a compiled [`Program`].
//!
//! Third specialization lane next to bcgen (Rust source, AOT) and jit
//! (Cranelift, runtime/native): pure byte emission via `wasm-encoder`, so the
//! emitter itself runs anywhere — including inside wasm, i.e. runtime codegen
//! on the browser host where neither rustc nor Cranelift can produce
//! executable code.
//!
//! Unlike the other lanes there is no interpreter *inside* the module to fall
//! back to: a body is emitted only when every register it touches resolves to
//! one scalar class (int/float/bool) and every op is in the supported set.
//! `emit` reports skipped bodies so the caller keeps them on the interpreter.
//!
//! Register model: instead of shadowing a `Val` window like bcgen/jit, each
//! register *is* a wasm local of its resolved class (i64/f64/i32). Calls are
//! statically typed: `CallDirect` (and `Call` whose callee reg is written
//! exactly once by a `LoadBody`) become direct wasm `call`s, with results
//! typed by a cross-body return-class fixpoint — no tags, no `Val`, no entry
//! bail.
//!
//! Divergences from the interpreter, all deliberate at this stage:
//! - checked int arith (`AddInt`/`SubInt`/`MultInt` and `*Imm`) emits an
//!   overflow check that traps (`unreachable`) instead of raising
//!   `RtErr::IntegerOverflow` — same halt, different payload.
//! - `x % 0` / `x / 0` trap in wasm where the interpreter raises
//!   `ModByZero`/`DivByZero` — same control transfer, no `RtErr`.
//! - `i64::MIN % -1` returns 0 in wasm; the interpreter raises overflow.
//!
//! Control flow is lowered structurally (jumps → `br`/`br_if` to `block`/`loop`
//! labels). mimas emits some irreducible CFGs — `while` inside `for` and
//! `break`/`continue` inside `for` place backward-jump landing pads outside the
//! scope any `loop` can span — those bodies are skipped. Covering them needs a
//! relooper or a label-variable dispatch, same as Emscripten does for `goto`.
//!
use std::collections::{BTreeMap, HashMap, HashSet};

use compile::{BinOp, BlockTarget, Constant, Op, Program, Reg, UnaryOp};

pub mod irgen;
use wasm_encoder::{
    BlockType, CodeSection, CustomSection, Encode, EntityType, ExportKind,
    ExportSection, Function, FunctionSection, GlobalType, ImportSection,
    Instruction, MemArg, MemorySection, MemoryType, Module, NameMap, NameSection, TypeSection,
    ValType,
};

/// Scalar register class — the wasm local's type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum K {
    Int,
    Float,
    Bool,
    /// opaque 8-byte word — native passthrough for classless/handle values
    /// (never a register class; only an import-arg/ret kind)
    Word,
}

impl K {
    pub(crate) fn val_type(self) -> ValType {
        match self {
            K::Int => ValType::I64,
            K::Float => ValType::F64,
            K::Bool => ValType::I32,
            K::Word => ValType::I64,
        }
    }
}

/// What one writer proves about a register it stores into.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum W {
    Int,
    Float,
    Bool,
    /// `Move` — inherits the source register's class.
    Copy(u32),
    /// `CallDirect`/resolvable `Call` — the callee body's return class.
    Call(usize),
    /// `GetField`/`GetIndex` — the loaded word's class is whatever the
    /// destination's readers expect (heap values are untyped words).
    Reads,
    Dyn,
}

/// Reason a body wasn't emitted — the caller keeps it on the interpreter.
#[derive(Debug, Clone)]
pub struct Skip {
    pub body: usize,
    pub reason: String,
}

/// One emitted body.
#[derive(Debug, Clone)]
pub struct Body {
    /// Chunk index in the program.
    pub body: usize,
    /// Function index inside the module (emission order).
    pub func: u32,
    /// Export name — `b{body}`.
    pub name: String,
}

pub struct Wasmgen {
    /// Finished `.wasm` binary.
    pub bytes: Vec<u8>,
    /// Emitted bodies, in body order.
    pub bodies: Vec<Body>,
    /// Bodies skipped (unsupported op, untypable register, crossing scopes).
    pub skipped: Vec<Skip>,
}

pub(crate) type Bail = String;

macro_rules! bail {
    ($($a:tt)*) => { return Err(format!($($a)*)) };
}

/// The register a supported pure op writes into. An unclassed dst means no
/// scalar reader exists anywhere, so the op is dead code and can be skipped.
/// (Call-like ops are excluded — they must still execute for side effects.)
pub(crate) fn op_dst(op: &Op) -> Option<Reg> {
    Some(match op {
        Op::Move { dst, .. }
        | Op::LoadConst { dst, .. }
        | Op::ToFloat { dst, .. }
        | Op::Sqrt { dst, .. }
        | Op::Unary { dst, .. }
        | Op::Bin { dst, .. }
        | Op::BoolEq { dst, .. }
        | Op::BoolNe { dst, .. }
        | Op::AddInt { dst, .. }
        | Op::SubInt { dst, .. }
        | Op::MultInt { dst, .. }
        | Op::ModInt { dst, .. }
        | Op::IntLt { dst, .. }
        | Op::IntLe { dst, .. }
        | Op::IntGt { dst, .. }
        | Op::IntGe { dst, .. }
        | Op::IntEq { dst, .. }
        | Op::IntNe { dst, .. }
        | Op::AddIntImm { dst, .. }
        | Op::SubIntImm { dst, .. }
        | Op::MultIntImm { dst, .. }
        | Op::ModIntImm { dst, .. }
        | Op::IntLtImm { dst, .. }
        | Op::IntLeImm { dst, .. }
        | Op::IntGtImm { dst, .. }
        | Op::IntGeImm { dst, .. }
        | Op::IntEqImm { dst, .. }
        | Op::IntNeImm { dst, .. }
        | Op::AddFloat { dst, .. }
        | Op::SubFloat { dst, .. }
        | Op::MultFloat { dst, .. }
        | Op::DivFloat { dst, .. }
        | Op::FloatLt { dst, .. }
        | Op::FloatLe { dst, .. }
        | Op::FloatGt { dst, .. }
        | Op::FloatGe { dst, .. }
        | Op::FloatEq { dst, .. }
        | Op::FloatNe { dst, .. }
        | Op::AddFloatImm { dst, .. }
        | Op::SubFloatImm { dst, .. }
        | Op::MultFloatImm { dst, .. }
        | Op::ModFloatImm { dst, .. }
        | Op::FloatLtImm { dst, .. }
        | Op::FloatLeImm { dst, .. }
        | Op::FloatGtImm { dst, .. }
        | Op::FloatGeImm { dst, .. }
        | Op::FloatEqImm { dst, .. }
        | Op::FloatNeImm { dst, .. } => *dst,
        _ => return None,
    })
}

fn put(writes: &mut HashMap<u32, Vec<(usize, W)>>, r: Reg, w: W, i: usize) {
    writes.entry(r.index() as u32).or_default().push((i, w));
}

fn tgt_off(t: &BlockTarget) -> usize {
    match t {
        BlockTarget::ByteOffset(o) => *o,
        BlockTarget::Block(b) => unreachable!("unresolved block target {:?}", b),
    }
}

// ---------- analysis ----------

pub(crate) const K_INT: u8 = 1;
pub(crate) const K_FLOAT: u8 = 2;
pub(crate) const K_BOOL: u8 = 4;

pub(crate) fn kbit(k: K) -> u8 {
    match k {
        K::Int => K_INT,
        K::Float => K_FLOAT,
        K::Bool => K_BOOL,
        K::Word => 0,
    }
}

pub(crate) struct Ana {
    /// reg -> resolved class — present only when the reg resolves to ONE class
    pub class: HashMap<u32, K>,
    /// call op index -> callee body (`Call` resolved via a constant callee reg, or `CallDirect`)
    pub callee: HashMap<usize, usize>,
    /// resolved return class (None = void/unknown — emitted as a void result)
    pub ret: Option<K>,
}

/// Classify one body's registers. `ret[c]` is the current best guess of body
/// `c`'s return class — iterated to a fixpoint across the program.
pub(crate) fn analyze(ops: &[(usize, Op)], nregs: u32, ret: &[Option<K>]) -> Result<Ana, Bail> {
    let mut writes: HashMap<u32, Vec<(usize, W)>> = HashMap::new();
    let mut reads_i: HashSet<u32> = HashSet::new();
    let mut reads_f: HashSet<u32> = HashSet::new();
    let mut reads_b: HashSet<u32> = HashSet::new();
    // regs that may be constant callees: None once written by anything else
    let mut const_callee: HashMap<u32, Option<usize>> = HashMap::new();
    let mut callee: HashMap<usize, usize> = HashMap::new();
    let mut ret_regs: HashSet<u32> = HashSet::new();

    let mut reads_w: HashSet<u32> = HashSet::new();
    let mut wread = |r: Reg| {
        reads_w.insert(r.index() as u32);
    };
    let mut iread = |r: Reg| {
        reads_i.insert(r.index() as u32);
    };
    let mut fread = |r: Reg| {
        reads_f.insert(r.index() as u32);
    };
    let mut bread = |r: Reg| {
        reads_b.insert(r.index() as u32);
    };

    for (i, (_, op)) in ops.iter().enumerate() {
        match op {
            Op::Move { dst, src } => put(&mut writes, *dst, W::Copy(src.index() as u32), i),
            Op::LoadConst { dst, constant } => put(
                &mut writes,
                *dst,
                match constant {
                    Constant::Int(_) => W::Int,
                    Constant::Float(_) => W::Float,
                    Constant::Bool(_) => W::Bool,
                    _ => W::Dyn,
                },
                i,
            ),
            Op::LoadBody { dst, body } => {
                put(&mut writes, *dst, W::Dyn, i);
                const_callee
                    .entry(dst.index() as u32)
                    .and_modify(|e| *e = None)
                    .or_insert(Some(body.index()));
            }
            Op::AddInt { dst, left, right }
            | Op::SubInt { dst, left, right }
            | Op::MultInt { dst, left, right }
            | Op::ModInt { dst, left, right } => {
                iread(*left);
                iread(*right);
                put(&mut writes, *dst, W::Int, i);
            }
            Op::IntLt { dst, left, right }
            | Op::IntLe { dst, left, right }
            | Op::IntGt { dst, left, right }
            | Op::IntGe { dst, left, right }
            | Op::IntEq { dst, left, right }
            | Op::IntNe { dst, left, right } => {
                iread(*left);
                iread(*right);
                put(&mut writes, *dst, W::Bool, i);
            }
            Op::AddIntImm { dst, left, .. }
            | Op::SubIntImm { dst, left, .. }
            | Op::MultIntImm { dst, left, .. }
            | Op::ModIntImm { dst, left, .. } => {
                iread(*left);
                put(&mut writes, *dst, W::Int, i);
            }
            Op::IntLtImm { dst, left, .. }
            | Op::IntLeImm { dst, left, .. }
            | Op::IntGtImm { dst, left, .. }
            | Op::IntGeImm { dst, left, .. }
            | Op::IntEqImm { dst, left, .. }
            | Op::IntNeImm { dst, left, .. } => {
                iread(*left);
                put(&mut writes, *dst, W::Bool, i);
            }
            Op::AddFloat { dst, left, right }
            | Op::SubFloat { dst, left, right }
            | Op::MultFloat { dst, left, right }
            | Op::DivFloat { dst, left, right } => {
                fread(*left);
                fread(*right);
                put(&mut writes, *dst, W::Float, i);
            }
            Op::FloatLt { dst, left, right }
            | Op::FloatLe { dst, left, right }
            | Op::FloatGt { dst, left, right }
            | Op::FloatGe { dst, left, right }
            | Op::FloatEq { dst, left, right }
            | Op::FloatNe { dst, left, right } => {
                fread(*left);
                fread(*right);
                put(&mut writes, *dst, W::Bool, i);
            }
            Op::AddFloatImm { dst, left, .. }
            | Op::SubFloatImm { dst, left, .. }
            | Op::MultFloatImm { dst, left, .. }
            | Op::ModFloatImm { dst, left, .. } => {
                fread(*left);
                put(&mut writes, *dst, W::Float, i);
            }
            Op::FloatLtImm { dst, left, .. }
            | Op::FloatLeImm { dst, left, .. }
            | Op::FloatGtImm { dst, left, .. }
            | Op::FloatGeImm { dst, left, .. }
            | Op::FloatEqImm { dst, left, .. }
            | Op::FloatNeImm { dst, left, .. } => {
                fread(*left);
                put(&mut writes, *dst, W::Bool, i);
            }
            Op::Len { dst, src } => {
                wread(*src);
                put(&mut writes, *dst, W::Int, i);
            }
            Op::ToFloat { dst, src } => {
                iread(*src);
                put(&mut writes, *dst, W::Float, i);
            }
            Op::Sqrt { dst, src } => {
                fread(*src);
                put(&mut writes, *dst, W::Float, i);
            }
            Op::ForNext { idx, bound, .. } => {
                iread(*idx);
                iread(*bound);
                put(&mut writes, *idx, W::Int, i);
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
            Op::BoolEq { dst, left, right } | Op::BoolNe { dst, left, right } => {
                bread(*left);
                bread(*right);
                put(&mut writes, *dst, W::Bool, i);
            }
            Op::StrEq { dst, left, right } | Op::StrNe { dst, left, right } => {
                wread(*left);
                wread(*right);
                put(&mut writes, *dst, W::Bool, i);
            }
            Op::IsRaised { dst, src } => {
                wread(*src);
                put(&mut writes, *dst, W::Bool, i);
            }
            Op::In { dst, needle, haystack, .. } => {
                wread(*needle);
                wread(*haystack);
                put(&mut writes, *dst, W::Bool, i);
            }
            Op::IsInstance { dst, .. } => {
                put(&mut writes, *dst, W::Bool, i);
            }
            Op::JumpIf { cond, .. } => bread(*cond),
            Op::Switch { scrut, .. } => iread(*scrut),
            Op::Return { val } => {
                ret_regs.insert(val.index() as u32);
                wread(*val);
            }
            Op::Bin { dst, .. } | Op::Unary { dst, .. } => {
                // operand class resolves at emit time; dst is classed by its reads
                put(&mut writes, *dst, W::Reads, i);
            }
            Op::CallDirect { dst, body, args } => {
                callee.insert(i, body.index());
                // emit_call reads each arg at the callee sig's class —
                // a raw word read is the safe superset
                for a in args {
                    wread(*a);
                }
                put(&mut writes, *dst, W::Call(body.index()), i);
            }
            Op::Call {
                dst,
                callee: creg,
                args,
            } => {
                for a in args {
                    wread(*a);
                }
                if let Some(&Some(b)) = const_callee.get(&(creg.index() as u32)) {
                    callee.insert(i, b);
                    put(&mut writes, *dst, W::Call(b), i);
                } else {
                    put(&mut writes, *dst, W::Dyn, i);
                }
            }
            // dst inherits the loaded word's class from its readers —
            // W::Reads contributes the dst's read-set to the union
            Op::GetField { dst, src, .. } => {
                wread(*src);
                put(&mut writes, *dst, W::Reads, i);
            }
            Op::GetIndex { dst, set, index, .. } => {
                iread(*index);
                wread(*set);
                put(&mut writes, *dst, W::Reads, i);
            }
            Op::LoadEntry { dst, .. }
            | Op::NewArray { dst }
            | Op::NewDict { dst }
            | Op::NewClosure { dst, .. }
            | Op::Format { dst, .. } => put(&mut writes, *dst, W::Dyn, i),
            // `regs[d] = regs[s]` minus Null/Raised — a word passthrough,
            // dst inherits src's class like Move
            Op::Unwrap { dst, src } => {
                wread(*src);
                put(&mut writes, *dst, W::Copy(src.index() as u32), i);
            }
            Op::UnwrapRaised { dst, src } => {
                wread(*src);
                put(&mut writes, *dst, W::Reads, i);
            }
            Op::UnwrapUnit { dst, src } => {
                wread(*src);
                put(&mut writes, *dst, W::Reads, i);
            }
            // native dst is classed by its reads; args are read raw
            Op::CallNative { .. } => {}
            Op::SetIndex { set, index, value, .. } => {
                iread(*index);
                wread(*set);
                wread(*value);
            }
            Op::Push { array, value, .. } => {
                wread(*array);
                wread(*value);
            }
            Op::SetField { receiver, value, .. } => {
                wread(*receiver);
                wread(*value);
            }
            Op::NewInstance { dst, fields, .. } => {
                put(&mut writes, *dst, W::Dyn, i);
                for f in fields {
                    wread(*f);
                }
            }
            Op::Jump { .. }
            | Op::Insert { .. }
            | Op::StoreEntry { .. }
            | Op::Panic {}
            | Op::Raise { .. } => {}
        }
    }

    // never-written regs (params, scratch): the reads decide
    let mut mask: HashMap<u32, u8> = HashMap::new();
    for r in 0..nregs {
        if writes.contains_key(&r) {
            continue;
        }
        let m = (reads_i.contains(&r) as u8) * K_INT
            | (reads_f.contains(&r) as u8) * K_FLOAT
            | (reads_b.contains(&r) as u8) * K_BOOL;
        if m != 0 {
            mask.insert(r, m);
        }
    }
    // fixpoint over written regs — a reg may carry a multi-bit mask
    // `extra`: read-classes demanded through Move copies — `r = copy(s)`
    // means s's slot must produce whatever r's readers demand
    let mut extra: HashMap<u32, u8> = HashMap::new();
    loop {
        let mut changed = false;
        let snap = mask.clone();
        let snap_extra = extra.clone();
        for (&r, ws) in &writes {
            // union of classes the writes produce — a reg may hold any of
            // them at runtime; each read site picks its own context's width
            let need = (reads_i.contains(&r) as u8) * K_INT
                | (reads_f.contains(&r) as u8) * K_FLOAT
                | (reads_b.contains(&r) as u8) * K_BOOL
                | snap_extra.get(&r).copied().unwrap_or(0);
            let mut m = 0u8;
            for w in ws {
                m |= match w.1 {
                    W::Int => K_INT,
                    W::Float => K_FLOAT,
                    W::Bool => K_BOOL,
                    W::Copy(s) => snap.get(&s).copied().unwrap_or(0),
                    W::Call(c) => match ret[c].map(kbit) {
                        Some(bits) if bits != 0 => bits,
                        // callee ret unconstrained (word/void): the dst's
                        // readers pick the width
                        _ => {
                            if need == 0 {
                                K_INT | K_FLOAT | K_BOOL
                            } else {
                                need
                            }
                        }
                    },
                    W::Reads => {
                        // untyped word: no scalar reads => grant all classes so
                        // consumers raw-load at whatever width they need (sound
                        // for class-stable heaps); else grant the read classes
                        if need == 0 {
                            K_INT | K_FLOAT | K_BOOL
                        } else {
                            need
                        }
                    }
                    W::Dyn => 0,
                };
            }
            // every declared read must be covered by the mask, else the reg
            // can't serve its readers and the body can't emit
            if need & !m != 0 {
                m = 0;
            }
            if snap.get(&r).copied().unwrap_or(0) != m {
                if m == 0 {
                    mask.remove(&r);
                } else {
                    mask.insert(r, m);
                }
                changed = true;
            }
            // copy sources owe every class their copy's readers demand
            for w in ws {
                if let W::Copy(s) = w.1 {
                    let e = extra.entry(s).or_insert(0);
                    if *e & need != need {
                        *e |= need;
                        changed = true;
                    }
                }
            }
            // word-read-ness flows through copies: `y = x; heap_op(y)`
            // means x's word is what the heap op sees
            if !reads_w.contains(&r)
                && ws.iter().any(|w| matches!(w.1, W::Copy(s) if reads_w.contains(&s)))
            {
                reads_w.insert(r);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // never-written regs (params) were seeded before the loop — merge
    // copy-demand classes in afterwards
    for (&s, &e) in &extra {
        if !writes.contains_key(&s) && e != 0 {
            *mask.entry(s).or_insert(0) |= e;
        }
    }
    // classless native args are raw-word reads at emit time (handle
    // passthrough) — their producers must keep the word
    for (_, op) in ops {
        if let Op::CallNative { args, .. } = op {
            for a in args {
                let r = a.index() as u32;
                if !mask.contains_key(&r) {
                    reads_w.insert(r);
                }
            }
        }
    }
    let class: HashMap<u32, K> = mask
        .iter()
        .filter_map(|(&r, &m)| match m {
            K_INT => Some((r, K::Int)),
            K_FLOAT => Some((r, K::Float)),
            K_BOOL => Some((r, K::Bool)),
            _ => None,
        })
        .collect();

    // the return class every Return reg agrees on (multi-class regs intersect)
    let mut cand = 7u8;
    for &r in &ret_regs {
        cand &= mask.get(&r).copied().unwrap_or(0);
    }
    let ret_k = match cand {
        K_INT => Some(K::Int),
        K_FLOAT => Some(K::Float),
        K_BOOL => Some(K::Bool),
        // returns disagree in class (or param passthrough): the raw word
        // carries any of them, callers read their own width
        _ if !ret_regs.is_empty() => Some(K::Word),
        _ => None,
    };
    Ok(Ana {
        class,
        callee,
        ret: ret_k,
    })
}

// ---------- scope analysis ----------

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum ScopeKind {
    Loop,
    Block,
}

/// A wasm `loop`/`block` delimiting a branch target. `open`/`close` are op
/// indices; `close` is exclusive — the scope covers ops `open..close`.
#[derive(Clone, Copy)]
pub(crate) struct Scope {
    pub open: usize,
    pub close: usize,
    pub kind: ScopeKind,
    pub target: usize, // op index the branch lands on
}

/// Every backward target → a `loop` opened at the target enclosing its last
/// jumper; every forward target → a `block` opened at the earliest jumper and
/// closed at the target. Mimas's compiler emits reducible CFGs so intervals
/// nest; a crossing is a bail.
pub(crate) fn scopes(ops: &[(usize, Op)], nops: usize) -> Result<Vec<Scope>, Bail> {
    let mut at: HashMap<usize, usize> = HashMap::new();
    for (i, (off, _)) in ops.iter().enumerate() {
        at.insert(*off, i);
    }
    // target op index -> (is_backward, extreme jumper)
    let mut targets: HashMap<usize, (bool, usize)> = HashMap::new();
    for (j, (_, op)) in ops.iter().enumerate() {
        let mut mark = |t: &BlockTarget| -> Result<(), Bail> {
            let to = tgt_off(t);
            let last_off = ops.last().map(|x| x.0).unwrap_or(0);
            let ti = if to > last_off || (to == last_off && !at.contains_key(&to)) {
                nops
            } else {
                *at.get(&to)
                    .ok_or_else(|| format!("jump target {to} mid-op"))?
            };
            let backward = ti <= j;
            let ent = targets.entry(ti).or_insert((backward, j));
            if ent.0 != backward {
                bail!("mixed fwd/back jumps to op {ti}");
            }
            ent.1 = if backward { ent.1.max(j) } else { ent.1.min(j) };
            Ok(())
        };
        match op {
            Op::Jump { target } | Op::JumpIf { target, .. } | Op::ForNext { target, .. } => {
                mark(target)?
            }
            Op::BIntLt { target, .. }
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
            | Op::BFloatNeImm { target, .. } => mark(target)?,
            Op::Switch { default, table, .. } => {
                mark(default)?;
                for t in table {
                    mark(t)?;
                }
            }
            _ => {}
        }
    }
    let mut out: Vec<Scope> = targets
        .into_iter()
        .map(|(t, (backward, j))| {
            if backward {
                Scope {
                    open: t,
                    close: j + 1,
                    kind: ScopeKind::Loop,
                    target: t,
                }
            } else {
                Scope {
                    open: j,
                    close: t,
                    kind: ScopeKind::Block,
                    target: t,
                }
            }
        })
        .collect();
    // repair until the intervals nest cleanly. A loop's `close` can always be
    // extended (its end is dead fallthrough); a block's `open` can always be
    // moved earlier (its branches still land at `target`). A block expiring
    // inside a `loop` is irreducible — that needs a relooper/label variable.
    for _ in 0..64 {
        match simulate(&mut out, nops) {
            Ok(()) => return Ok(out),
            Err(Repair::ExtendLoop(ix, nc)) => {
                out[ix].close = nc;
            }
            Err(Repair::ShiftBlock(ix, no)) => {
                out[ix].open = no;
            }
            Err(Repair::Fatal(e)) => return Err(e),
        }
    }
    bail!("scope repair did not converge")
}

pub(crate) enum Repair {
    /// loop `ix` expired while `top` was still open: extend its close past `top`'s.
    ExtendLoop(usize, usize),
    /// block `ix` expired inside block `top`: move `top`'s open to `ix`'s so `ix` nests inside it.
    ShiftBlock(usize, usize),
    Fatal(Bail),
}

pub(crate) fn simulate(out: &mut Vec<Scope>, nops: usize) -> Result<(), Repair> {
    // index-based simulation: opens/closes hold indices into `out`. Order
    // matters for nesting: an outer scope (earlier open or, at a tie, later
    // close) must be pushed first; pops at a shared close go innermost-first.
    out.sort_by_key(|s| (s.open, std::cmp::Reverse(s.close), s.kind));
    let mut open_ix: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut close_ix: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (ix, s) in out.iter().enumerate() {
        open_ix.entry(s.open).or_default().push(ix);
    }
    for (ix, s) in out.iter().enumerate().rev() {
        close_ix.entry(s.close).or_default().push(ix);
    }
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..=nops {
        if let Some(list) = close_ix.get(&i) {
            for &ix in list {
                match stack.last().copied() {
                    Some(top) if top == ix => {
                        stack.pop();
                    }
                    Some(top) => {
                        if !stack.contains(&ix) {
                            return Err(Repair::Fatal(format!("scope underflow at {i}")));
                        }
                        return Err(match (out[ix].kind, out[top].kind) {
                            (ScopeKind::Loop, _) => Repair::ExtendLoop(ix, out[top].close),
                            (ScopeKind::Block, ScopeKind::Block) => {
                                Repair::ShiftBlock(top, out[ix].open)
                            }
                            _ => Repair::Fatal(format!(
                                "crossing scopes: {:?} [{},{}) expires inside {:?} [{},{})",
                                out[ix].kind,
                                out[ix].open,
                                out[ix].close,
                                out[top].kind,
                                out[top].open,
                                out[top].close
                            )),
                        });
                    }
                    None => return Err(Repair::Fatal(format!("scope underflow at {i}"))),
                }
            }
        }
        if let Some(list) = open_ix.remove(&i) {
            for ix in list {
                stack.push(ix);
            }
        }
    }
    if !stack.is_empty() {
        return Err(Repair::Fatal("unclosed scopes at end".into()));
    }
    Ok(())
}

// ---------- emission ----------

#[derive(Clone, Debug)]
pub(crate) struct Sig {
    pub params: Vec<K>,
    pub ret: Option<K>,
}

struct Em<'a> {
    ops: &'a [(usize, Op)],
    class: &'a HashMap<u32, K>,
    local: &'a HashMap<u32, u32>,
    sigs: &'a [Option<Sig>],
    func_map: &'a HashMap<usize, u32>,
    callee: &'a HashMap<usize, usize>,
    natives: &'a HashMap<(u32, Vec<K>, Option<K>), u32>,
    ret_k: Option<K>,
    tmp: u32, // i64 scratch local for checked arith
    stack: Vec<Scope>,
    f: Function,
    /// byte offset into the function body, for the srcmap
    code_off: u32,
    /// (code_off, op_index) per emitted op
    srcmap: Vec<(u32, u32)>,
    fuel_g: Option<u32>,
    pause_g: Option<u32>,
    /// memory address of this body's coverage byte for op i
    cov_base: u32,
}

impl Em<'_> {
    fn ins(&mut self, i: Instruction) -> &mut Self {
        let mut v = Vec::with_capacity(8);
        i.encode(&mut v);
        self.code_off += v.len() as u32;
        self.f.raw(v);
        self
    }

    /// Per-op instrumentation, emitted just before the op itself.
    fn tick(&mut self, i: usize) {
        if let Some(g) = self.fuel_g {
            self.ins(Instruction::GlobalGet(g));
            self.ins(Instruction::I64Const(1));
            self.ins(Instruction::I64Sub);
            self.ins(Instruction::GlobalSet(g));
            self.ins(Instruction::GlobalGet(g));
            self.ins(Instruction::I64Const(0));
            self.ins(Instruction::I64LtS);
            self.trap_if();
        }
        if self.cov_base != u32::MAX {
            self.ins(Instruction::I32Const((self.cov_base + i as u32) as i32));
            self.ins(Instruction::I32Const(1));
            self.ins(Instruction::I32Store8(MemArg {
                offset: 0,
                align: 0,
                memory_index: 0,
            }));
        }
    }

    fn k(&self, r: Reg) -> Result<K, Bail> {
        self.class
            .get(&(r.index() as u32))
            .copied()
            .ok_or_else(|| format!("reg {} has no scalar class", r.index()))
    }

    fn li(&self, r: Reg) -> u32 {
        self.local[&(r.index() as u32)]
    }

    /// `local.get r` coerced to `want`.
    fn get(&mut self, r: Reg, want: K) -> Result<(), Bail> {
        let k = self.k(r)?;
        self.ins(Instruction::LocalGet(self.li(r)));
        if k != want {
            match (k, want) {
                (K::Int, K::Float) => {
                    self.ins(Instruction::F64ConvertI64S);
                }
                (K::Float, K::Int) => {
                    self.ins(Instruction::I64TruncF64S);
                }
                (K::Bool, K::Int) => {
                    self.ins(Instruction::I64ExtendI32U);
                }
                (K::Bool, K::Float) => {
                    self.ins(Instruction::F64ConvertI32S);
                }
                (K::Int, K::Bool) => {
                    self.ins(Instruction::I64Const(0));
                    self.ins(Instruction::I64Ne);
                }
                (K::Float, K::Bool) => {
                    self.ins(Instruction::F64Const(0.0f64.into()));
                    self.ins(Instruction::F64Ne);
                }
                _ => bail!("coerce {k:?}->{want:?}"),
            }
        }
        Ok(())
    }

    fn set(&mut self, r: Reg) {
        self.ins(Instruction::LocalSet(self.li(r)));
    }

    fn depth(&self, t: usize) -> Result<(u32, ScopeKind), Bail> {
        for (i, s) in self.stack.iter().enumerate().rev() {
            if s.target == t {
                return Ok(((self.stack.len() - 1 - i) as u32, s.kind));
            }
        }
        bail!("no open scope for target op {t}")
    }

    fn target_idx(&self, t: &BlockTarget, off2idx: &HashMap<usize, usize>) -> Result<usize, Bail> {
        let to = tgt_off(t);
        off2idx
            .get(&to)
            .copied()
            .ok_or_else(|| format!("jump target {to} mid-op"))
    }

    fn br(&mut self, t: &BlockTarget, off2idx: &HashMap<usize, usize>) -> Result<(), Bail> {
        let (d, kind) = self.depth(self.target_idx(t, off2idx)?)?;
        if kind == ScopeKind::Loop {
            if let Some(g) = self.pause_g {
                self.ins(Instruction::GlobalGet(g));
                self.trap_if();
            }
        }
        self.ins(Instruction::Br(d));
        Ok(())
    }

    fn br_if(
        &mut self,
        t: &BlockTarget,
        is_true: bool,
        off2idx: &HashMap<usize, usize>,
    ) -> Result<(), Bail> {
        if !is_true {
            self.ins(Instruction::I32Eqz);
        }
        let (d, kind) = self.depth(self.target_idx(t, off2idx)?)?;
        if kind == ScopeKind::Loop {
            if let Some(g) = self.pause_g {
                self.ins(Instruction::GlobalGet(g));
                self.trap_if();
            }
        }
        self.ins(Instruction::BrIf(d));
        Ok(())
    }

    fn trap_if(&mut self) {
        // stack: i32 cond — trap when true
        self.ins(Instruction::If(BlockType::Empty));
        self.ins(Instruction::Unreachable);
        self.ins(Instruction::End);
    }

    /// `l OP r` checked for i64 overflow; leaves the result on the stack.
    fn checked_int(&mut self, l: Reg, r: Reg, op: BinOp) -> Result<(), Bail> {
        match op {
            BinOp::Mod | BinOp::IDiv => {
                // wasm traps on 0 divisor itself (same halt as RtErr)
                self.get(l, K::Int)?;
                self.get(r, K::Int)?;
                self.ins(if op == BinOp::Mod {
                    Instruction::I64RemS
                } else {
                    Instruction::I64DivS
                });
                return Ok(());
            }
            _ => {}
        }
        let tmp = self.tmp;
        self.get(l, K::Int)?;
        self.get(r, K::Int)?;
        self.ins(match op {
            BinOp::Add => Instruction::I64Add,
            BinOp::Sub => Instruction::I64Sub,
            BinOp::Mult => Instruction::I64Mul,
            _ => bail!("checked_int {op:?}"),
        });
        self.ins(Instruction::LocalSet(tmp));
        match op {
            // add: r>0 && tmp<l  ||  r<0 && tmp>l ; sub: r<0 && tmp>l || r>0 && tmp<l
            BinOp::Add | BinOp::Sub => {
                let (c1, c2) = if op == BinOp::Add {
                    (Instruction::I64LtS, Instruction::I64GtS)
                } else {
                    (Instruction::I64GtS, Instruction::I64LtS)
                };
                self.ins(Instruction::LocalGet(self.li(r)));
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64GtS);
                self.ins(Instruction::LocalGet(tmp));
                self.ins(Instruction::LocalGet(self.li(l)));
                self.ins(c1);
                self.ins(Instruction::I32And);
                self.ins(Instruction::LocalGet(self.li(r)));
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64LtS);
                self.ins(Instruction::LocalGet(tmp));
                self.ins(Instruction::LocalGet(self.li(l)));
                self.ins(c2);
                self.ins(Instruction::I32And);
                self.ins(Instruction::I32Or);
                self.trap_if();
            }
            // mul: r!=0 && tmp/r != l → trap (INT_MIN*-1 also traps in the div)
            BinOp::Mult => {
                self.ins(Instruction::LocalGet(self.li(r)));
                self.ins(Instruction::I64Eqz);
                self.ins(Instruction::I32Eqz);
                self.ins(Instruction::If(BlockType::Empty));
                self.ins(Instruction::LocalGet(tmp));
                self.ins(Instruction::LocalGet(self.li(r)));
                self.ins(Instruction::I64DivS);
                self.ins(Instruction::LocalGet(self.li(l)));
                self.ins(Instruction::I64Ne);
                self.trap_if();
                self.ins(Instruction::End);
            }
            _ => unreachable!(),
        }
        self.ins(Instruction::LocalGet(tmp));
        Ok(())
    }

    /// `l OP imm` checked; leaves the result on the stack.
    fn checked_int_imm(&mut self, l: Reg, v: i64, op: BinOp) -> Result<(), Bail> {
        let tmp = self.tmp;
        match op {
            BinOp::Mod | BinOp::IDiv => {
                if v == 0 {
                    self.ins(Instruction::Unreachable);
                    return Ok(());
                }
                self.get(l, K::Int)?;
                self.ins(Instruction::I64Const(v));
                self.ins(if op == BinOp::Mod {
                    Instruction::I64RemS
                } else {
                    Instruction::I64DivS
                });
                return Ok(());
            }
            BinOp::Mult if v == 0 => {
                // l * 0 == 0, no overflow possible
                self.ins(Instruction::I64Const(0));
                return Ok(());
            }
            _ => {}
        }
        self.get(l, K::Int)?;
        self.ins(Instruction::I64Const(v));
        self.ins(match op {
            BinOp::Add => Instruction::I64Add,
            BinOp::Sub => Instruction::I64Sub,
            BinOp::Mult => Instruction::I64Mul,
            _ => bail!("checked_int_imm {op:?}"),
        });
        self.ins(Instruction::LocalSet(tmp));
        match op {
            BinOp::Add | BinOp::Sub => {
                let (c1, c2) = if op == BinOp::Add {
                    (Instruction::I64LtS, Instruction::I64GtS)
                } else {
                    (Instruction::I64GtS, Instruction::I64LtS)
                };
                self.ins(Instruction::I64Const(v));
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64GtS);
                self.ins(Instruction::LocalGet(tmp));
                self.ins(Instruction::LocalGet(self.li(l)));
                self.ins(c1);
                self.ins(Instruction::I32And);
                self.ins(Instruction::I64Const(v));
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64LtS);
                self.ins(Instruction::LocalGet(tmp));
                self.ins(Instruction::LocalGet(self.li(l)));
                self.ins(c2);
                self.ins(Instruction::I32And);
                self.ins(Instruction::I32Or);
                self.trap_if();
            }
            BinOp::Mult => {
                self.ins(Instruction::LocalGet(tmp));
                self.ins(Instruction::I64Const(v));
                self.ins(Instruction::I64DivS);
                self.ins(Instruction::LocalGet(self.li(l)));
                self.ins(Instruction::I64Ne);
                self.trap_if();
            }
            _ => unreachable!(),
        }
        self.ins(Instruction::LocalGet(tmp));
        Ok(())
    }

    fn cmp(&mut self, l: Reg, r: Reg, want: K, i: Instruction) -> Result<(), Bail> {
        self.get(l, want)?;
        self.get(r, want)?;
        self.ins(i);
        Ok(())
    }

    fn cmp_imm(&mut self, l: Reg, v: i64, want: K, i: Instruction) -> Result<(), Bail> {
        self.get(l, want)?;
        match want {
            K::Int => {
                self.ins(Instruction::I64Const(v));
            }
            K::Float => {
                self.ins(Instruction::F64Const(f64::from_bits(v as u64).into()));
            }
            K::Bool => {
                self.ins(Instruction::I32Const(v as i32));
            }
            K::Word => unreachable!("word is never a comparison class"),
        }
        self.ins(i);
        Ok(())
    }

    /// dynamic `Bin` when both operands resolve scalar
    fn emit_bin(&mut self, dst: Reg, l: Reg, op: BinOp, r: Reg) -> Result<(), Bail> {
        match (self.k(l)?, self.k(r)?) {
            (K::Int, K::Int) => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mult | BinOp::Mod | BinOp::IDiv => {
                    self.checked_int(l, r, op)?;
                }
                BinOp::Div => {
                    self.get(l, K::Int)?;
                    self.ins(Instruction::F64ConvertI64S);
                    self.get(r, K::Int)?;
                    self.ins(Instruction::F64ConvertI64S);
                    self.ins(Instruction::F64Div);
                }
                BinOp::LessThan => self.cmp(l, r, K::Int, Instruction::I64LtS)?,
                BinOp::LessEqual => self.cmp(l, r, K::Int, Instruction::I64LeS)?,
                BinOp::GreaterThan => self.cmp(l, r, K::Int, Instruction::I64GtS)?,
                BinOp::GreaterEqual => self.cmp(l, r, K::Int, Instruction::I64GeS)?,
                BinOp::Identity => self.cmp(l, r, K::Int, Instruction::I64Eq)?,
                BinOp::NotEqual => self.cmp(l, r, K::Int, Instruction::I64Ne)?,
                BinOp::BitAnd => self.cmp(l, r, K::Int, Instruction::I64And)?,
                BinOp::BitOr => self.cmp(l, r, K::Int, Instruction::I64Or)?,
                BinOp::BitXor | BinOp::Xor => self.cmp(l, r, K::Int, Instruction::I64Xor)?,
                BinOp::BitShiftLeft => self.cmp(l, r, K::Int, Instruction::I64Shl)?,
                BinOp::BitShiftRight => self.cmp(l, r, K::Int, Instruction::I64ShrS)?,
                _ => bail!("Bin {op:?} on ints"),
            },
            (K::Float, K::Float) => match op {
                BinOp::Add => self.cmp(l, r, K::Float, Instruction::F64Add)?,
                BinOp::Sub => self.cmp(l, r, K::Float, Instruction::F64Sub)?,
                BinOp::Mult => self.cmp(l, r, K::Float, Instruction::F64Mul)?,
                BinOp::Div | BinOp::IDiv => self.cmp(l, r, K::Float, Instruction::F64Div)?,
                BinOp::Mod => {
                    // a - trunc(a/b)*b — wasm has no fmod
                    self.get(l, K::Float)?;
                    self.get(l, K::Float)?;
                    self.get(r, K::Float)?;
                    self.ins(Instruction::F64Div);
                    self.ins(Instruction::F64Trunc);
                    self.get(r, K::Float)?;
                    self.ins(Instruction::F64Mul);
                    self.ins(Instruction::F64Sub);
                }
                BinOp::LessThan => self.cmp(l, r, K::Float, Instruction::F64Lt)?,
                BinOp::LessEqual => self.cmp(l, r, K::Float, Instruction::F64Le)?,
                BinOp::GreaterThan => self.cmp(l, r, K::Float, Instruction::F64Gt)?,
                BinOp::GreaterEqual => self.cmp(l, r, K::Float, Instruction::F64Ge)?,
                BinOp::Identity => self.cmp(l, r, K::Float, Instruction::F64Eq)?,
                BinOp::NotEqual => self.cmp(l, r, K::Float, Instruction::F64Ne)?,
                _ => bail!("Bin {op:?} on floats"),
            },
            (K::Bool, K::Bool) => match op {
                BinOp::Identity => self.cmp(l, r, K::Bool, Instruction::I32Eq)?,
                BinOp::NotEqual | BinOp::Xor => self.cmp(l, r, K::Bool, Instruction::I32Ne)?,
                BinOp::And => self.cmp(l, r, K::Bool, Instruction::I32And)?,
                BinOp::Or => self.cmp(l, r, K::Bool, Instruction::I32Or)?,
                _ => bail!("Bin {op:?} on bools"),
            },
            (a, b) => bail!("Bin {op:?} on {a:?}/{b:?}"),
        }
        let dk = self.k(dst)?;
        // result of comparisons/bool ops lands as i32 → store into a Bool local
        let produced = match op {
            BinOp::LessThan
            | BinOp::LessEqual
            | BinOp::GreaterThan
            | BinOp::GreaterEqual
            | BinOp::Identity
            | BinOp::NotEqual
            | BinOp::And
            | BinOp::Or
            | BinOp::Xor => K::Bool,
            BinOp::Div if self.k(l)? == K::Int => K::Float,
            _ => dk,
        };
        if produced != dk {
            bail!("Bin dst class {dk:?} != produced {produced:?}");
        }
        self.set(dst);
        Ok(())
    }

    fn emit(&mut self, off2idx: &HashMap<usize, usize>, scopes: Vec<Scope>) -> Result<(), Bail> {
        let _ = self.ops.len();
        let mut opens: BTreeMap<usize, Vec<Scope>> = BTreeMap::new();
        for s in scopes {
            opens.entry(s.open).or_default().push(s);
        }
        for (i, (_, op)) in self.ops.iter().enumerate() {
            while let Some(last) = self.stack.last() {
                if last.close <= i {
                    self.ins(Instruction::End);
                    self.stack.pop();
                } else {
                    break;
                }
            }
            if let Some(mut group) = opens.remove(&i) {
                group.sort_by_key(|s| (std::cmp::Reverse(s.close), s.kind));
                for s in group {
                    self.ins(match s.kind {
                        ScopeKind::Loop => Instruction::Loop(BlockType::Empty),
                        ScopeKind::Block => Instruction::Block(BlockType::Empty),
                    });
                    self.stack.push(s);
                }
            }
            self.srcmap.push((self.code_off, i as u32));
            self.tick(i);
            // a scalar op whose dst is unclassed has no scalar reader — the
            // write is dead, so the whole op is skipped (all such ops are pure)
            if let Some(d) = op_dst(op) {
                if !self.class.contains_key(&(d.index() as u32)) {
                    continue;
                }
            }
            match op {
                Op::Move { dst, src } => {
                    let k = self.k(*dst)?;
                    self.get(*src, k)?;
                    self.set(*dst);
                }
                Op::LoadConst { dst, constant } => {
                    match (self.k(*dst)?, constant) {
                        (K::Int, Constant::Int(v)) => {
                            self.ins(Instruction::I64Const(*v));
                        }
                        (K::Float, Constant::Float(v)) => {
                            self.ins(Instruction::F64Const((*v).into()));
                        }
                        (K::Bool, Constant::Bool(v)) => {
                            self.ins(Instruction::I32Const(*v as i32));
                        }
                        (k, c) => bail!("LoadConst {c:?} into {k:?}"),
                    }
                    self.set(*dst);
                }
                Op::AddInt { dst, left, right } => {
                    self.checked_int(*left, *right, BinOp::Add)?;
                    self.set(*dst);
                }
                Op::SubInt { dst, left, right } => {
                    self.checked_int(*left, *right, BinOp::Sub)?;
                    self.set(*dst);
                }
                Op::MultInt { dst, left, right } => {
                    self.checked_int(*left, *right, BinOp::Mult)?;
                    self.set(*dst);
                }
                Op::ModInt { dst, left, right } => {
                    self.checked_int(*left, *right, BinOp::Mod)?;
                    self.set(*dst);
                }
                Op::AddIntImm { dst, left, val } => {
                    self.checked_int_imm(*left, *val, BinOp::Add)?;
                    self.set(*dst);
                }
                Op::SubIntImm { dst, left, val } => {
                    self.checked_int_imm(*left, *val, BinOp::Sub)?;
                    self.set(*dst);
                }
                Op::MultIntImm { dst, left, val } => {
                    self.checked_int_imm(*left, *val, BinOp::Mult)?;
                    self.set(*dst);
                }
                Op::ModIntImm { dst, left, val } => {
                    self.checked_int_imm(*left, *val, BinOp::Mod)?;
                    self.set(*dst);
                }
                Op::IntLt { dst, left, right } => self
                    .cmp(*left, *right, K::Int, Instruction::I64LtS)
                    .map(|_| self.set(*dst))?,
                Op::IntLe { dst, left, right } => self
                    .cmp(*left, *right, K::Int, Instruction::I64LeS)
                    .map(|_| self.set(*dst))?,
                Op::IntGt { dst, left, right } => self
                    .cmp(*left, *right, K::Int, Instruction::I64GtS)
                    .map(|_| self.set(*dst))?,
                Op::IntGe { dst, left, right } => self
                    .cmp(*left, *right, K::Int, Instruction::I64GeS)
                    .map(|_| self.set(*dst))?,
                Op::IntEq { dst, left, right } => self
                    .cmp(*left, *right, K::Int, Instruction::I64Eq)
                    .map(|_| self.set(*dst))?,
                Op::IntNe { dst, left, right } => self
                    .cmp(*left, *right, K::Int, Instruction::I64Ne)
                    .map(|_| self.set(*dst))?,
                Op::IntLtImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Int, Instruction::I64LtS)
                    .map(|_| self.set(*dst))?,
                Op::IntLeImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Int, Instruction::I64LeS)
                    .map(|_| self.set(*dst))?,
                Op::IntGtImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Int, Instruction::I64GtS)
                    .map(|_| self.set(*dst))?,
                Op::IntGeImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Int, Instruction::I64GeS)
                    .map(|_| self.set(*dst))?,
                Op::IntEqImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Int, Instruction::I64Eq)
                    .map(|_| self.set(*dst))?,
                Op::IntNeImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Int, Instruction::I64Ne)
                    .map(|_| self.set(*dst))?,
                Op::AddFloat { dst, left, right } => self
                    .cmp(*left, *right, K::Float, Instruction::F64Add)
                    .map(|_| self.set(*dst))?,
                Op::SubFloat { dst, left, right } => self
                    .cmp(*left, *right, K::Float, Instruction::F64Sub)
                    .map(|_| self.set(*dst))?,
                Op::MultFloat { dst, left, right } => self
                    .cmp(*left, *right, K::Float, Instruction::F64Mul)
                    .map(|_| self.set(*dst))?,
                Op::DivFloat { dst, left, right } => self
                    .cmp(*left, *right, K::Float, Instruction::F64Div)
                    .map(|_| self.set(*dst))?,
                Op::FloatLt { dst, left, right } => self
                    .cmp(*left, *right, K::Float, Instruction::F64Lt)
                    .map(|_| self.set(*dst))?,
                Op::FloatLe { dst, left, right } => self
                    .cmp(*left, *right, K::Float, Instruction::F64Le)
                    .map(|_| self.set(*dst))?,
                Op::FloatGt { dst, left, right } => self
                    .cmp(*left, *right, K::Float, Instruction::F64Gt)
                    .map(|_| self.set(*dst))?,
                Op::FloatGe { dst, left, right } => self
                    .cmp(*left, *right, K::Float, Instruction::F64Ge)
                    .map(|_| self.set(*dst))?,
                Op::FloatEq { dst, left, right } => self
                    .cmp(*left, *right, K::Float, Instruction::F64Eq)
                    .map(|_| self.set(*dst))?,
                Op::FloatNe { dst, left, right } => self
                    .cmp(*left, *right, K::Float, Instruction::F64Ne)
                    .map(|_| self.set(*dst))?,
                Op::AddFloatImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Float, Instruction::F64Add)
                    .map(|_| self.set(*dst))?,
                Op::SubFloatImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Float, Instruction::F64Sub)
                    .map(|_| self.set(*dst))?,
                Op::MultFloatImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Float, Instruction::F64Mul)
                    .map(|_| self.set(*dst))?,
                Op::ModFloatImm { dst, left, val } => {
                    let v = f64::from_bits(*val as u64);
                    self.get(*left, K::Float)?;
                    self.get(*left, K::Float)?;
                    self.ins(Instruction::F64Const(v.into()));
                    self.ins(Instruction::F64Div);
                    self.ins(Instruction::F64Trunc);
                    self.ins(Instruction::F64Const(v.into()));
                    self.ins(Instruction::F64Mul);
                    self.ins(Instruction::F64Sub);
                    self.set(*dst);
                }
                Op::FloatLtImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Float, Instruction::F64Lt)
                    .map(|_| self.set(*dst))?,
                Op::FloatLeImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Float, Instruction::F64Le)
                    .map(|_| self.set(*dst))?,
                Op::FloatGtImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Float, Instruction::F64Gt)
                    .map(|_| self.set(*dst))?,
                Op::FloatGeImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Float, Instruction::F64Ge)
                    .map(|_| self.set(*dst))?,
                Op::FloatEqImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Float, Instruction::F64Eq)
                    .map(|_| self.set(*dst))?,
                Op::FloatNeImm { dst, left, val } => self
                    .cmp_imm(*left, *val, K::Float, Instruction::F64Ne)
                    .map(|_| self.set(*dst))?,
                Op::BoolEq { dst, left, right } => self
                    .cmp(*left, *right, K::Bool, Instruction::I32Eq)
                    .map(|_| self.set(*dst))?,
                Op::BoolNe { dst, left, right } => self
                    .cmp(*left, *right, K::Bool, Instruction::I32Ne)
                    .map(|_| self.set(*dst))?,
                Op::ToFloat { dst, src } => {
                    self.get(*src, K::Int)?;
                    self.ins(Instruction::F64ConvertI64S);
                    self.set(*dst);
                }
                Op::Sqrt { dst, src } => {
                    self.get(*src, K::Float)?;
                    self.ins(Instruction::F64Sqrt);
                    self.set(*dst);
                }
                Op::Unary { dst, op, src } => {
                    match op {
                        UnaryOp::Negative => match self.k(*src)? {
                            K::Int => {
                                self.ins(Instruction::I64Const(0));
                                self.get(*src, K::Int)?;
                                self.ins(Instruction::I64Sub);
                            }
                            K::Float => {
                                self.get(*src, K::Float)?;
                                self.ins(Instruction::F64Neg);
                            }
                            k => bail!("unary neg on {k:?}"),
                        },
                        UnaryOp::Not => {
                            self.get(*src, K::Bool)?;
                            self.ins(Instruction::I32Eqz);
                        }
                        UnaryOp::BitwiseNot => {
                            self.get(*src, K::Int)?;
                            self.ins(Instruction::I64Const(-1));
                            self.ins(Instruction::I64Xor);
                        }
                        UnaryOp::Positive => {
                            self.get(*src, K::Int)?;
                        }
                    }
                    self.set(*dst);
                }
                Op::Bin {
                    dst,
                    left,
                    op,
                    right,
                } => self.emit_bin(*dst, *left, *op, *right)?,
                Op::Jump { target } => self.br(target, off2idx)?,
                Op::JumpIf {
                    cond,
                    target,
                    is_true,
                } => {
                    self.get(*cond, K::Bool)?;
                    self.br_if(target, *is_true, off2idx)?;
                }
                Op::ForNext { idx, bound, target } => {
                    self.get(*idx, K::Int)?;
                    self.ins(Instruction::I64Const(1));
                    self.ins(Instruction::I64Add);
                    self.set(*idx);
                    self.get(*idx, K::Int)?;
                    self.get(*bound, K::Int)?;
                    self.ins(Instruction::I64LtS);
                    self.br_if(target, true, off2idx)?;
                }
                Op::Switch {
                    scrut,
                    base,
                    default,
                    table,
                } => {
                    self.get(*scrut, K::Int)?;
                    if *base != 0 {
                        self.ins(Instruction::I64Const(*base as i64));
                        self.ins(Instruction::I64Sub);
                    }
                    self.ins(Instruction::I32WrapI64);
                    let mut tbl = Vec::with_capacity(table.len());
                    for t in table {
                        tbl.push(self.depth(self.target_idx(t, off2idx)?)?.0);
                    }
                    let d = self.depth(self.target_idx(default, off2idx)?)?.0;
                    self.ins(Instruction::BrTable(tbl.into(), d));
                }
                Op::BIntLt {
                    target,
                    left,
                    right,
                    is_true,
                } => self.icmp_br(
                    target,
                    *left,
                    *right,
                    *is_true,
                    Instruction::I64LtS,
                    off2idx,
                )?,
                Op::BIntLe {
                    target,
                    left,
                    right,
                    is_true,
                } => self.icmp_br(
                    target,
                    *left,
                    *right,
                    *is_true,
                    Instruction::I64LeS,
                    off2idx,
                )?,
                Op::BIntGt {
                    target,
                    left,
                    right,
                    is_true,
                } => self.icmp_br(
                    target,
                    *left,
                    *right,
                    *is_true,
                    Instruction::I64GtS,
                    off2idx,
                )?,
                Op::BIntGe {
                    target,
                    left,
                    right,
                    is_true,
                } => self.icmp_br(
                    target,
                    *left,
                    *right,
                    *is_true,
                    Instruction::I64GeS,
                    off2idx,
                )?,
                Op::BIntEq {
                    target,
                    left,
                    right,
                    is_true,
                } => self.icmp_br(target, *left, *right, *is_true, Instruction::I64Eq, off2idx)?,
                Op::BIntNe {
                    target,
                    left,
                    right,
                    is_true,
                } => self.icmp_br(target, *left, *right, *is_true, Instruction::I64Ne, off2idx)?,
                Op::BIntLtImm {
                    target,
                    left,
                    val,
                    is_true,
                } => {
                    self.icmp_imm_br(target, *left, *val, *is_true, Instruction::I64LtS, off2idx)?
                }
                Op::BIntLeImm {
                    target,
                    left,
                    val,
                    is_true,
                } => {
                    self.icmp_imm_br(target, *left, *val, *is_true, Instruction::I64LeS, off2idx)?
                }
                Op::BIntGtImm {
                    target,
                    left,
                    val,
                    is_true,
                } => {
                    self.icmp_imm_br(target, *left, *val, *is_true, Instruction::I64GtS, off2idx)?
                }
                Op::BIntGeImm {
                    target,
                    left,
                    val,
                    is_true,
                } => {
                    self.icmp_imm_br(target, *left, *val, *is_true, Instruction::I64GeS, off2idx)?
                }
                Op::BIntEqImm {
                    target,
                    left,
                    val,
                    is_true,
                } => {
                    self.icmp_imm_br(target, *left, *val, *is_true, Instruction::I64Eq, off2idx)?
                }
                Op::BIntNeImm {
                    target,
                    left,
                    val,
                    is_true,
                } => {
                    self.icmp_imm_br(target, *left, *val, *is_true, Instruction::I64Ne, off2idx)?
                }
                Op::BFloatLt {
                    target,
                    left,
                    right,
                    is_true,
                } => self.fcmp_br(target, *left, *right, *is_true, Instruction::F64Lt, off2idx)?,
                Op::BFloatLe {
                    target,
                    left,
                    right,
                    is_true,
                } => self.fcmp_br(target, *left, *right, *is_true, Instruction::F64Le, off2idx)?,
                Op::BFloatGt {
                    target,
                    left,
                    right,
                    is_true,
                } => self.fcmp_br(target, *left, *right, *is_true, Instruction::F64Gt, off2idx)?,
                Op::BFloatGe {
                    target,
                    left,
                    right,
                    is_true,
                } => self.fcmp_br(target, *left, *right, *is_true, Instruction::F64Ge, off2idx)?,
                Op::BFloatEq {
                    target,
                    left,
                    right,
                    is_true,
                } => self.fcmp_br(target, *left, *right, *is_true, Instruction::F64Eq, off2idx)?,
                Op::BFloatNe {
                    target,
                    left,
                    right,
                    is_true,
                } => self.fcmp_br(target, *left, *right, *is_true, Instruction::F64Ne, off2idx)?,
                Op::BFloatLtImm {
                    target,
                    left,
                    val,
                    is_true,
                } => {
                    self.fcmp_imm_br(target, *left, *val, *is_true, Instruction::F64Lt, off2idx)?
                }
                Op::BFloatLeImm {
                    target,
                    left,
                    val,
                    is_true,
                } => {
                    self.fcmp_imm_br(target, *left, *val, *is_true, Instruction::F64Le, off2idx)?
                }
                Op::BFloatGtImm {
                    target,
                    left,
                    val,
                    is_true,
                } => {
                    self.fcmp_imm_br(target, *left, *val, *is_true, Instruction::F64Gt, off2idx)?
                }
                Op::BFloatGeImm {
                    target,
                    left,
                    val,
                    is_true,
                } => {
                    self.fcmp_imm_br(target, *left, *val, *is_true, Instruction::F64Ge, off2idx)?
                }
                Op::BFloatEqImm {
                    target,
                    left,
                    val,
                    is_true,
                } => {
                    self.fcmp_imm_br(target, *left, *val, *is_true, Instruction::F64Eq, off2idx)?
                }
                Op::BFloatNeImm {
                    target,
                    left,
                    val,
                    is_true,
                } => {
                    self.fcmp_imm_br(target, *left, *val, *is_true, Instruction::F64Ne, off2idx)?
                }
                Op::CallDirect { dst, body, args } => {
                    self.emit_call(*dst, body.index(), args)?;
                }
                Op::Call { dst, args, .. } => {
                    let b = *self
                        .callee
                        .get(&i)
                        .ok_or("dynamic Call to non-const callee")?;
                    self.emit_call(*dst, b, args)?;
                }
                Op::CallNative { dst, id, args } => {
                    let mut params = Vec::with_capacity(args.len());
                    for a in args {
                        let k = self.k(*a)?;
                        params.push(k);
                        self.ins(Instruction::LocalGet(self.li(*a)));
                    }
                    let dk = self.class.get(&(dst.index() as u32)).copied();
                    let ret = dk.unwrap_or(K::Int);
                    let fi = self.natives[&(id.index() as u32, params, Some(ret))];
                    self.ins(Instruction::Call(fi));
                    match dk {
                        Some(_) => self.set(*dst),
                        None => {
                            self.ins(Instruction::Drop);
                        }
                    }
                }
                Op::LoadBody { .. } => {}
                Op::Return { val } => {
                    if let Some(k) = self.ret_k {
                        self.get(*val, k)?;
                    }
                    self.ins(Instruction::Return);
                }
                other => bail!("unsupported op {other:?}"),
            }
        }
        while self.stack.pop().is_some() {
            self.ins(Instruction::End);
        }
        self.ins(Instruction::End); // function end
        Ok(())
    }

    fn emit_call(&mut self, dst: Reg, b: usize, args: &[Reg]) -> Result<(), Bail> {
        let sig = self.sigs[b]
            .as_ref()
            .ok_or_else(|| format!("callee body {b} not emitted"))?;
        let dk = self.class.get(&(dst.index() as u32)).copied();
        let mut void = false;
        if let Some(k) = dk {
            let ret = sig.ret.ok_or("callee returns void but dst is read")?;
            if k != ret {
                bail!("call dst class != callee ret");
            }
        } else {
            void = sig.ret.is_none();
        }
        if sig.params.len() != args.len() {
            bail!("arity mismatch calling body {b}");
        }
        let params = sig.params.clone();
        for (a, pk) in args.iter().zip(&params) {
            self.get(*a, *pk)?;
        }
        let fi = self.func_map[&b];
        self.ins(Instruction::Call(fi));
        match dk {
            Some(_) => self.set(dst),
            None if !void => {
                self.ins(Instruction::Drop);
            }
            _ => {}
        }
        Ok(())
    }

    fn icmp_br(
        &mut self,
        t: &BlockTarget,
        l: Reg,
        r: Reg,
        is_true: bool,
        i: Instruction,
        off2idx: &HashMap<usize, usize>,
    ) -> Result<(), Bail> {
        self.cmp(l, r, K::Int, i)?;
        self.br_if(t, is_true, off2idx)
    }
    fn icmp_imm_br(
        &mut self,
        t: &BlockTarget,
        l: Reg,
        v: i64,
        is_true: bool,
        i: Instruction,
        off2idx: &HashMap<usize, usize>,
    ) -> Result<(), Bail> {
        self.cmp_imm(l, v, K::Int, i)?;
        self.br_if(t, is_true, off2idx)
    }
    fn fcmp_br(
        &mut self,
        t: &BlockTarget,
        l: Reg,
        r: Reg,
        is_true: bool,
        i: Instruction,
        off2idx: &HashMap<usize, usize>,
    ) -> Result<(), Bail> {
        self.cmp(l, r, K::Float, i)?;
        self.br_if(t, is_true, off2idx)
    }
    fn fcmp_imm_br(
        &mut self,
        t: &BlockTarget,
        l: Reg,
        v: i64,
        is_true: bool,
        i: Instruction,
        off2idx: &HashMap<usize, usize>,
    ) -> Result<(), Bail> {
        self.cmp_imm(l, v, K::Float, i)?;
        self.br_if(t, is_true, off2idx)
    }
}

/// Emit the module. Skipped bodies are reported, not fatal — the caller keeps
/// them on the interpreter lane.
/// Instrumentation knobs for the structured lane.
#[derive(Default, Clone, Copy)]
pub struct Opts {
    /// Decrement imported mutable global `env.__fuel` per op; trap when it
    /// goes negative.
    pub fuel: bool,
    /// Trap when imported mutable global `env.__pause` is nonzero, checked at
    /// every loop back-edge. (Structured lane can't suspend mid-function —
    /// this halts, it doesn't resume.)
    pub pause: bool,
    /// Write one byte per executed op into exported `memory` at op index.
    pub coverage: bool,
}

/// Passthrough-native classification for [`irgen::emit_ir`]'s
/// linear-memory sink: the named natives record `(id, arg words)` into an
/// exported buffer instead of crossing the wasm import boundary, and the
/// host replays them in order. Return semantics are fixed per kind — the
/// sink is only for natives whose result the wasm side can produce itself.
#[derive(Clone, Copy)]
pub enum SinkKind {
    /// No result — `cov::hit(p)`.
    Hit,
    /// Result is the last argument, bit-preserved —
    /// `lhs`/`rhs`/`cmp`/`cond`/`dec` all follow this shape.
    Passthru,
    /// Result is constant 1 — `cov::begin(d)`'s always-true gate.
    Begin,
    /// Point-marker only: `cov::hit(p)` — writes `u8[ptmap+p]=1`, no record,
    /// no result. Point hits are ~70% of sink traffic; a bitmap drains them
    /// O(new-points) instead of O(hits).
    Point,
    /// Point-marker + passthrough: `cov::pass(p, v)` — marks `ptmap[p]` and
    /// returns `v` bit-preserved.
    PointPass,
    /// Operand push: `cov::lhs`/`cov::rhs` — pushes the arg onto the
    /// wasm-side operand stack (f64 + numeric flag), returns it bit-preserved.
    Leaf,
    /// Non-comparison leaf: `cov::cond(d, k, v)` — sets the leaf bit and
    /// the flag-side dist cell, returns `v`.
    Cond,
    /// Comparison leaf: `cov::cmp(d, k, op, v)` — pops two operands,
    /// computes the gap/near into the dist cells, sets the leaf bit,
    /// returns `v`.
    Cmp,
    /// Decision outcome: `cov::dec(d, v)` — writes one sink record
    /// `(d, v, leafbits[d])` for the row item, returns `v`.
    Dec,
}

/// native id -> sink kind; see [`SinkKind`].
pub type CovSink = std::collections::HashMap<u32, SinkKind>;

/// Pure math natives that map 1:1 onto wasm float ops — emitted inline
/// instead of as `env` imports (a JS boundary call is ~100x an f64.abs).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MathOp {
    /// `Float::abs` (f64 -> f64)
    Abs,
    /// `Float::min` (f64 f64 -> f64)
    Min,
    /// `Float::max` (f64 f64 -> f64)
    Max,
    /// `Float::floor` (f64 -> f64)
    Floor,
    /// identity passthrough — `Float::to` (units are advisory; `x.to(m) = x`)
    Identity,
}

/// native id -> inline math op, keyed by `NativeId.index()`.
pub type MathNatives = std::collections::HashMap<u32, MathOp>;

pub fn emit(program: &Program) -> Result<Wasmgen, Bail> {
    emit_opts(program, &Opts::default())
}

pub fn emit_opts(program: &Program, opts: &Opts) -> Result<Wasmgen, Bail> {
    let nbodies = program.chunks.len();
    let bodies_ops: Vec<Vec<(usize, Op)>> = (0..nbodies)
        .map(|b| program.ops(compile::BodyId::from(b as u32)))
        .collect();

    // cross-body fixpoint on return classes (call dsts class off callee rets)
    let mut ret: Vec<Option<K>> = vec![None; nbodies];
    let mut anas: Vec<Option<Ana>> = (0..nbodies).map(|_| None).collect();
    for _ in 0..16 {
        let mut stable = true;
        for b in 0..nbodies {
            let nregs = program.chunks[compile::BodyId::from(b as u32)].regs as u32;
            match analyze(&bodies_ops[b], nregs, &ret) {
                Ok(a) => {
                    if a.ret != ret[b] {
                        ret[b] = a.ret;
                        stable = false;
                    }
                    anas[b] = Some(a);
                }
                Err(_) => {}
            }
        }
        if stable {
            break;
        }
    }

    let mut skipped: Vec<Skip> = Vec::new();
    let mut sigs: Vec<Option<Sig>> = (0..nbodies).map(|_| None).collect();
    for b in 0..nbodies {
        let chunk = &program.chunks[compile::BodyId::from(b as u32)];
        let Some(ana) = &anas[b] else {
            skipped.push(Skip {
                body: b,
                reason: "analysis failed".into(),
            });
            continue;
        };
        let mut params = Vec::new();
        let mut ok = true;
        for p in &chunk.params {
            match ana.class.get(&(p.index() as u32)) {
                Some(&k) => params.push(k),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            skipped.push(Skip {
                body: b,
                reason: "param has no scalar class".into(),
            });
            continue;
        }
        sigs[b] = Some(Sig {
            params,
            ret: ana.ret,
        });
    }

    // native call sites across all sig'd bodies — the trial-emit map
    let mut trial_natives: HashMap<(u32, Vec<K>, Option<K>), u32> = HashMap::new();
    for b in 0..nbodies {
        let Some(ana) = &anas[b] else { continue };
        for (_, op) in &bodies_ops[b] {
            if let Op::CallNative { dst, id, args } = op {
                let params: Vec<K> = args
                    .iter()
                    .filter_map(|a| ana.class.get(&(a.index() as u32)).copied())
                    .collect();
                let ret = Some(
                    ana.class
                        .get(&(dst.index() as u32))
                        .copied()
                        .unwrap_or(K::Int),
                );
                trial_natives
                    .entry((id.index() as u32, params, ret))
                    .or_insert(0);
            }
        }
    }
    let dummy_func_map: HashMap<usize, u32> = (0..nbodies).map(|b| (b, b as u32)).collect();

    // settle the emitted set: a body survives iff its callees are emitted, its
    // scopes nest, and a trial emission succeeds — emit failures are skips,
    // not fatal (the caller keeps those bodies on the interpreter lane)
    let mut emitted: Vec<bool> = sigs.iter().map(|s| s.is_some()).collect();
    let mut reasons: Vec<String> = (0..nbodies).map(|_| String::new()).collect();
    loop {
        let mut changed = false;
        for b in 0..nbodies {
            if !emitted[b] {
                continue;
            }
            let ana = anas[b].as_ref().unwrap();
            let native_ok = bodies_ops[b].iter().all(|(_, op)| match op {
                Op::CallNative { args, .. } => args
                    .iter()
                    .all(|a| ana.class.contains_key(&(a.index() as u32))),
                _ => true,
            });
            let mut why = String::new();
            if let Some(c) = ana.callee.values().find(|c| !emitted[**c]) {
                why = format!("calls skipped body {c}");
            } else if !native_ok {
                why = "native arg has no scalar class".into();
            } else if let Err(e) = try_body(
                b,
                &bodies_ops[b],
                &program,
                ana,
                &sigs,
                &dummy_func_map,
                &trial_natives,
                &ProbeGlobals::default(),
            ) {
                why = e;
            }
            if !why.is_empty() {
                emitted[b] = false;
                reasons[b] = why;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for b in 0..nbodies {
        if sigs[b].is_some() && !emitted[b] {
            skipped.push(Skip {
                body: b,
                reason: reasons[b].clone(),
            });
            sigs[b] = None;
        }
    }

    // native imports: (id, params, ret) -> func idx; imports precede defined fns
    let mut natives: HashMap<(u32, Vec<K>, Option<K>), u32> = HashMap::new();
    let mut native_list: Vec<(u32, Vec<K>, Option<K>)> = Vec::new();
    for b in 0..nbodies {
        if sigs[b].is_none() {
            continue;
        }
        let ana = anas[b].as_ref().unwrap();
        for (_, op) in &bodies_ops[b] {
            if let Op::CallNative { dst, id, args } = op {
                let params: Vec<K> = args
                    .iter()
                    .map(|a| ana.class[&(a.index() as u32)])
                    .collect();
                let ret = Some(
                    ana.class
                        .get(&(dst.index() as u32))
                        .copied()
                        .unwrap_or(K::Int),
                );
                let key = (id.index() as u32, params, ret);
                if !natives.contains_key(&key) {
                    let fi = native_list.len() as u32;
                    natives.insert(key.clone(), fi);
                    native_list.push(key);
                }
            }
        }
    }
    let nimports = native_list.len() as u32;

    let mut func_map: HashMap<usize, u32> = HashMap::new();
    for (fi, b) in emitted
        .iter()
        .enumerate()
        .filter(|(_, e)| **e)
        .map(|(i, _)| i)
        .enumerate()
    {
        func_map.insert(b, nimports + fi as u32);
    }

    let mut types = TypeSection::new();
    let mut imports = ImportSection::new();
    let mut funcs = FunctionSection::new();
    let mut code = CodeSection::new();
    let mut exports = ExportSection::new();
    let mut names = NameMap::new();
    let mut out_bodies: Vec<Body> = Vec::new();
    let mut type_ids: HashMap<(Vec<K>, Option<K>), u32> = HashMap::new();
    let mut srcmaps: Vec<(u32, Vec<(u32, u32)>)> = Vec::new();
    let mut locs: Vec<(u32, u32, u32)> = Vec::new();
    let mut loc_map: HashMap<(u32, u32, u32), u32> = HashMap::new();
    let mut cov_next = 0u32;

    // import types + entries first so natives occupy the leading func indices
    for (id, params, ret) in &native_list {
        let key = (params.clone(), *ret);
        let ntypes = types.len();
        let ty = *type_ids.entry(key).or_insert(ntypes);
        if ty == ntypes {
            types.ty().function(
                params.iter().map(|k| k.val_type()),
                ret.map(|k| k.val_type()),
            );
        }
        let sig_desc = params
            .iter()
            .map(|k| match k {
                K::Int => 'i',
                K::Float => 'f',
                K::Bool => 'b',
                K::Word => 'i',
            })
            .collect::<String>();
        imports.import(
            "env",
            &format!("n{id}_{sig_desc}"),
            EntityType::Function(ty),
        );
    }

    // instrumentation imports — globals live in their own index space
    let mut gnext = 0u32;
    let mut import_global = |module: &mut ImportSection, name, vt| {
        let g = gnext;
        gnext += 1;
        module.import(
            "env",
            name,
            EntityType::Global(GlobalType {
                val_type: vt,
                mutable: true,
                shared: false,
            }),
        );
        g
    };
    let fuel_g = opts.fuel.then(|| import_global(&mut imports, "__fuel", ValType::I64));
    let pause_g = opts.pause.then(|| import_global(&mut imports, "__pause", ValType::I32));
    let mut globs = ProbeGlobals {
        fuel_g,
        pause_g,
        cov_base: u32::MAX,
    };
    let nimport_entries = imports.len();

    for b in 0..nbodies {
        let Some(sig) = &sigs[b] else { continue };
        let ana = anas[b].as_ref().unwrap();
        let key = (sig.params.clone(), sig.ret);
        let ntypes = types.len();
        let ty = *type_ids.entry(key).or_insert(ntypes);
        if ty == ntypes {
            types.ty().function(
                sig.params.iter().map(|k| k.val_type()),
                sig.ret.map(|k| k.val_type()),
            );
        }
        funcs.function(ty);
        if opts.coverage {
            globs.cov_base = cov_next;
            cov_next += bodies_ops[b].len() as u32;
        }
        let (f, sm) = try_body(b, &bodies_ops[b], &program, ana, &sigs, &func_map, &natives, &globs)
            .map_err(|e| format!("body {b} passed trial but failed emit: {e}"))?;
        let chunk = &program.chunks[compile::BodyId::from(b as u32)];
        let sm_loc: Vec<(u32, u32)> = sm
            .iter()
            .map(|&(off, i)| {
                let (boff, _) = bodies_ops[b][i as usize];
                let loc = chunk.loc_at((boff - chunk.offset) as u32);
                let key = (
                    loc.file_id as u32,
                    loc.span.start as u32,
                    loc.span.end as u32,
                );
                let li = *loc_map.entry(key).or_insert_with(|| {
                    locs.push(key);
                    (locs.len() - 1) as u32
                });
                (off, li)
            })
            .collect();
        srcmaps.push((func_map[&b], sm_loc));
        let fidx = func_map[&b];
        code.function(&f);
        let name = format!("b{b}");
        exports.export(&name, ExportKind::Func, fidx);
        names.append(fidx, &name);
        out_bodies.push(Body {
            body: b,
            func: fidx,
            name,
        });
    }

    let mut module = Module::new();
    module.section(&types);
    if nimport_entries > 0 {
        module.section(&imports);
    }
    module.section(&funcs);
    if opts.coverage {
        let pages = (cov_next as u64 + 65535) / 65536;
        let mut mems = MemorySection::new();
        mems.memory(MemoryType {
            minimum: pages.max(1),
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&mems);
        exports.export("memory", ExportKind::Memory, 0);
    }
    module.section(&exports);
    module.section(&code);
    let mut ns = NameSection::new();
    ns.functions(&names);
    module.section(&ns);
    if !srcmaps.is_empty() {
        // mimas.srcmap: magic, per-func (func_idx, [(code_off, loc_idx)]),
        // then the shared loc table of (file_id, span_lo, span_hi)
        let mut d = Vec::new();
        d.extend_from_slice(&0x4D534D31u32.to_le_bytes());
        d.extend_from_slice(&(srcmaps.len() as u32).to_le_bytes());
        for (fi, entries) in &srcmaps {
            d.extend_from_slice(&fi.to_le_bytes());
            d.extend_from_slice(&(entries.len() as u32).to_le_bytes());
            for (off, li) in entries {
                d.extend_from_slice(&off.to_le_bytes());
                d.extend_from_slice(&li.to_le_bytes());
            }
        }
        d.extend_from_slice(&(locs.len() as u32).to_le_bytes());
        for (f, lo, hi) in &locs {
            d.extend_from_slice(&f.to_le_bytes());
            d.extend_from_slice(&lo.to_le_bytes());
            d.extend_from_slice(&hi.to_le_bytes());
        }
        module.section(&CustomSection {
            name: std::borrow::Cow::Borrowed("mimas.srcmap"),
            data: std::borrow::Cow::Borrowed(&d),
        });
    }
    Ok(Wasmgen {
        bytes: module.finish(),
        bodies: out_bodies,
        skipped,
    })
}

/// Trial/final emission of one body into a fresh `Function`.
/// Module-level indices try_body needs for instrumentation.
#[derive(Default)]
pub(crate) struct ProbeGlobals {
    pub fuel_g: Option<u32>,
    pub pause_g: Option<u32>,
    /// coverage byte base for this body's op 0 (u32::MAX = off)
    pub cov_base: u32,
}

fn try_body(
    b: usize,
    ops: &[(usize, Op)],
    program: &Program,
    ana: &Ana,
    sigs: &[Option<Sig>],
    func_map: &HashMap<usize, u32>,
    natives: &HashMap<(u32, Vec<K>, Option<K>), u32>,
    globs: &ProbeGlobals,
) -> Result<(Function, Vec<(u32, u32)>), Bail> {
    let chunk = &program.chunks[compile::BodyId::from(b as u32)];
    let Some(sig) = &sigs[b] else {
        bail!("no sig for body {b}");
    };
    // params occupy locals 0..arity in order; classed regs follow
    let mut local: HashMap<u32, u32> = HashMap::new();
    for (i, p) in chunk.params.iter().enumerate() {
        local.insert(p.index() as u32, i as u32);
    }
    let mut groups: Vec<(u32, ValType)> = Vec::new();
    let mut next = chunk.params.len() as u32;
    let mut regs_sorted: Vec<u32> = ana.class.keys().copied().collect();
    regs_sorted.sort();
    for r in regs_sorted {
        if local.contains_key(&r) {
            continue;
        }
        let k = ana.class[&r];
        local.insert(r, next);
        next += 1;
        if let Some(g) = groups.last_mut() {
            if g.1 == k.val_type() {
                g.0 += 1;
                continue;
            }
        }
        groups.push((1, k.val_type()));
    }
    let tmp = next;
    groups.push((1, ValType::I64));

    let sc = scopes(ops, ops.len())?;
    let mut em = Em {
        ops,
        class: &ana.class,
        local: &local,
        sigs,
        func_map,
        callee: &ana.callee,
        natives,
        ret_k: sig.ret,
        tmp,
        stack: Vec::new(),
        f: Function::new(groups),
        code_off: 0,
        srcmap: Vec::new(),
        fuel_g: globs.fuel_g,
        pause_g: globs.pause_g,
        cov_base: globs.cov_base,
    };
    let mut off2idx = HashMap::new();
    for (i, (off, _)) in ops.iter().enumerate() {
        off2idx.insert(*off, i);
    }
    em.emit(&off2idx, sc)?;
    Ok((em.f, em.srcmap))
}

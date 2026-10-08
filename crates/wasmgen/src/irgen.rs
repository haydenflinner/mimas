//! `emit_ir` — the structured wasm lane fed on `compile::Ir` SSA bodies
//! instead of flattened `Program` bytecode.
//!
//! Two structural wins over the `Op`-level emitter in `lib.rs`:
//! - classes live on SSA `InstId`s, not bytecode `Reg`s — a register written
//!   Int in one arm and Float in another used to union to an unusable mask
//!   and skip the whole body; the two `InstId`s each get their own class.
//! - call typing resolves through def-use (`RefBody` producers, single-write
//!   locals, phis) instead of "register written exactly once by `LoadBody`".
//!
//! Control flow runs through the same interval-scope machinery as the
//! bytecode lane (`scopes_ir` positions are whole `Block`s rather than byte
//! offsets), with phis lowered to `local.set` copies at predecessor tails.

use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;

use compile::{
    BinOp, BlockId, Body as IrBody, Constant, FormatPart, Inst, InstId, Ir, OperandKind, UnaryOp,
};
use shared::{StrId, StrInterner};
use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, CustomSection, DataSection, ElementSection, Elements,
    Encode, EntityType, ExportKind, ExportSection, Function, FunctionSection, GlobalSection,
    GlobalType, ImportSection, Instruction, MemArg, MemorySection, MemoryType, Module, NameMap,
    NameSection, RefType, TableSection, TableType, TypeSection, ValType,
};

use crate::{
    Bail, Body, K, K_BOOL, K_FLOAT, K_INT, Opts, ProbeGlobals, Repair, Scope, ScopeKind, Sig, Skip,
    Wasmgen, kbit, simulate,
};

macro_rules! bail {
    ($($a:tt)*) => { return Err(format!($($a)*)) };
}

fn mem_arg(offset: u32, align: u32) -> MemArg {
    MemArg {
        offset: offset as u64,
        align,
        memory_index: 0,
    }
}

// ---------- analysis ----------

/// Demand/produce bit for "the raw 8-byte word" — heap handles, strids,
/// payloads. Unlike the scalar bits it is not a *class*: every materialized
/// value can serve a word demand, and a word slot can serve a scalar read by
/// reinterpretation (sound only where the solver's types agree — a mismatched
/// read is dead code on any valid path, same convention as `resume.rs`).
const K_WORD: u8 = 8;

// Heap-object tags — the leading u32 of every __hp object, so a host
// walking linear memory can self-describe everything (the "one tag word
// per object" contract). Keep < 0x10 so a tag reads unlike any pointer.
pub(crate) const TAG_ARRAY: u32 = 1;
pub(crate) const TAG_STR: u32 = 2;
pub(crate) const TAG_DICT: u32 = 3;
pub(crate) const TAG_CLOSURE: u32 = 4;
pub(crate) const TAG_INSTANCE: u32 = 5;
/// untagged-by-language internal regions (dict buckets) still carry a tag
/// word so a sequential __hp walk stays parseable
pub(crate) const TAG_RAW: u32 = 0xff;

// internal helper functions appended after the emitted bodies; names index
// `helpers` maps so bodies can `Call` them during emission.
const H_ALLOC: &str = "__alloc";
const H_IS_STR: &str = "__is_str";
const H_STR_EQ: &str = "__str_eq";
const H_STR_CAT: &str = "__str_cat";
const H_STR_CMP: &str = "__str_cmp";
const H_I64_STR: &str = "__i64_str";
const H_F64_STR: &str = "__f64_str";
const H_STR_CHAR: &str = "__str_charat";
const H_STR_IN: &str = "__str_in";
const H_STR_OR_OBJ: &str = "__str_or_obj";
const H_KEY_EQ: &str = "__key_eq";
const H_KEY_HASH: &str = "__key_hash";
const H_DICT_FIND: &str = "__dict_find";
const H_DICT_SET: &str = "__dict_set";
const H_DICT_ENTRY: &str = "__dict_entry";
pub(crate) const HELPER_NAMES: &[&str] = &[
    H_ALLOC,
    H_IS_STR,
    H_STR_EQ,
    H_STR_CAT,
    H_STR_CMP,
    H_I64_STR,
    H_F64_STR,
    H_STR_CHAR,
    H_STR_IN,
    H_STR_OR_OBJ,
    H_KEY_EQ,
    H_KEY_HASH,
    H_DICT_FIND,
    H_DICT_SET,
    H_DICT_ENTRY,
];

// ---- Go-Explore coverage sink (the resumable lane's contract) ----
// cov::* natives don't cross the import boundary — they record into
// linear-memory regions the host drains per frame (see `SinkKind`).
// Fixed layout so `wgame`'s snapshot ranges stay valid:
//   [0, nb*8)            pc_table — reserved zeros (no suspension here)
//   [nb*8, +entry_bytes) entry locals — under __sp so snapshots image
//                        module-level `let` state and seeds scan them
//   [__sp, __sp+1MB)     hole — static str objects + call/i64 scratch
//   [__sp+1MB, ...)      coverage: [op bitmap][ptmap][dec cells][opstk]
//   [__covbuf, +8MB)     cov::dec record ring
//   [__hp, ...)          bump arena
pub(crate) const STACK_CAP: u32 = 1 << 20;
pub(crate) const SINK_CAP: u32 = 8 << 20;
/// decision cell stride: [leafbits u64][dist_t 8×f64 @8][dist_f 8×f64 @72]
pub(crate) const DCELL: u32 = 136;
/// cmp operand-stack entries (16B each: f64 + u8 numeric flag)
pub(crate) const OPSTK_N: u32 = 64;
const F64_INF: f64 = f64::INFINITY;
const EPS: f64 = 1e-9;

const H_COV_LEAF: &str = "__cov_leaf";
const H_COV_BEGIN: &str = "__cov_begin";
const H_COV_COND: &str = "__cov_cond";
const H_COV_CMP: &str = "__cov_cmp";
const H_COV_DEC: &str = "__cov_dec";
const H_COV_GAP: &str = "__cov_gap";
const H_COV_NEAR: &str = "__cov_cellnear";
const H_COV_BIT: &str = "__cov_leafbit";
pub(crate) const COV_HELPER_NAMES: &[&str] = &[
    H_COV_LEAF,
    H_COV_BEGIN,
    H_COV_COND,
    H_COV_CMP,
    H_COV_DEC,
    H_COV_GAP,
    H_COV_NEAR,
    H_COV_BIT,
];

/// Everything the coverage-sink arms need inside a body: memory bases are
/// compile-time constants; the globals and helper funcs are module indices
/// settled after the emit set stabilizes.
pub(crate) struct CovCtx {
    /// `u8[ptmap_base + point]` — hit/pass write one byte, no record
    pub ptmap_base: u32,
    /// decision cells base (`d * DCELL` strides)
    pub decv_base: u32,
    /// cmp operand stack base (16B slots)
    pub opstk_base: u32,
    /// `cov::dec` record ring base (16B entries)
    pub sink_base: u32,
    /// `__covp` global — record cursor
    pub g_covp: u32,
    /// `__osp` global — operand-stack depth
    pub g_osp: u32,
    /// func indices of the cov helpers (leaf/begin/cond/cmp/dec/gap/near/bit)
    pub h: [u32; 8],
}

/// static-address / scratch context shared by every body's emission.
pub(crate) struct Statics {
    /// StrId idx -> absolute address of its tagged str object
    pub str_objs: HashMap<u32, u32>,
    pub true_obj: u32,
    pub false_obj: u32,
    pub obj_obj: u32,
    pub null_obj: u32,
    /// i64 scratch: Call marshals args here, one word slot each
    pub call_scratch: u32,
    /// 32B digit buffer for i64->ascii
    pub i64_scratch: u32,
}

pub(crate) struct AnaI {
    /// inst idx -> resolved scalar class — present only when the inst's mask
    /// collapsed to exactly one bit
    pub class: HashMap<u32, K>,
    /// inst idx -> needs a materialized word (i64) slot — unclassed but read
    pub wused: HashSet<u32>,
    /// local idx -> resolved class
    pub lclass: HashMap<u32, K>,
    /// local idx -> has any reader (word or scalar) — such locals get a slot
    pub lused: HashSet<u32>,
    /// inst idx -> callee body (resolved `Call` + every `CallDirect`)
    pub callee: HashMap<u32, usize>,
    /// resolved return class (None = void)
    pub ret: Option<K>,
}

/// The class a `BinOp` produces given its operand kind — `None` = can't emit.
pub(crate) fn binop_prod(op: BinOp, kind: OperandKind) -> Option<K> {
    use BinOp::*;
    Some(match kind {
        OperandKind::Int => match op {
            Add | Sub | Mult | IDiv | Mod | BitAnd | BitOr | BitXor | BitShiftLeft
            | BitShiftRight => K::Int,
            Div => K::Float,
            Identity | NotEqual | LessThan | LessEqual | GreaterThan | GreaterEqual => K::Bool,
            _ => return None,
        },
        OperandKind::Float => match op {
            Add | Sub | Mult | Div | IDiv | Mod => K::Float,
            Identity | NotEqual | LessThan | LessEqual | GreaterThan | GreaterEqual => K::Bool,
            _ => return None,
        },
        OperandKind::Bool => match op {
            Identity | NotEqual | Xor | And | Or => K::Bool,
            _ => return None,
        },
        // strs are tagged objects: `+` concatenates into a fresh word,
        // compares run byte-wise through `__str_eq`/`__str_cmp`
        OperandKind::Str => match op {
            Add => K::Word,
            Identity | NotEqual | LessThan | LessEqual | GreaterThan | GreaterEqual => K::Bool,
            _ => return None,
        },
        // `==`/`!=` on untyped words is identity — heap words compare by
        // pointer, strs by the same pointer trick or `__key_eq` at callsites
        OperandKind::Generic => match op {
            Identity | NotEqual => K::Bool,
            _ => return None,
        },
    })
}

/// Resolve a `Call`'s callee inst to a static body through the obvious
/// producers: `RefBody`, a `GetLocal` whose local has exactly one write, or a
/// phi whose every incoming resolves the same way.
fn resolve_callee(
    body: &IrBody,
    writes: &HashMap<u32, Vec<u32>>,
    iid: InstId,
    depth: u8,
) -> Option<usize> {
    if depth == 0 {
        return None;
    }
    match &body.instructions[iid] {
        Inst::RefBody(b) => Some(b.index()),
        Inst::GetLocal(l) => {
            let ws = writes.get(&(l.index() as u32))?;
            if ws.len() == 1 {
                resolve_callee(body, writes, InstId::from(ws[0]), depth - 1)
            } else {
                None
            }
        }
        Inst::Unwrap(v) => resolve_callee(body, writes, *v, depth - 1),
        Inst::Phi(branches) => {
            let mut it = branches.iter();
            let first = resolve_callee(body, writes, it.next()?.1, depth - 1)?;
            if it.all(|(_, v)| resolve_callee(body, writes, *v, depth - 1) == Some(first)) {
                Some(first)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Classify one body's insts. `ret[c]` is the current best guess of body `c`'s
/// return class — iterated to a fixpoint across the program.
pub(crate) fn analyze_body(body: &IrBody, ret: &[Option<K>]) -> Result<AnaI, Bail> {
    let n = body.instructions.len();
    let nl = body.locals.len();
    // classes the inst's readers demand of it
    let mut need: Vec<u8> = vec![0; n];
    // classes reads of local `l` demand
    let mut lneed: Vec<u8> = vec![0; nl];
    // local -> value insts stored into it
    let mut writes: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut callee: HashMap<u32, usize> = HashMap::new();

    // local writes collected up front — `resolve_callee` follows
    // single-write locals and needs the complete picture, not just the
    // writes that happen to precede the call site
    for (_, block) in body.blocks.iter() {
        for &iid in &block.stream {
            if let Inst::SetLocal(l, v) = &body.instructions[iid] {
                writes
                    .entry(l.index() as u32)
                    .or_default()
                    .push(v.index() as u32);
            }
        }
    }

    for (_, block) in body.blocks.iter() {
        for &iid in &block.stream {
            match &body.instructions[iid] {
                Inst::BinOp {
                    left, right, kind, ..
                } => {
                    let d = match kind {
                        OperandKind::Int => K_INT,
                        OperandKind::Float => K_FLOAT,
                        OperandKind::Bool => K_BOOL,
                        // str/generic binops read raw words — identity
                        // compares use them directly, dynamic ops read the
                        // word through the guessed scalar class
                        OperandKind::Str | OperandKind::Generic => K_WORD,
                    };
                    need[left.index()] |= d;
                    need[right.index()] |= d;
                }
                Inst::UnaryOp { op, right } => {
                    need[right.index()] |= match op {
                        UnaryOp::Not => K_BOOL,
                        UnaryOp::BitwiseNot => K_INT,
                        // produce is the operand's own class
                        UnaryOp::Negative | UnaryOp::Positive => 0,
                    };
                }
                Inst::JumpIfFalse { condition, .. } => need[condition.index()] |= K_BOOL,
                Inst::Switch { scrut, .. } => need[scrut.index()] |= K_INT,
                Inst::ForNext { idx, bound, .. } => {
                    if let Inst::GetLocal(l) = &body.instructions[*idx] {
                        lneed[l.index()] |= K_INT;
                    }
                    need[bound.index()] |= K_INT;
                }
                Inst::SetLocal(..) => {}
                Inst::ToFloat(v) => need[v.index()] |= K_INT,
                Inst::Sqrt(v) => need[v.index()] |= K_FLOAT,
                // heap ops traffic in raw words — an index is a word too:
                // dict keys are str words, array indices coerce from int
                Inst::GetIndex { set, index, .. } => {
                    need[set.index()] |= K_WORD;
                    need[index.index()] |= K_WORD;
                }
                Inst::SetIndex {
                    set,
                    index,
                    value,
                } => {
                    need[set.index()] |= K_WORD;
                    need[index.index()] |= K_WORD;
                    need[value.index()] |= K_WORD;
                }
                Inst::GetField { src, .. } => need[src.index()] |= K_WORD,
                Inst::SetField {
                    receiver, value, ..
                } => {
                    need[receiver.index()] |= K_WORD;
                    need[value.index()] |= K_WORD;
                }
                Inst::Push { array, value } => {
                    need[array.index()] |= K_WORD;
                    need[value.index()] |= K_WORD;
                }
                Inst::Insert { dict, value, .. } => {
                    need[dict.index()] |= K_WORD;
                    need[value.index()] |= K_WORD;
                }
                Inst::NewInstance { fields, .. } => {
                    for f in fields {
                        need[f.index()] |= K_WORD;
                    }
                }
                Inst::SetEntry(_, v) => need[v.index()] |= K_WORD,
                Inst::Len(v)
                | Inst::Unwrap(v)
                | Inst::UnwrapUnit(v)
                | Inst::UnwrapRaised(v)
                | Inst::IsRaised(v)
                | Inst::Raise(v) => need[v.index()] |= K_WORD,
                Inst::IsInstance { src, .. } => need[src.index()] |= K_WORD,
                Inst::In(a, b, _) => {
                    need[a.index()] |= K_WORD;
                    need[b.index()] |= K_WORD;
                }
                Inst::Format(parts) => {
                    for p in parts {
                        if let FormatPart::Value(v) = p {
                            // "any materialization" — the formatter dispatches
                            // on the value's class, so the demand must not
                            // pin it to Word (an int stays Int → __i64_str)
                            need[v.index()] |= K_INT | K_FLOAT | K_BOOL | K_WORD;
                        }
                    }
                }
                Inst::MakeClosure { captures, .. } => {
                    for c in captures {
                        need[c.index()] |= K_WORD;
                    }
                }
                Inst::Return(v) => need[v.index()] |= K_WORD,
                // call args pass the callee's param class — word demand
                // suffices to materialize them; `get` converts per-param
                Inst::CallDirect { body: b, args } => {
                    callee.insert(iid.index() as u32, b.index());
                    for a in args {
                        need[a.index()] |= K_WORD;
                    }
                }
                Inst::Call { callee: c, args } => {
                    if let Some(b) = resolve_callee(body, &writes, *c, 8) {
                        callee.insert(iid.index() as u32, b);
                    }
                    // a callee word must materialize even when resolved — the
                    // call_indirect path reads it whenever the direct-call
                    // arity check fails (captured bodies)
                    need[c.index()] |= K_WORD;
                    for a in args {
                        need[a.index()] |= K_WORD;
                    }
                }
                Inst::CallNative { args, .. } => {
                    for a in args {
                        need[a.index()] |= K_WORD;
                    }
                }
                _ => {}
            }
        }
    }

    // fixpoint: producer masks flow forward, reader needs flow backward
    // through copy-ish edges (SetLocal / GetLocal / Phi / Unwrap).
    // masks accumulate (`|=`) rather than replace — a self-referential
    // local (`x = -x` in a loop) makes the SetLocal→local→GetLocal→inst
    // cycle a delayed-feedback recurrence with multiple fixpoints, and
    // replace-assignment orbits between them forever. Join semantics
    // converge in ≤4 rounds/element; widening is safe since an all-bits
    // mask already means "serves any repr" (the need==0 convention).
    let mut mask: Vec<u8> = vec![0; n];
    let mut lmask: Vec<u8> = vec![0; nl];
    loop {
        let mut changed = false;
        let snap_need = need.clone();
        let snap_lneed = lneed.clone();
        let snap_mask = mask.clone();
        let snap_lmask = lmask.clone();

        for (_, block) in body.blocks.iter() {
            for &iid in &block.stream {
                let ix = iid.index();
                match &body.instructions[iid] {
                    Inst::SetLocal(l, v) => {
                        let d = snap_lneed[l.index()];
                        if snap_need[v.index()] & d != d {
                            need[v.index()] |= d;
                            changed = true;
                        }
                    }
                    Inst::GetLocal(l) => {
                        let d = snap_need[ix];
                        if snap_lneed[l.index()] & d != d {
                            lneed[l.index()] |= d;
                            changed = true;
                        }
                    }
                    Inst::Phi(branches) => {
                        let d = snap_need[ix];
                        for (_, v) in branches {
                            if snap_need[v.index()] & d != d {
                                need[v.index()] |= d;
                                changed = true;
                            }
                        }
                    }
                    Inst::Unwrap(v) => {
                        let d = snap_need[ix];
                        if snap_need[v.index()] & d != d {
                            need[v.index()] |= d;
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        for l in 0..nl {
            let mut m = match writes.get(&(l as u32)) {
                Some(ws) => ws.iter().fold(0u8, |m, &v| m | snap_mask[v as usize]),
                None => lneed[l],
            };
            // a local with a slot can always serve a word read (raw copy)
            if m != 0 {
                m |= K_WORD;
            }
            if lmask[l] | m != lmask[l] {
                lmask[l] |= m;
                changed = true;
            }
        }

        for (_, block) in body.blocks.iter() {
            for &iid in &block.stream {
                let ix = iid.index();
                let inst = &body.instructions[iid];
                let m = match inst {
                    Inst::Constant(c) => match c {
                        // every materialized const can serve a word read —
                        // emit_const writes the raw repr
                        Constant::Int(_) => K_INT | K_WORD,
                        Constant::Float(_) => K_FLOAT | K_WORD,
                        Constant::Bool(_) => K_BOOL | K_WORD,
                        Constant::Str(_) | Constant::Null => K_WORD,
                        Constant::Array(_) => 0,
                    },
                    Inst::GetLocal(l) => snap_lmask[l.index()],
                    Inst::BinOp { op, kind, .. } => {
                        binop_prod(*op, *kind).map(kbit).unwrap_or(0) | K_WORD
                    }
                    Inst::UnaryOp { op, right } => (match op {
                        UnaryOp::Negative | UnaryOp::Positive => snap_mask[right.index()],
                        UnaryOp::Not => K_BOOL,
                        _ => K_INT,
                    }) | K_WORD,
                    Inst::Phi(branches) => branches
                        .iter()
                        .fold(0u8, |m, (_, v)| m | snap_mask[v.index()]),
                    Inst::Unwrap(v) => snap_mask[v.index()] | K_WORD,
                    Inst::ToFloat(_) | Inst::Sqrt(_) => K_FLOAT | K_WORD,
                    Inst::Len(_) => K_INT | K_WORD,
                    Inst::IsRaised(_) | Inst::IsInstance { .. } | Inst::In(..) => K_BOOL | K_WORD,
                    Inst::NewArray
                    | Inst::NewDict
                    | Inst::GetIndex { .. }
                    | Inst::GetField { .. }
                    | Inst::NewInstance { .. }
                    | Inst::UnwrapUnit(_)
                    | Inst::UnwrapRaised(_)
                    | Inst::GetEntry(_)
                    | Inst::RefBody(_)
                    | Inst::MakeClosure { .. }
                    | Inst::Format(_) => K_WORD,
                    Inst::Call { .. } | Inst::CallDirect { .. } => {
                        match callee.get(&(ix as u32)) {
                            Some(&c) => match ret[c] {
                                Some(k) => kbit(k) | K_WORD,
                                // callee ret unconstrained — the dst's readers
                                // pick the width (same as W::Call)
                                None => {
                                    if snap_need[ix] == 0 {
                                        K_INT | K_FLOAT | K_BOOL | K_WORD
                                    } else {
                                        snap_need[ix]
                                    }
                                }
                            },
                            // an unresolved Call goes through call_indirect —
                            // its trampoline returns a raw i64 word
                            None => {
                                if matches!(inst, Inst::Call { .. }) {
                                    K_WORD
                                } else {
                                    0
                                }
                            }
                        }
                    }
                    Inst::CallNative { .. } => {
                        if snap_need[ix] == 0 {
                            K_INT | K_FLOAT | K_BOOL | K_WORD
                        } else {
                            snap_need[ix]
                        }
                    }
                    _ => 0,
                };
                // no partial-demand zeroing: a word-materialized value can
                // serve any repr via coerce, so producible is producible —
                // only mask = 0 (can't materialize at all) fails a reader
                if mask[ix] | m != mask[ix] {
                    mask[ix] |= m;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // a mask's scalar class: the single non-word bit, if any
    let scalar = |m: u8| match m & !K_WORD {
        K_INT => Some(K::Int),
        K_FLOAT => Some(K::Float),
        K_BOOL => Some(K::Bool),
        _ => None,
    };
    let class: HashMap<u32, K> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| scalar(m).map(|k| (i as u32, k)))
        .collect();
    // demanded but not scalar-classed → materialize the raw word. `m != 0`
    // keeps unproducible insts (RefBody et al) out of the slot map — their
    // readers bail in `get` instead of silently reading a dead local
    let wused: HashSet<u32> = mask
        .iter()
        .enumerate()
        .filter(|&(i, &m)| scalar(m).is_none() && m != 0 && need[i] != 0)
        .map(|(i, _)| i as u32)
        .collect();
    let lclass: HashMap<u32, K> = lmask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| scalar(m).map(|k| (i as u32, k)))
        .collect();
    let lused: HashSet<u32> = lneed
        .iter()
        .enumerate()
        .filter(|&(_, &d)| d != 0)
        .map(|(i, _)| i as u32)
        .collect();

    // the return class every Return agrees on
    let mut cand = 7u8;
    let mut has_ret = false;
    for (_, block) in body.blocks.iter() {
        for &iid in &block.stream {
            if let Inst::Return(v) = &body.instructions[iid] {
                has_ret = true;
                cand &= mask[v.index()];
            }
        }
    }
    let ret_k = match cand {
        K_INT => Some(K::Int),
        K_FLOAT => Some(K::Float),
        K_BOOL => Some(K::Bool),
        _ if has_ret => Some(K::Word),
        _ => None,
    };
    Ok(AnaI {
        class,
        wused,
        lclass,
        lused,
        callee,
        ret: ret_k,
    })
}

// ---------- block order + scopes ----------

/// (pred block, original target block) -> retargeted target. The layout
/// repair installs these to tail-duplicate a latch-merge block: the pred's
/// edge runs the dup'd block's insts inline then jumps to its successor —
/// the merge scope that could not nest disappears.
type Retargets = HashMap<(u32, u32), u32>;

fn rtgt(rt: &Retargets, p: BlockId, t: BlockId) -> BlockId {
    rt.get(&(p.index() as u32, t.index() as u32))
        .map(|&n| BlockId::from(n))
        .unwrap_or(t)
}

/// Successor blocks of `bid`: (`Jump` target, conditional targets). Mirrors
/// the codegen DFS so layout order (and which jumps are fallthroughs) matches
/// the bytecode the interpreter runs.
fn block_succs(body: &IrBody, bid: BlockId, rt: &Retargets) -> (Option<BlockId>, Vec<BlockId>) {
    let mut fallthrough = None;
    let mut branches = Vec::new();
    for &iid in &body.blocks[bid].stream {
        match &body.instructions[iid] {
            Inst::Jump { target } => fallthrough = Some(rtgt(rt, bid, *target)),
            Inst::JumpIfFalse { target, .. } | Inst::ForNext { target, .. } => {
                branches.push(rtgt(rt, bid, *target))
            }
            Inst::Switch { table, default, .. } => {
                branches.extend(table.iter().map(|&t| rtgt(rt, bid, t)));
                branches.push(rtgt(rt, bid, *default));
            }
            _ => {}
        }
    }
    (fallthrough, branches)
}

/// DFS emission order — the full traversal order including empty blocks.
fn dfs_order(body: &IrBody, rt: &Retargets) -> Vec<BlockId> {
    let mut order = Vec::new();
    let mut placed = vec![false; body.blocks.len()];
    let mut stack = vec![BlockId::ZERO];
    while let Some(b) = stack.pop() {
        if std::mem::replace(&mut placed[b.index()], true) {
            continue;
        }
        order.push(b);
        let (fallthrough, branches) = block_succs(body, b, rt);
        for s in branches {
            if !placed[s.index()] {
                stack.push(s);
            }
        }
        if let Some(f) = fallthrough
            && !placed[f.index()]
        {
            stack.push(f);
        }
    }
    order
}

/// position -> ([earliest fwd jumper], [latest back jumper]) — shared by
/// the layout repair and scope construction
fn mark_targets(
    body: &IrBody,
    order: &[BlockId],
    pos: &HashMap<BlockId, usize>,
    rt: &Retargets,
) -> Result<HashMap<usize, [Option<usize>; 2]>, Bail> {
    let mut targets: HashMap<usize, [Option<usize>; 2]> = HashMap::new();
    for (j, &bid) in order.iter().enumerate() {
        for &iid in &body.blocks[bid].stream {
            let mut mark = |t: BlockId| -> Result<(), Bail> {
                let t = rtgt(rt, bid, t);
                let ti = *pos
                    .get(&t)
                    .ok_or_else(|| format!("jump to unreachable block b{}", t.index()))?;
                let ent = targets.entry(ti).or_default();
                if ti <= j {
                    ent[1] = Some(ent[1].map_or(j, |m| m.max(j)));
                } else {
                    ent[0] = Some(ent[0].map_or(j, |m| m.min(j)));
                }
                Ok(())
            };
            match &body.instructions[iid] {
                Inst::Jump { target }
                | Inst::JumpIfFalse { target, .. }
                | Inst::ForNext { target, .. } => mark(*target)?,
                Inst::Switch { table, default, .. } => {
                    for &t in table.iter().chain(std::iter::once(default)) {
                        mark(t)?;
                    }
                }
                _ => {}
            }
        }
    }
    // retargeted edges also carry the dup'd tail's mid-stream branches:
    // a conditional continue inside the tail is emitted at the pred's
    // position, so its target gets an edge *from* the pred (a forward
    // loop-entry when the pred sits before the loop)
    for (&(p, lb), _) in rt {
        let p = BlockId::from(p);
        let Some(&pj) = pos.get(&p) else { continue };
        for &iid in &body.blocks[BlockId::from(lb)].stream {
            match &body.instructions[iid] {
                Inst::JumpIfFalse { target, .. } => {
                    let t = *target;
                    let ti = *pos
                        .get(&t)
                        .ok_or_else(|| {
                            format!("jump to unreachable block b{}", t.index())
                        })?;
                    let ent = targets.entry(ti).or_default();
                    if ti <= pj {
                        ent[1] = Some(ent[1].map_or(pj, |m| m.max(pj)));
                    } else {
                        ent[0] = Some(ent[0].map_or(pj, |m| m.min(pj)));
                    }
                }
                Inst::Jump { .. } => break,
                _ => {}
            }
        }
    }
    Ok(targets)
}

/// Last-resort relayout for a livelocked or dead-ended repair: rebuild `all`
/// as a topological order over forward edges. Edges into the DFS active
/// path are the real back edges (loop latches); every other edge must land
/// forward for merge/loop scopes to nest. Ties resolve toward the current
/// order so the result stays close to DFS where possible.
fn topo_relax(body: &IrBody, all: &[BlockId], rt: &Retargets) -> Vec<BlockId> {
    // DFS spanning tree — edges to a block on the active path are back edges.
    let mut back: HashSet<(BlockId, BlockId)> = HashSet::new();
    {
        let mut on_path: HashSet<BlockId> = HashSet::new();
        let mut done: HashSet<BlockId> = HashSet::new();
        let mut stack: Vec<(BlockId, Vec<BlockId>)> = Vec::new();
        let push = |stack: &mut Vec<(BlockId, Vec<BlockId>)>,
                        on_path: &mut HashSet<BlockId>,
                        b: BlockId| {
            let (ft, br) = block_succs(body, b, rt);
            on_path.insert(b);
            stack.push((b, br.into_iter().chain(ft).collect()));
        };
        if let Some(&first) = all.first() {
            push(&mut stack, &mut on_path, first);
        }
        while let Some((b, succs)) = stack.last_mut() {
            if let Some(s) = succs.pop() {
                if on_path.contains(&s) {
                    back.insert((*b, s));
                } else if !done.contains(&s) {
                    push(&mut stack, &mut on_path, s);
                }
            } else {
                done.insert(*b);
                on_path.remove(b);
                stack.pop();
            }
        }
    }
    // Kahn over non-back edges, preferring the current order.
    let prio: HashMap<BlockId, usize> =
        all.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    let mut indeg: HashMap<BlockId, usize> = all.iter().map(|&b| (b, 0)).collect();
    let mut fwd: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for &b in all {
        let (ft, br) = block_succs(body, b, rt);
        for s in br.into_iter().chain(ft) {
            if back.contains(&(b, s)) || !indeg.contains_key(&s) {
                continue;
            }
            *indeg.entry(s).or_default() += 1;
            fwd.entry(b).or_default().push(s);
        }
    }
    let mut avail: BinaryHeap<Reverse<(usize, u32)>> = all
        .iter()
        .filter(|b| indeg[b] == 0)
        .map(|&b| Reverse((prio[&b], b.index() as u32)))
        .collect();
    let mut out = Vec::with_capacity(all.len());
    let mut emitted: HashSet<BlockId> = HashSet::new();
    while let Some(Reverse((_, bi))) = avail.pop() {
        let b = BlockId::from(bi);
        if !emitted.insert(b) {
            continue;
        }
        out.push(b);
        for &s in fwd.get(&b).into_iter().flatten() {
            if let Some(d) = indeg.get_mut(&s) {
                *d -= 1;
                if *d == 0 {
                    avail.push(Reverse((prio[&s], s.index() as u32)));
                }
            }
        }
    }
    // anything left was only reachable through a removed back edge — keep
    // it in original order rather than dropping the block.
    for &b in all {
        if !emitted.contains(&b) {
            out.push(b);
        }
    }
    out
}

/// dominator sets over the block CFG — plain fixpoint (graphs are small)
fn dominators(
    body: &IrBody,
    all: &[BlockId],
    rt: &Retargets,
) -> HashMap<BlockId, HashSet<BlockId>> {
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for &b in all {
        let (ft, mut succs) = block_succs(body, b, rt);
        if let Some(f) = ft {
            succs.push(f);
        }
        for s in succs {
            preds.entry(s).or_default().push(b);
        }
    }
    let universe: HashSet<BlockId> = all.iter().copied().collect();
    let mut dom: HashMap<BlockId, HashSet<BlockId>> =
        all.iter().map(|&b| (b, universe.clone())).collect();
    if let Some(&e) = all.first() {
        dom.insert(e, [e].into());
    }
    loop {
        let mut changed = false;
        for &b in all.iter().skip(1) {
            let mut new: HashSet<BlockId> = [b].into();
            if let Some(ps) = preds.get(&b)
                && let Some((&p0, rest)) = ps.split_first()
            {
                let mut acc = dom[&p0].clone();
                for p in rest {
                    acc.retain(|x| dom[p].contains(x));
                }
                new.extend(acc);
            }
            if dom[&b] != new {
                dom.insert(b, new);
                changed = true;
            }
        }
        if !changed {
            return dom;
        }
    }
}

/// can `from` reach `to` through block successors?
fn reaches(body: &IrBody, from: BlockId, to: BlockId, rt: &Retargets) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![from];
    while let Some(b) = stack.pop() {
        if b == to {
            return true;
        }
        if !seen.insert(b) {
            continue;
        }
        let (ft, br) = block_succs(body, b, rt);
        stack.extend(br);
        stack.extend(ft);
    }
    false
}

/// `(order, pos)`: `order` is the non-empty DFS order; `pos[bid]` is the
/// position a block would occupy — empty blocks fold onto the next non-empty
/// block's position (same semantics as codegen's `block_offset`).
///
/// Two layout repairs iterate to a fixpoint:
/// - fake back edges: a "back edge" j→h where h cannot reach j is no loop —
///   the jumper is just misplaced (an if/else arm DFS visited late). Move
///   its run (plus dominated runs) to just before h's run.
/// - merges inside real loops: when a merge position lands strictly inside
///   a loop's positional extent (a `block` scope would expire inside a
///   `loop` scope), relocate that merge's run plus dominated runs to just
///   after the loop's last back-jumper.
/// - merges *on* the loop's latch (`t == j`): a forward edge enters the
///   loop at its tail. When the latch is a pure tail (unconditional `Jump`,
///   no phi, `Jump`/`JumpIfFalse` preds only), retarget each pred edge past
///   it — the pred inlines the tail's insts and branches to its successor
///   (tail duplication), and the crossing merge vanishes.
/// Runs (an empty-label prefix plus the non-empty block it aliases) keep
/// jumpers' aliases consistent; dominated runs move so forward edges stay
/// forward. Runs that are back-jump sources stay anchored.
fn layout(body: &IrBody) -> (Vec<BlockId>, HashMap<BlockId, usize>, Retargets) {
    let mut rt = Retargets::new();
    let mut all = dfs_order(body, &rt);
    let mut order = Vec::new();
    let mut pos = HashMap::new();
    let mut seen: HashSet<Vec<BlockId>> = HashSet::new();
    // repair livelock (or a dead end): fall back to a topological relayout
    // over the ORIGINAL edges (fresh retargets — accumulated tail-dup
    // retargets would poison the DFS back-edge detection). Nested
    // instrumented diamonds (else-if chains under the coverage rewriter)
    // livelock the single-run moves.
    macro_rules! relax {
        () => {{
            let fresh = Retargets::new();
            let relaxed = topo_relax(body, &all, &fresh);
            if relaxed == all && rt.is_empty() {
                break;
            }
            rt = fresh;
            all = relaxed;
            continue;
        }};
    }
    for _ in 0..64 {
        if !seen.insert(all.clone()) {
            relax!();
        }
        order.clear();
        pos.clear();
        for &b in &all {
            pos.insert(b, order.len());
            if !body.blocks[b].stream.is_empty() {
                order.push(b);
            }
        }
        let Ok(targets) = mark_targets(body, &order, &pos, &rt) else {
            if std::env::var_os("WG_DEBUG").is_some() {
                eprintln!("layout repair: mark_targets failed");
            }
            break;
        };
        // runs: each ends at a non-empty block so run index == position
        let mut runs: Vec<Vec<BlockId>> = Vec::new();
        let mut cur = Vec::new();
        for &b in &all {
            cur.push(b);
            if !body.blocks[b].stream.is_empty() {
                runs.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            runs.push(cur);
        }
        let back_jumpers: HashSet<usize> =
            targets.values().filter_map(|&[_, b]| b).collect();
        // fake back edge? j -> h where h cannot reach j
        let mut fake = None;
        for (&h, &[_, b]) in &targets {
            let Some(j) = b else { continue };
            if j >= runs.len() || h >= runs.len() {
                continue;
            }
            let (hb, jb) = (*runs[h].last().unwrap(), *runs[j].last().unwrap());
            if !reaches(body, hb, jb, &rt) {
                fake = Some((h, j));
                break;
            }
        }
        if let Some((h, j)) = fake {
            let hb = *runs[h].last().unwrap();
            let jb = *runs[j].last().unwrap();
            let dom = dominators(body, &all, &rt);
            let (mut kept, mut moved) = (Vec::new(), Vec::new());
            for (i, r) in runs.into_iter().enumerate() {
                if i == j
                    || (i != h
                        && !back_jumpers.contains(&i)
                        && r.iter().all(|b| dom[b].contains(&jb)))
                {
                    moved.push(r);
                } else {
                    kept.push(r);
                }
            }
            let hi = kept
                .iter()
                .position(|r| r.contains(&hb))
                .expect("merge run must survive");
            kept.splice(hi..hi, moved);
            all = kept.concat();
            if std::env::var_os("WG_DEBUG").is_some() {
                eprintln!("layout repair: fake back-edge j={j} before h={h}");
            }
            continue;
        }
        // merges at t (earliest fwd jumper f) strictly inside a loop span
        // [h, j] when f < h < t <= j — try each candidate until one moves
        let mut cands = Vec::new();
        for (&t, &[f, _]) in &targets {
            let Some(f) = f else { continue };
            for (&h, &[_, b]) in &targets {
                let Some(j) = b else { continue };
                if f < h && h < t && t <= j {
                    cands.push((t, h, j));
                }
            }
        }
        cands.sort();
        let mut moved = false;
        for (t, h, j) in cands {
            if t == j {
                moved = dup_latch_tail(body, &all, &runs, t, &mut rt);
                if moved {
                    if std::env::var_os("WG_DEBUG").is_some() {
                        eprintln!("layout repair: dup latch-tail at pos {t}");
                    }
                    break;
                }
                continue;
            }
            let mblock = *runs[t].last().unwrap();
            // t may itself jump back — moving it past j is only safe when
            // its back edges target headers before this loop's (a target
            // at or inside this loop means t belongs to it and moving it
            // out would break that loop's extent)
            let (ft, br) = block_succs(body, mblock, &rt);
            let own_loop_broken = br.iter().chain(ft.iter()).any(|tb| {
                pos.get(tb).is_some_and(|&p| p <= t && p >= h)
            });
            if own_loop_broken {
                // t is part of this loop's cycle — it can't leave past the
                // latch, but it may lift *before* the header: its earliest
                // jumper is already pre-loop, so if no in-loop dominator
                // anchors it, moving its run (plus dominated runs outside
                // the loop) ahead of runs[h] keeps every edge forward and
                // the loop intact. The merge then ends before the loop —
                // disjoint.
                let dom = dominators(body, &all, &rt);
                let in_loop_anchor = dom[&mblock].iter().any(|b| {
                    pos.get(b)
                        .is_some_and(|&pp| pp > h && pp <= j && pp != t)
                });
                if in_loop_anchor {
                    continue;
                }
                let hblock = *runs[h].last().unwrap();
                let (mut kept, mut moved_runs) = (Vec::new(), Vec::new());
                for (i, r) in runs.clone().into_iter().enumerate() {
                    if i == t
                        || (!(h..=j).contains(&i)
                            && !back_jumpers.contains(&i)
                            && r.iter().all(|b| dom[b].contains(&mblock)))
                    {
                        moved_runs.push(r);
                    } else {
                        kept.push(r);
                    }
                }
                let Some(hi) = kept.iter().position(|r| r.contains(&hblock))
                else {
                    continue;
                };
                kept.splice(hi..hi, moved_runs);
                all = kept.concat();
                if std::env::var_os("WG_DEBUG").is_some() {
                    eprintln!("layout repair: merge t={t} lifted before loop h={h}");
                }
                moved = true;
                break;
            }
            let dom = dominators(body, &all, &rt);
            let jblock = *runs[j].last().unwrap();
            let (mut kept, mut moved_runs) = (Vec::new(), Vec::new());
            for (i, r) in runs.clone().into_iter().enumerate() {
                if i == t
                    || (!back_jumpers.contains(&i)
                        && r.iter().all(|b| dom[b].contains(&mblock)))
                {
                    moved_runs.push(r);
                } else {
                    kept.push(r);
                }
            }
            let Some(ji) = kept.iter().position(|r| r.contains(&jblock))
            else {
                continue; // anchor lost — try the next candidate
            };
            kept.splice(ji + 1..ji + 1, moved_runs);
            all = kept.concat();
            if std::env::var_os("WG_DEBUG").is_some() {
                eprintln!("layout repair: merge t={t} past loop [{h},{j}]");
            }
            moved = true;
            break;
        }
        if !moved {
            relax!();
        }
    }
    (order, pos, rt)
}

/// Try tail-duplicating the merge block at position `t`: it must be a
/// single-block run whose last inst is an unconditional `Jump` to `nt`, all
/// other insts non-terminators, and every predecessor edge must be a `Jump`
/// or `JumpIfFalse` (the only terminators the emitter can inline past).
/// On success installs `rt[(pred, lb)] = nt` for every pred and returns true.
fn dup_latch_tail(
    body: &IrBody,
    all: &[BlockId],
    runs: &[Vec<BlockId>],
    t: usize,
    rt: &mut Retargets,
) -> bool {
    if runs[t].len() != 1 {
        return false;
    }
    let lb = *runs[t].last().unwrap();
    let stream = &body.blocks[lb].stream;
    let Some(&last) = stream.last() else {
        return false;
    };
    let Inst::Jump { target: nt } = &body.instructions[last] else {
        return false;
    };
    if *nt == lb {
        return false;
    }
    // every earlier inst must be inlinable: plain ops plus conditional
    // branches (each becomes a virtual edge at the pred's position); a
    // Jump ends the tail early, everything else terminates
    let mut seen_jump = false;
    for &iid in stream {
        if seen_jump {
            break;
        }
        match &body.instructions[iid] {
            Inst::Jump { .. } => seen_jump = true,
            Inst::JumpIfFalse { .. } | Inst::Phi(_) => {}
            Inst::ForNext { .. } | Inst::Switch { .. } | Inst::Return(_) => {
                return false;
            }
            _ => {}
        }
    }
    // collect pred edges — only Jump/JumpIfFalse terminators may retarget
    let mut preds: Vec<BlockId> = Vec::new();
    for &p in all {
        for &iid in &body.blocks[p].stream {
            match &body.instructions[iid] {
                Inst::Jump { target } | Inst::JumpIfFalse { target, .. } if *target == lb => {
                    preds.push(p);
                }
                Inst::ForNext { target, .. } if *target == lb => return false,
                Inst::Switch { table, default, .. }
                    if table.contains(&lb) || *default == lb =>
                {
                    return false;
                }
                _ => {}
            }
        }
    }
    if preds.is_empty() {
        return false;
    }
    if std::env::var_os("WG_DEBUG").is_some() {
        let ps: Vec<String> = preds.iter().map(|p| format!("b{}", p.index())).collect();
        eprintln!("dup b{} (pos {t}) -> b{}; preds {ps:?}", lb.index(), nt.index());
    }
    for p in preds {
        rt.insert((p.index() as u32, lb.index() as u32), nt.index() as u32);
    }
    true
}

/// Branch targets → `block`/`loop` scopes at *block* granularity, then the
/// same interval-repair the bytecode lane uses.
fn scopes_ir(
    body: &IrBody,
    order: &[BlockId],
    pos: &HashMap<BlockId, usize>,
    rt: &Retargets,
) -> Result<Vec<Scope>, Bail> {
    let nops = order.len();
    // A position may be BOTH a forward merge and a loop header (the `for`
    // loop's header is entered by a forward jump from the pre-header and
    // re-entered by the latch's back edge) — that splits into a `block`
    // closing at the pos and a `loop` opening at it, disjoint intervals,
    // always nestable.
    let targets = mark_targets(body, order, pos, rt)?;
    if std::env::var_os("WG_DEBUG").is_some() {
        eprintln!("== layout ==");
        for (j, &b) in order.iter().enumerate() {
            let names: Vec<String> = body.blocks[b]
                .stream
                .iter()
                .map(|&iid| format!("{:?}", body.instructions[iid]).chars().take(64).collect())
                .collect();
            eprintln!("  pos{j} = b{} {:?}", b.index(), names);
        }
        eprintln!("  targets = {targets:?}");
    }
    let mut out: Vec<Scope> = targets
        .into_iter()
        .flat_map(|(t, [fwd, back])| {
            let mut v = Vec::with_capacity(2);
            if let Some(j) = fwd {
                v.push(Scope {
                    open: j,
                    close: t,
                    kind: ScopeKind::Block,
                    target: t,
                });
            }
            if let Some(j) = back {
                v.push(Scope {
                    open: t,
                    close: j + 1,
                    kind: ScopeKind::Loop,
                    target: t,
                });
            }
            v
        })
        .collect();
    for _ in 0..64 {
        match simulate(&mut out, nops) {
            Ok(()) => {
                if std::env::var_os("WG_DEBUG").is_some() {
                    eprintln!("== scopes ==");
                    for (j, &b) in order.iter().enumerate() {
                        let names: Vec<String> = body.blocks[b]
                            .stream
                            .iter()
                            .map(|&iid| {
                                format!("{:?}", body.instructions[iid])
                                    .chars()
                                    .take(64)
                                    .collect()
                            })
                            .collect();
                        eprintln!("  pos{j} = b{} {:?}", b.index(), names);
                    }
                    for s in &out {
                        eprintln!("  {:?}[{},{}) t={}", s.kind, s.open, s.close, s.target);
                    }
                }
                return Ok(out);
            }
            Err(Repair::ExtendLoop(ix, nc)) => out[ix].close = nc,
            Err(Repair::ShiftBlock(ix, no)) => out[ix].open = no,
            Err(Repair::Fatal(e)) => return Err(e),
        }
    }
    bail!("scope repair did not converge")
}

/// (pred, target) -> [(phi inst, value inst)] to copy on that edge. Invariant
/// checked: every phi predecessor ends in a plain `Jump` to the phi's block.
fn phi_copies(body: &IrBody) -> Result<HashMap<(u32, u32), Vec<(u32, u32)>>, Bail> {
    let mut copies: HashMap<(u32, u32), Vec<(u32, u32)>> = HashMap::new();
    for (bid, block) in body.blocks.iter() {
        for &iid in &block.stream {
            let Inst::Phi(branches) = &body.instructions[iid] else {
                continue;
            };
            for (pred, v) in branches {
                let pstream = &body.blocks[*pred].stream;
                match pstream.last().map(|&i| &body.instructions[i]) {
                    Some(Inst::Jump { target }) if *target == bid => {}
                    _ => bail!("phi predecessor b{} not jump-terminated", pred.index()),
                }
                copies
                    .entry((pred.index() as u32, bid.index() as u32))
                    .or_default()
                    .push((iid.index() as u32, v.index() as u32));
            }
        }
    }
    Ok(copies)
}

// ---------- emission ----------

struct Emi<'a> {
    body: &'a IrBody,
    class: &'a HashMap<u32, K>,
    lclass: &'a HashMap<u32, K>,
    /// inst idx -> wasm local (classed and word-materialized insts;
    /// GetLocal aliases, consts inline)
    ilocal: &'a HashMap<u32, u32>,
    /// inst idx -> i64 slot holds the raw word (else it holds the class)
    wused: &'a HashSet<u32>,
    /// local idx -> wasm local
    llocal: &'a HashMap<u32, u32>,
    sigs: &'a [Option<Sig>],
    func_map: &'a HashMap<usize, u32>,
    /// inst idx -> callee body
    callee: &'a HashMap<u32, usize>,
    natives: &'a HashMap<(u32, Vec<K>, Option<K>), u32>,
    ret_k: Option<K>,
    tmp: u32,
    /// i32 scratch: alloc size / haystack base
    sz: u32,
    /// i32 scratch: object base / `In` found flag
    hp: u32,
    /// i32 scratch: len / index
    tb: u32,
    /// i32 scratch: new cap / loop counter
    tc: u32,
    /// body 0 — its entry-shared locals live in the memory region below
    is_entry: bool,
    /// local indices addressed via GetEntry/SetEntry anywhere in the module
    entry_locals: &'a HashSet<u32>,
    /// linear-memory base of the entry-local region (8B per entry local)
    entry_base: u32,
    /// static-object addresses + scratch regions
    ctx: &'a Statics,
    /// internal helper func index by name
    helpers: &'a HashMap<&'static str, u32>,
    /// body idx -> function-table index for its call_indirect trampoline
    tramp: &'a HashMap<usize, u32>,
    /// type index of the uniform trampoline signature (i32,i32,i32)->i64
    tramp_ty: u32,
    /// per-body emitted flags — RefBody/MakeClosure to a skipped body bails
    emitted: &'a [bool],
    /// layout repair's tail-duplication edge retargets
    rt: &'a Retargets,
    /// (pred, target) -> phi copies emitted on that edge
    copies: &'a HashMap<(u32, u32), Vec<(u32, u32)>>,
    stack: Vec<Scope>,
    f: Function,
    code_off: u32,
    srcmap: Vec<(u32, u32)>,
    fuel_g: Option<u32>,
    pause_g: Option<u32>,
    cov_base: u32,
    /// cov::* sink map — those CallNatives emit memory records, not imports
    sink: &'a crate::CovSink,
    /// float natives inlined as f64 ops (else they stay imports)
    math: &'a crate::MathNatives,
    /// cov-sink layout + helper indices — present iff the sink is live
    cov: Option<&'a CovCtx>,
}

impl Emi<'_> {
    fn ins(&mut self, i: Instruction) {
        let mut v = Vec::with_capacity(8);
        i.encode(&mut v);
        self.code_off += v.len() as u32;
        self.f.raw(v);
    }

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

    fn trap_if(&mut self) {
        self.ins(Instruction::If(BlockType::Empty));
        self.ins(Instruction::Unreachable);
        self.ins(Instruction::End);
    }

    /// The class an inst's slot stores — scalar class, or `Word` for the raw
    /// 8-byte repr (heap handles, strids, payloads).
    fn stored_k(&self, iid: InstId) -> K {
        if let Inst::GetLocal(l) = &self.body.instructions[iid] {
            self.lclass
                .get(&(l.index() as u32))
                .copied()
                .unwrap_or(K::Word)
        } else {
            self.class
                .get(&(iid.index() as u32))
                .copied()
                .unwrap_or(K::Word)
        }
    }

    /// Push `iid`'s value, coerced to `want`. Constants emit inline;
    /// `GetLocal` aliases the local's slot. `want = K::Word` produces the
    /// raw word; a word slot read in a scalar context reinterprets.
    fn get(&mut self, iid: InstId, want: K) -> Result<(), Bail> {
        if let Inst::Constant(c) = &self.body.instructions[iid] {
            return self.emit_const(c, want);
        }
        let k = self.stored_k(iid);
        let li = if let Inst::GetLocal(l) = &self.body.instructions[iid] {
            let lx = l.index() as u32;
            if self.is_entry && self.entry_locals.contains(&lx) {
                // entry-shared locals live in the memory region — the
                // loaded word coerces to whatever repr the reader wants
                self.ins(Instruction::I32Const((self.entry_base + lx * 8) as i32));
                self.ins(Instruction::I64Load(mem_arg(0, 3)));
                if K::Word != want {
                    self.coerce(K::Word, want)?;
                }
                return Ok(());
            }
            *self.llocal.get(&lx).ok_or("read of dead local")?
        } else {
            *self
                .ilocal
                .get(&(iid.index() as u32))
                .ok_or_else(|| {
                    format!(
                        "inst {} not materialized: {:?}",
                        iid.index(),
                        self.body.instructions[iid]
                    )
                })?
        };
        self.ins(Instruction::LocalGet(li));
        if k != want {
            self.coerce(k, want)?;
        }
        Ok(())
    }

    fn emit_const(&mut self, c: &Constant, want: K) -> Result<(), Bail> {
        match (c, want) {
            // word reprs: Null is 0, strs are interned strids, scalars are
            // their bit patterns — the same convention `resume.rs` writes
            (_, K::Word) => match c {
                Constant::Int(v) => self.ins(Instruction::I64Const(*v)),
                Constant::Float(v) => self.ins(Instruction::I64Const(v.to_bits() as i64)),
                Constant::Bool(v) => self.ins(Instruction::I64Const(*v as i64)),
                // tagged static object — the word is its absolute address,
                // not a strid (no shared intern table on this lane)
                Constant::Str(s) => {
                    let a = self
                        .ctx
                        .str_objs
                        .get(&(s.index() as u32))
                        .copied()
                        .ok_or("str const not laid out")?;
                    self.ins(Instruction::I64Const(a as i64));
                }
                Constant::Null => self.ins(Instruction::I64Const(0)),
                Constant::Array(_) => bail!("const array literal can't materialize"),
            },
            (Constant::Int(v), K::Int) => self.ins(Instruction::I64Const(*v)),
            (Constant::Int(v), K::Float) => {
                self.ins(Instruction::F64Const((*v as f64).into()));
            }
            (Constant::Int(v), K::Bool) => {
                self.ins(Instruction::I32Const((*v != 0) as i32));
            }
            (Constant::Float(v), K::Float) => self.ins(Instruction::F64Const((*v).into())),
            (Constant::Float(v), K::Int) => self.ins(Instruction::I64Const(*v as i64)),
            (Constant::Float(v), K::Bool) => {
                self.ins(Instruction::I32Const((*v != 0.0) as i32));
            }
            (Constant::Bool(v), K::Bool) => self.ins(Instruction::I32Const(*v as i32)),
            (Constant::Bool(v), K::Int) => self.ins(Instruction::I64Const(*v as i64)),
            (Constant::Bool(v), K::Float) => {
                self.ins(Instruction::F64Const((*v as i32 as f64).into()));
            }
            _ => bail!("const {c:?} can't serve {want:?}"),
        };
        Ok(())
    }

    fn coerce(&mut self, k: K, want: K) -> Result<(), Bail> {
        match (k, want) {
            (K::Int, K::Float) => self.ins(Instruction::F64ConvertI64S),
            (K::Float, K::Int) => self.ins(Instruction::I64TruncF64S),
            (K::Bool, K::Int) => self.ins(Instruction::I64ExtendI32U),
            (K::Bool, K::Float) => self.ins(Instruction::F64ConvertI32S),
            (K::Int, K::Bool) => {
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64Ne);
            }
            (K::Float, K::Bool) => {
                self.ins(Instruction::F64Const(0.0f64.into()));
                self.ins(Instruction::F64Ne);
            }
            // the word IS the repr: int/strid/handle words pass through,
            // float words are raw bits, bools extend/wrap
            (K::Int, K::Word) | (K::Word, K::Int) => {}
            (K::Float, K::Word) => self.ins(Instruction::I64ReinterpretF64),
            (K::Word, K::Float) => self.ins(Instruction::F64ReinterpretI64),
            (K::Bool, K::Word) => self.ins(Instruction::I64ExtendI32U),
            (K::Word, K::Bool) => self.ins(Instruction::I32WrapI64),
            _ => bail!("coerce {k:?}->{want:?}"),
        };
        Ok(())
    }

    /// Store the stack-top value (produced in `produced` repr) into `iid`'s
    /// slot — its scalar class if classed, its word slot if word-materialized,
    /// or drop it when the result is dead.
    fn store_dst(&mut self, iid: InstId, produced: K) -> Result<(), Bail> {
        let ix = iid.index() as u32;
        let dk = match self.class.get(&ix) {
            Some(&k) => k,
            None if self.wused.contains(&ix) => K::Word,
            None => {
                self.ins(Instruction::Drop);
                return Ok(());
            }
        };
        if produced != dk {
            self.coerce(produced, dk)?;
        }
        self.ins(Instruction::LocalSet(self.ilocal[&ix]));
        Ok(())
    }

    /// only `AccessKind::Direct` field/index reads lower to bare word loads —
    /// Option access would need a Null-on-miss we don't model
    fn direct(&self, kind: &compile::AccessKind) -> Result<(), Bail> {
        match kind {
            compile::AccessKind::Direct => Ok(()),
            _ => bail!("optional access needs Null values"),
        }
    }

    /// bump-allocate `[size]` bytes from the __hp arena via the `__alloc`
    /// helper; pops size (i32) from the wasm stack, pushes the handle.
    fn alloc(&mut self) {
        self.ins(Instruction::Call(self.helpers[H_ALLOC]));
    }

    fn depth(&self, t: usize, kind: ScopeKind) -> Result<u32, Bail> {
        for (i, s) in self.stack.iter().enumerate().rev() {
            if s.target == t && s.kind == kind {
                return Ok((self.stack.len() - 1 - i) as u32);
            }
        }
        bail!("no open {kind:?} scope for target block pos {t}")
    }

    fn br(&mut self, t: usize, kind: ScopeKind) -> Result<(), Bail> {
        let d = self.depth(t, kind)?;
        if kind == ScopeKind::Loop
            && let Some(g) = self.pause_g
        {
            self.ins(Instruction::GlobalGet(g));
            self.trap_if();
        }
        self.ins(Instruction::Br(d));
        Ok(())
    }

    fn br_if(&mut self, t: usize, kind: ScopeKind, is_true: bool) -> Result<(), Bail> {
        if !is_true {
            self.ins(Instruction::I32Eqz);
        }
        let d = self.depth(t, kind)?;
        if kind == ScopeKind::Loop
            && let Some(g) = self.pause_g
        {
            self.ins(Instruction::GlobalGet(g));
            self.trap_if();
        }
        self.ins(Instruction::BrIf(d));
        Ok(())
    }

    /// `l OP r` checked for i64 overflow; leaves the result on the stack.
    fn checked_int(&mut self, l: InstId, r: InstId, op: BinOp) -> Result<(), Bail> {
        if matches!(op, BinOp::Mod | BinOp::IDiv) {
            self.get(l, K::Int)?;
            self.get(r, K::Int)?;
            self.ins(if op == BinOp::Mod {
                Instruction::I64RemS
            } else {
                Instruction::I64DivS
            });
            return Ok(());
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
            BinOp::Add | BinOp::Sub => {
                let (c1, c2) = if op == BinOp::Add {
                    (Instruction::I64LtS, Instruction::I64GtS)
                } else {
                    (Instruction::I64GtS, Instruction::I64LtS)
                };
                self.get(r, K::Int)?;
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64GtS);
                self.ins(Instruction::LocalGet(tmp));
                self.get(l, K::Int)?;
                self.ins(c1);
                self.ins(Instruction::I32And);
                self.get(r, K::Int)?;
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64LtS);
                self.ins(Instruction::LocalGet(tmp));
                self.get(l, K::Int)?;
                self.ins(c2);
                self.ins(Instruction::I32And);
                self.ins(Instruction::I32Or);
                self.trap_if();
            }
            BinOp::Mult => {
                self.get(r, K::Int)?;
                self.ins(Instruction::I64Eqz);
                self.ins(Instruction::I32Eqz);
                self.ins(Instruction::If(BlockType::Empty));
                self.ins(Instruction::LocalGet(tmp));
                self.get(r, K::Int)?;
                self.ins(Instruction::I64DivS);
                self.get(l, K::Int)?;
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
    fn checked_int_imm(&mut self, l: InstId, v: i64, op: BinOp) -> Result<(), Bail> {
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
                self.get(l, K::Int)?;
                self.ins(c1);
                self.ins(Instruction::I32And);
                self.ins(Instruction::I64Const(v));
                self.ins(Instruction::I64Const(0));
                self.ins(Instruction::I64LtS);
                self.ins(Instruction::LocalGet(tmp));
                self.get(l, K::Int)?;
                self.ins(c2);
                self.ins(Instruction::I32And);
                self.ins(Instruction::I32Or);
                self.trap_if();
            }
            BinOp::Mult => {
                self.ins(Instruction::LocalGet(tmp));
                self.ins(Instruction::I64Const(v));
                self.ins(Instruction::I64DivS);
                self.get(l, K::Int)?;
                self.ins(Instruction::I64Ne);
                self.trap_if();
            }
            _ => unreachable!(),
        }
        self.ins(Instruction::LocalGet(tmp));
        Ok(())
    }

    fn cmp(&mut self, l: InstId, r: InstId, want: K, i: Instruction) -> Result<(), Bail> {
        self.get(l, want)?;
        self.get(r, want)?;
        self.ins(i);
        Ok(())
    }

    fn cmp_imm(&mut self, l: InstId, v: i64, want: K, i: Instruction) -> Result<(), Bail> {
        self.get(l, want)?;
        match want {
            K::Int => self.ins(Instruction::I64Const(v)),
            K::Float => self.ins(Instruction::F64Const(f64::from_bits(v as u64).into())),
            K::Bool => self.ins(Instruction::I32Const(v as i32)),
            K::Word => unreachable!("word is never a comparison class"),
        };
        self.ins(i);
        Ok(())
    }

    /// If `iid` is a constant usable as an immediate in `kind`, return its
    /// value (float constants bit-cast into the i64 slot, same as codegen).
    fn imm(&self, iid: InstId, kind: OperandKind) -> Option<i64> {
        match (kind, &self.body.instructions[iid]) {
            (OperandKind::Int, Inst::Constant(Constant::Int(v))) => Some(*v),
            (OperandKind::Float, Inst::Constant(Constant::Float(f))) => Some(f.to_bits() as i64),
            _ => None,
        }
    }

    fn emit_bin(
        &mut self,
        dst: InstId,
        l: InstId,
        op: BinOp,
        r: InstId,
        kind: OperandKind,
    ) -> Result<(), Bail> {
        // fold a constant operand into the immediate form, like codegen does.
        // `nc` is the non-const operand when `imm` fired.
        let mut nc = l;
        let mut op = op;
        let mut imm: Option<i64> = None;
        match op {
            BinOp::Add | BinOp::Mult => {
                if let Some(v) = self.imm(r, kind) {
                    imm = Some(v);
                } else if let Some(v) = self.imm(l, kind) {
                    imm = Some(v);
                    nc = r;
                }
            }
            BinOp::Sub | BinOp::Mod | BinOp::IDiv => {
                imm = self.imm(r, kind);
            }
            BinOp::LessThan
            | BinOp::LessEqual
            | BinOp::GreaterThan
            | BinOp::GreaterEqual
            | BinOp::Identity
            | BinOp::NotEqual => {
                if let Some(v) = self.imm(r, kind) {
                    imm = Some(v);
                } else if let Some(v) = self.imm(l, kind) {
                    // `c OP x` -> `x OP' c` with the operator flipped
                    imm = Some(v);
                    nc = r;
                    op = match op {
                        BinOp::LessThan => BinOp::GreaterThan,
                        BinOp::LessEqual => BinOp::GreaterEqual,
                        BinOp::GreaterThan => BinOp::LessThan,
                        BinOp::GreaterEqual => BinOp::LessEqual,
                        other => other,
                    };
                }
            }
            _ => {}
        }

        match kind {
            OperandKind::Int => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mult | BinOp::Mod | BinOp::IDiv => {
                    if let Some(v) = imm {
                        self.checked_int_imm(nc, v, op)?;
                    } else {
                        self.checked_int(l, r, op)?;
                    }
                }
                BinOp::Div => {
                    self.get(l, K::Int)?;
                    self.ins(Instruction::F64ConvertI64S);
                    self.get(r, K::Int)?;
                    self.ins(Instruction::F64ConvertI64S);
                    self.ins(Instruction::F64Div);
                }
                BinOp::LessThan
                | BinOp::LessEqual
                | BinOp::GreaterThan
                | BinOp::GreaterEqual
                | BinOp::Identity
                | BinOp::NotEqual => {
                    let i = match op {
                        BinOp::LessThan => Instruction::I64LtS,
                        BinOp::LessEqual => Instruction::I64LeS,
                        BinOp::GreaterThan => Instruction::I64GtS,
                        BinOp::GreaterEqual => Instruction::I64GeS,
                        BinOp::Identity => Instruction::I64Eq,
                        BinOp::NotEqual => Instruction::I64Ne,
                        _ => unreachable!(),
                    };
                    if let Some(v) = imm {
                        self.cmp_imm(nc, v, K::Int, i)?;
                    } else {
                        self.cmp(l, r, K::Int, i)?;
                    }
                }
                BinOp::BitAnd => self.cmp(l, r, K::Int, Instruction::I64And)?,
                BinOp::BitOr => self.cmp(l, r, K::Int, Instruction::I64Or)?,
                BinOp::BitXor => self.cmp(l, r, K::Int, Instruction::I64Xor)?,
                BinOp::BitShiftLeft => self.cmp(l, r, K::Int, Instruction::I64Shl)?,
                BinOp::BitShiftRight => self.cmp(l, r, K::Int, Instruction::I64ShrS)?,
                _ => bail!("BinOp {op:?} on ints"),
            },
            OperandKind::Float => {
                if let Some(v) = imm {
                    match op {
                        BinOp::Add => self.cmp_imm(nc, v, K::Float, Instruction::F64Add)?,
                        BinOp::Sub => self.cmp_imm(nc, v, K::Float, Instruction::F64Sub)?,
                        BinOp::Mult => self.cmp_imm(nc, v, K::Float, Instruction::F64Mul)?,
                        BinOp::Mod => {
                            let f = f64::from_bits(v as u64);
                            self.get(nc, K::Float)?;
                            self.get(nc, K::Float)?;
                            self.ins(Instruction::F64Const(f.into()));
                            self.ins(Instruction::F64Div);
                            self.ins(Instruction::F64Trunc);
                            self.ins(Instruction::F64Const(f.into()));
                            self.ins(Instruction::F64Mul);
                            self.ins(Instruction::F64Sub);
                        }
                        BinOp::LessThan => self.cmp_imm(nc, v, K::Float, Instruction::F64Lt)?,
                        BinOp::LessEqual => self.cmp_imm(nc, v, K::Float, Instruction::F64Le)?,
                        BinOp::GreaterThan => self.cmp_imm(nc, v, K::Float, Instruction::F64Gt)?,
                        BinOp::GreaterEqual => self.cmp_imm(nc, v, K::Float, Instruction::F64Ge)?,
                        BinOp::Identity => self.cmp_imm(nc, v, K::Float, Instruction::F64Eq)?,
                        BinOp::NotEqual => self.cmp_imm(nc, v, K::Float, Instruction::F64Ne)?,
                        _ => bail!("BinOp {op:?} on float imm"),
                    }
                } else {
                    match op {
                        BinOp::Add => self.cmp(l, r, K::Float, Instruction::F64Add)?,
                        BinOp::Sub => self.cmp(l, r, K::Float, Instruction::F64Sub)?,
                        BinOp::Mult => self.cmp(l, r, K::Float, Instruction::F64Mul)?,
                        BinOp::Div | BinOp::IDiv => {
                            self.cmp(l, r, K::Float, Instruction::F64Div)?
                        }
                        BinOp::Mod => {
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
                        _ => bail!("BinOp {op:?} on floats"),
                    }
                }
            }
            OperandKind::Bool => match op {
                BinOp::Identity => self.cmp(l, r, K::Bool, Instruction::I32Eq)?,
                BinOp::NotEqual | BinOp::Xor => self.cmp(l, r, K::Bool, Instruction::I32Ne)?,
                BinOp::And => self.cmp(l, r, K::Bool, Instruction::I32And)?,
                BinOp::Or => self.cmp(l, r, K::Bool, Instruction::I32Or)?,
                _ => bail!("BinOp {op:?} on bools"),
            },
            // `==`/`!=` on untyped words is identity — heap words compare
            // by pointer, raw words by value
            OperandKind::Generic if matches!(op, BinOp::Identity | BinOp::NotEqual) => {
                let i = if op == BinOp::Identity {
                    Instruction::I64Eq
                } else {
                    Instruction::I64Ne
                };
                self.cmp(l, r, K::Word, i)?;
            }
            // strs are tagged objects — real byte ops, not interning
            OperandKind::Str => match op {
                BinOp::Add => {
                    self.get(l, K::Word)?;
                    self.ins(Instruction::I32WrapI64);
                    self.get(r, K::Word)?;
                    self.ins(Instruction::I32WrapI64);
                    self.ins(Instruction::Call(self.helpers[H_STR_CAT]));
                    self.ins(Instruction::I64ExtendI32U);
                }
                BinOp::Identity | BinOp::NotEqual => {
                    self.get(l, K::Word)?;
                    self.ins(Instruction::I32WrapI64);
                    self.get(r, K::Word)?;
                    self.ins(Instruction::I32WrapI64);
                    self.ins(Instruction::Call(self.helpers[H_STR_EQ]));
                    if op == BinOp::NotEqual {
                        self.ins(Instruction::I32Eqz);
                    }
                }
                BinOp::LessThan
                | BinOp::LessEqual
                | BinOp::GreaterThan
                | BinOp::GreaterEqual => {
                    self.get(l, K::Word)?;
                    self.ins(Instruction::I32WrapI64);
                    self.get(r, K::Word)?;
                    self.ins(Instruction::I32WrapI64);
                    self.ins(Instruction::Call(self.helpers[H_STR_CMP]));
                    self.ins(Instruction::I32Const(0));
                    let i = match op {
                        BinOp::LessThan => Instruction::I32LtS,
                        BinOp::LessEqual => Instruction::I32LeS,
                        BinOp::GreaterThan => Instruction::I32GtS,
                        _ => Instruction::I32GeS,
                    };
                    self.ins(i);
                }
                _ => bail!("BinOp {op:?} on strs"),
            },
            // dynamic binops: guess the operand class — dst's, then an
            // operand's, then Float (heap words in mimas programs are
            // overwhelmingly numeric-float; resume.rs picks the same way)
            OperandKind::Generic => {
                let g = self
                    .class
                    .get(&(dst.index() as u32))
                    .or_else(|| self.class.get(&(l.index() as u32)))
                    .or_else(|| self.class.get(&(r.index() as u32)))
                    .copied()
                    .unwrap_or(K::Float);
                let gk = match g {
                    K::Int => OperandKind::Int,
                    K::Float => OperandKind::Float,
                    K::Bool => OperandKind::Bool,
                    K::Word => unreachable!("guess is scalar"),
                };
                return self.emit_bin(dst, l, op, r, gk);
            }
        }
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
            BinOp::Div if kind == OperandKind::Int => K::Float,
            _ => binop_prod(op, kind).unwrap_or(K::Word),
        };
        self.store_dst(dst, produced)
    }

    /// Push the display string (i32 tagged object) for a `Format` value
    /// part — ints/floats go through wasm-side ascii helpers, bools pick a
    /// static object, raw words dispatch on their tag.
    fn format_part(&mut self, v: InstId) -> Result<(), Bail> {
        match self.stored_k(v) {
            K::Int => {
                self.get(v, K::Int)?;
                self.ins(Instruction::Call(self.helpers[H_I64_STR]));
            }
            K::Float => {
                self.get(v, K::Float)?;
                self.ins(Instruction::Call(self.helpers[H_F64_STR]));
            }
            K::Bool => {
                self.get(v, K::Bool)?;
                self.ins(Instruction::If(BlockType::Result(ValType::I32)));
                self.ins(Instruction::I32Const(self.ctx.true_obj as i32));
                self.ins(Instruction::Else);
                self.ins(Instruction::I32Const(self.ctx.false_obj as i32));
                self.ins(Instruction::End);
            }
            K::Word => {
                self.get(v, K::Word)?;
                self.ins(Instruction::Call(self.helpers[H_STR_OR_OBJ]));
            }
        }
        Ok(())
    }

    fn emit_call(&mut self, dst: InstId, b: usize, args: &[InstId]) -> Result<(), Bail> {
        let sig = self.sigs[b]
            .as_ref()
            .ok_or_else(|| format!("callee body {b} not emitted"))?;
        let ix = dst.index() as u32;
        if (self.class.contains_key(&ix) || self.wused.contains(&ix)) && sig.ret.is_none() {
            bail!("callee returns void but dst is read");
        }
        if sig.params.len() != args.len() {
            bail!("arity mismatch calling body {b}");
        }
        for (a, pk) in args.iter().zip(&sig.params) {
            self.get(*a, *pk)?;
        }
        let fi = self.func_map[&b];
        self.ins(Instruction::Call(fi));
        // a dead result drops; live ones store in the callee's ret class
        if let Some(rk) = sig.ret {
            self.store_dst(dst, rk)?;
        }
        Ok(())
    }

    fn emit_body(
        &mut self,
        order: &[BlockId],
        pos: &HashMap<BlockId, usize>,
        scopes: Vec<Scope>,
    ) -> Result<(), Bail> {
        let mut opens: BTreeMap<usize, Vec<Scope>> = BTreeMap::new();
        for s in scopes {
            opens.entry(s.open).or_default().push(s);
        }
        // entry-shared params arrive in wasm locals — mirror them into the
        // memory region so other bodies' entry ops see the same words
        if self.is_entry {
            for (i, p) in self.body.params.iter().enumerate() {
                let px = p.index() as u32;
                if self.entry_locals.contains(&px) {
                    let pk = self.lclass.get(&px).copied().unwrap_or(K::Word);
                    self.ins(Instruction::I32Const((self.entry_base + px * 8) as i32));
                    self.ins(Instruction::LocalGet(i as u32));
                    if pk != K::Word {
                        self.coerce(pk, K::Word)?;
                    }
                    self.ins(Instruction::I64Store(mem_arg(0, 3)));
                }
            }
        }
        let mut seq_i = 0usize;
        for (bi, &bid) in order.iter().enumerate() {
            while matches!(self.stack.last(), Some(s) if s.close <= bi) {
                self.ins(Instruction::End);
                self.stack.pop();
            }
            if let Some(mut group) = opens.remove(&bi) {
                group.sort_by_key(|s| (std::cmp::Reverse(s.close), s.kind));
                for s in group {
                    self.ins(match s.kind {
                        ScopeKind::Loop => Instruction::Loop(BlockType::Empty),
                        ScopeKind::Block => Instruction::Block(BlockType::Empty),
                    });
                    self.stack.push(s);
                }
            }
            for &iid in &self.body.blocks[bid].stream {
                self.srcmap.push((self.code_off, iid.index() as u32));
                self.tick(seq_i);
                seq_i += 1;
                self.emit_one(bid, bi, iid, pos)?;
            }
        }
        while self.stack.pop().is_some() {
            self.ins(Instruction::End);
        }
        // a diverging tail (return/br/br_table) only marks the innermost
        // frame unreachable; each `end` above pops back into a reachable
        // outer frame, so the function's own `end` is validated as reachable
        // and would demand the result on an empty stack. Mark the function
        // frame unreachable too. A reachable fall-through tail must NOT get
        // this (it would trap instead of returning).
        let diverges = order.last().is_some_and(|&b| {
            self.body.blocks[b].stream.iter().any(|&iid| {
                matches!(
                    self.body.instructions[iid],
                    Inst::Jump { .. }
                        | Inst::Switch { .. }
                        | Inst::Return(_)
                        | Inst::Raise(_)
                        | Inst::Panic
                )
            })
        });
        if diverges {
            self.ins(Instruction::Unreachable);
        }
        self.ins(Instruction::End);
        Ok(())
    }

    /// phi copies on the pred→dst edge: read all sources first (into the
    /// wasm operand stack), then assign, so interdependent phis stay
    /// simultaneous. Dead phis have no slot and are skipped; word phis copy
    /// raw words.
    fn emit_copies(&mut self, pred: BlockId, dst: BlockId) -> Result<(), Bail> {
        if let Some(cs) = self
            .copies
            .get(&(pred.index() as u32, dst.index() as u32))
        {
            let mut pairs = Vec::with_capacity(cs.len());
            for &(phi, v) in cs {
                let live = self.class.contains_key(&phi) || self.wused.contains(&phi);
                if live {
                    pairs.push((phi, v, self.stored_k(InstId::from(phi))));
                }
            }
            for &(_, v, pk) in &pairs {
                self.get(InstId::from(v), pk)?;
            }
            for &(phi, _, _) in pairs.iter().rev() {
                self.ins(Instruction::LocalSet(self.ilocal[&phi]));
            }
        }
        Ok(())
    }

    /// Emit the tail-duplicated block `lb`'s insts at the retargeted edge —
    /// everything but its final `Jump`, which the caller's own branch takes
    /// over. Only reached for layouts `dup_latch_tail` already vetted.
    fn emit_inline(
        &mut self,
        lb: BlockId,
        bi: usize,
        pos: &HashMap<BlockId, usize>,
    ) -> Result<(), Bail> {
        for &iid in &self.body.blocks[lb].stream {
            match &self.body.instructions[iid] {
                Inst::Jump { .. } => break,
                _ => self.emit_one(lb, bi, iid, pos)?,
            }
        }
        Ok(())
    }

    /// Emit one instruction of block `bid` (layout position `bi`).
    fn emit_one(
        &mut self,
        bid: BlockId,
        bi: usize,
        iid: InstId,
        pos: &HashMap<BlockId, usize>,
    ) -> Result<(), Bail> {
        let tpos = |t: &BlockId| -> Result<usize, Bail> {
            pos.get(t)
                .copied()
                .ok_or_else(|| format!("jump to unreachable block b{}", t.index()))
        };
        {
            let inst = &self.body.instructions[iid];
            let ix = iid.index() as u32;
            let live = self.class.contains_key(&ix) || self.wused.contains(&ix);
            match inst {
                // aliases and merge-points emit no code of their own
                Inst::Constant(_) | Inst::GetLocal(_) | Inst::Phi(_) => return Ok(()),
                // dead pure producers emit nothing — a fault on a dead
                // value is elided (same dst-gated skip as both Op lanes)
                Inst::NewArray
                | Inst::NewDict
                | Inst::GetField { .. }
                | Inst::Len(_)
                | Inst::ToFloat(_)
                | Inst::Sqrt(_)
                | Inst::In(..)
                | Inst::Format(_)
                | Inst::MakeClosure { .. }
                | Inst::NewInstance { .. }
                | Inst::IsInstance { .. }
                | Inst::IsRaised(_)
                | Inst::GetEntry(_)
                | Inst::RefBody(_)
                | Inst::BinOp { .. }
                | Inst::UnaryOp { .. }
                    if !live =>
                {
                    return Ok(());
                }
                _ => {}
            }
            match inst {
                    Inst::SetLocal(l, v) => {
                        let lx = l.index() as u32;
                        if self.is_entry && self.entry_locals.contains(&lx) {
                            // mirror entry-shared locals into the memory
                            // region (and the wasm slot too, if something
                            // reads it — ForNext touches the slot directly)
                            if let Some(&slot) = self.llocal.get(&lx) {
                                let lk =
                                    self.lclass.get(&lx).copied().unwrap_or(K::Word);
                                self.get(*v, lk)?;
                                self.ins(Instruction::LocalSet(slot));
                            }
                            self.ins(Instruction::I32Const((self.entry_base + lx * 8) as i32));
                            self.get(*v, K::Word)?;
                            self.ins(Instruction::I64Store(mem_arg(0, 3)));
                        } else if let Some(&slot) = self.llocal.get(&lx) {
                            // a store into a local nobody reads is dead;
                            // scalar locals take their class, word locals
                            // the raw word
                            let lk = self.lclass.get(&lx).copied().unwrap_or(K::Word);
                            self.get(*v, lk)?;
                            self.ins(Instruction::LocalSet(slot));
                        }
                    }
                    Inst::BinOp {
                        left,
                        op,
                        right,
                        kind,
                    } => {
                        self.emit_bin(iid, *left, *op, *right, *kind)?;
                    }
                    Inst::UnaryOp { op, right } => {
                        match op {
                            // class off the dst's readers, else the operand's
                            UnaryOp::Negative | UnaryOp::Positive => {
                                let dk = self
                                    .class
                                    .get(&ix)
                                    .or_else(|| self.class.get(&(right.index() as u32)))
                                    .copied()
                                    .unwrap_or(K::Float);
                                match (op, dk) {
                                    // `-x` traps on i64::MIN (checked neg)
                                    (UnaryOp::Negative, K::Int) => {
                                        self.ins(Instruction::I64Const(0));
                                        self.get(*right, K::Int)?;
                                        self.ins(Instruction::I64Sub);
                                        self.ins(Instruction::LocalSet(self.tmp));
                                        self.get(*right, K::Int)?;
                                        self.ins(Instruction::I64Const(i64::MIN));
                                        self.ins(Instruction::I64Eq);
                                        self.trap_if();
                                        self.ins(Instruction::LocalGet(self.tmp));
                                        self.store_dst(iid, K::Int)?;
                                    }
                                    (UnaryOp::Negative, K::Float) => {
                                        self.get(*right, K::Float)?;
                                        self.ins(Instruction::F64Neg);
                                        self.store_dst(iid, K::Float)?;
                                    }
                                    // `+x` is checked abs on ints, fabs on
                                    // floats — it is NOT a no-op
                                    (UnaryOp::Positive, K::Int) => {
                                        self.get(*right, K::Int)?;
                                        self.ins(Instruction::LocalSet(self.tmp));
                                        self.ins(Instruction::LocalGet(self.tmp));
                                        self.ins(Instruction::I64Const(i64::MIN));
                                        self.ins(Instruction::I64Eq);
                                        self.trap_if();
                                        self.ins(Instruction::LocalGet(self.tmp));
                                        self.ins(Instruction::I64Const(0));
                                        self.ins(Instruction::I64LtS);
                                        self.ins(Instruction::If(BlockType::Result(ValType::I64)));
                                        self.ins(Instruction::I64Const(0));
                                        self.ins(Instruction::LocalGet(self.tmp));
                                        self.ins(Instruction::I64Sub);
                                        self.ins(Instruction::Else);
                                        self.ins(Instruction::LocalGet(self.tmp));
                                        self.ins(Instruction::End);
                                        self.store_dst(iid, K::Int)?;
                                    }
                                    (UnaryOp::Positive, K::Float) => {
                                        self.get(*right, K::Float)?;
                                        self.ins(Instruction::F64Abs);
                                        self.store_dst(iid, K::Float)?;
                                    }
                                    _ => bail!("unary {op:?} on {dk:?}"),
                                }
                            }
                            UnaryOp::Not => {
                                self.get(*right, K::Bool)?;
                                self.ins(Instruction::I32Eqz);
                                self.store_dst(iid, K::Bool)?;
                            }
                            UnaryOp::BitwiseNot => {
                                self.get(*right, K::Int)?;
                                self.ins(Instruction::I64Const(-1));
                                self.ins(Instruction::I64Xor);
                                self.store_dst(iid, K::Int)?;
                            }
                        }
                    }
                    Inst::Jump { target } => {
                        let mut t = *target;
                        if let Some(&nt) =
                            self.rt.get(&(bid.index() as u32, t.index() as u32))
                        {
                            // tail-duplicated edge: the dup'd block's phis take
                            // this pred's copies, its insts run inline, its
                            // successor's phis take its copies — then jump there
                            let nt = BlockId::from(nt);
                            self.emit_copies(bid, t)?;
                            self.emit_inline(t, bi, pos)?;
                            self.emit_copies(t, nt)?;
                            t = nt;
                        } else {
                            self.emit_copies(bid, t)?;
                        }
                        // a jump to the next block is a fallthrough
                        let ti = tpos(&t)?;
                        if ti != bi + 1 {
                            let kind = if ti <= bi {
                                ScopeKind::Loop
                            } else {
                                ScopeKind::Block
                            };
                            self.br(ti, kind)?;
                        }
                    }
                    Inst::JumpIfFalse { condition, target } => {
                        if let Some(&nt) = self
                            .rt
                            .get(&(bid.index() as u32, target.index() as u32))
                        {
                            // same tail duplication on the taken (false) arm —
                            // an ad-hoc if/else frames the inlined tail
                            let nt = BlockId::from(nt);
                            self.get(*condition, K::Bool)?;
                            self.ins(Instruction::If(BlockType::Empty));
                            self.stack.push(Scope {
                                open: bi,
                                close: usize::MAX,
                                kind: ScopeKind::Block,
                                target: usize::MAX,
                            });
                            self.ins(Instruction::Else);
                            self.emit_copies(bid, *target)?;
                            self.emit_inline(*target, bi, pos)?;
                            self.emit_copies(*target, nt)?;
                            let ti = tpos(&nt)?;
                            let kind = if ti <= bi {
                                ScopeKind::Loop
                            } else {
                                ScopeKind::Block
                            };
                            self.br(ti, kind)?;
                            self.ins(Instruction::End);
                            self.stack.pop();
                        } else {
                            self.get(*condition, K::Bool)?;
                            let ti = tpos(target)?;
                            let kind = if ti <= bi {
                                ScopeKind::Loop
                            } else {
                                ScopeKind::Block
                            };
                            self.br_if(ti, kind, false)?;
                        }
                    }
                    Inst::ForNext { idx, bound, target } => {
                        let Inst::GetLocal(l) = &self.body.instructions[*idx] else {
                            bail!("for_next idx not a local read");
                        };
                        let lw = *self
                            .llocal
                            .get(&(l.index() as u32))
                            .ok_or("for_next idx local unclassed")?;
                        self.ins(Instruction::LocalGet(lw));
                        self.ins(Instruction::I64Const(1));
                        self.ins(Instruction::I64Add);
                        self.ins(Instruction::LocalSet(lw));
                        self.ins(Instruction::LocalGet(lw));
                        self.get(*bound, K::Int)?;
                        self.ins(Instruction::I64LtS);
                        self.br_if(tpos(target)?, ScopeKind::Loop, true)?;
                    }
                    Inst::Switch {
                        scrut,
                        base,
                        table,
                        default,
                    } => {
                        self.get(*scrut, K::Int)?;
                        if *base != 0 {
                            self.ins(Instruction::I64Const(*base as i64));
                            self.ins(Instruction::I64Sub);
                        }
                        self.ins(Instruction::I32WrapI64);
                        let mut tbl = Vec::with_capacity(table.len());
                        for t in table {
                            let ti = tpos(t)?;
                            let kind = if ti <= bi {
                                ScopeKind::Loop
                            } else {
                                ScopeKind::Block
                            };
                            tbl.push(self.depth(ti, kind)?);
                        }
                        let td = tpos(default)?;
                        let kind = if td <= bi {
                            ScopeKind::Loop
                        } else {
                            ScopeKind::Block
                        };
                        let d = self.depth(td, kind)?;
                        self.ins(Instruction::BrTable(tbl.into(), d));
                    }
                    Inst::CallDirect { body: b, args } => {
                        self.emit_call(iid, b.index(), args)?;
                    }
                    Inst::Call { callee, args } => {
                        // resolvable + arity-matching → direct call; anything
                        // else goes through the callee's closure object and a
                        // uniform-signature trampoline via call_indirect
                        let direct = self
                            .callee
                            .get(&ix)
                            .map(|&b| {
                                self.sigs[b]
                                    .as_ref()
                                    .is_some_and(|s| s.params.len() == args.len())
                            })
                            .unwrap_or(false);
                        if direct {
                            let b = self.callee[&ix];
                            self.emit_call(iid, b, args)?;
                        } else {
                            self.get(*callee, K::Word)?;
                            self.ins(Instruction::LocalSet(self.tmp));
                            self.ins(Instruction::LocalGet(self.tmp));
                            self.ins(Instruction::I32WrapI64);
                            self.ins(Instruction::LocalSet(self.hp));
                            // callee must be a tagged closure object
                            self.ins(Instruction::LocalGet(self.hp));
                            self.ins(Instruction::I32Eqz);
                            self.ins(Instruction::If(BlockType::Empty));
                            self.ins(Instruction::Unreachable);
                            self.ins(Instruction::End);
                            self.ins(Instruction::LocalGet(self.hp));
                            self.ins(Instruction::I32Load(mem_arg(0, 2)));
                            self.ins(Instruction::I32Const(TAG_CLOSURE as i32));
                            self.ins(Instruction::I32Ne);
                            self.trap_if();
                            // marshal args to i64 words in the call scratch
                            for (i, a) in args.iter().enumerate() {
                                self.ins(Instruction::I32Const(
                                    (self.ctx.call_scratch + i as u32 * 8) as i32,
                                ));
                                self.get(*a, K::Word)?;
                                self.ins(Instruction::I64Store(mem_arg(0, 3)));
                            }
                            // call_indirect(env, args_ptr, nargs) -> i64
                            self.ins(Instruction::LocalGet(self.hp));
                            self.ins(Instruction::I32Const(self.ctx.call_scratch as i32));
                            self.ins(Instruction::I32Const(args.len() as i32));
                            self.ins(Instruction::LocalGet(self.hp));
                            self.ins(Instruction::I32Load(mem_arg(4, 2)));
                            self.ins(Instruction::CallIndirect {
                                type_index: self.tramp_ty,
                                table_index: 0,
                            });
                            self.store_dst(iid, K::Word)?;
                        }
                    }
                    Inst::CallNative { id, args } => {
                        let ni = id.index() as u32;
                        // dst class — single-class insts keep it, word-slot
                        // readers get a raw word, undemanded results drop
                        let dk = self
                            .class
                            .get(&ix)
                            .copied()
                            .or_else(|| self.wused.contains(&ix).then_some(K::Word));
                        // pure float natives inline — an import crossing is
                        // ~100x an f64 op. SSA classes are exact here (each
                        // inst has one producer class), so `get` always
                        // coerces; only a dead result falls back to import.
                        if let Some(&mop) = self.math.get(&ni) {
                            use crate::MathOp::*;
                            if dk.is_some() && (mop == Identity || args.len() <= 2) {
                                let k = dk.unwrap();
                                match mop {
                                    Identity => self.get(args[0], k)?,
                                    _ => {
                                        for a in args.iter().take(2) {
                                            self.get(*a, K::Float)?;
                                        }
                                        self.ins(match mop {
                                            Abs => Instruction::F64Abs,
                                            Min => Instruction::F64Min,
                                            Max => Instruction::F64Max,
                                            Floor => Instruction::F64Floor,
                                            Identity => unreachable!(),
                                        });
                                    }
                                }
                                self.store_dst(
                                    iid,
                                    if mop == Identity { k } else { K::Float },
                                )?;
                                return Ok(());
                            }
                        }
                        // cov::* sink natives record into linear memory —
                        // the host drains __covbuf/__ptmap/__dcov per frame.
                        if let Some(&kind) = self.sink.get(&ni) {
                            let Some(cov) = self.cov else {
                                bail!("sink native {ni} without cov layout");
                            };
                            use crate::SinkKind::*;
                            // w(a): arg as a raw i32 word
                            let w = |em: &mut Self, a: InstId| -> Result<(), Bail> {
                                em.get(a, K::Word)?;
                                em.ins(Instruction::I32WrapI64);
                                Ok(())
                            };
                            match kind {
                                // u8[ptmap + p] = 1 — ~70% of probe traffic
                                Point | PointPass => {
                                    self.ins(Instruction::I32Const(
                                        cov.ptmap_base as i32,
                                    ));
                                    w(self, args[0])?;
                                    self.ins(Instruction::I32Add);
                                    self.ins(Instruction::I32Const(1));
                                    self.ins(Instruction::I32Store8(mem_arg(0, 0)));
                                }
                                // begin(d): leafbits[d] = 0, result const 1
                                Begin => {
                                    w(self, args[0])?;
                                    self.ins(Instruction::Call(cov.h[1]));
                                }
                                // lhs/rhs(v): push (f64 value, numeric flag)
                                // — numeric iff the operand class is Int/Float
                                Leaf => {
                                    let a = args[0];
                                    let cls = self.stored_k(a);
                                    let num =
                                        matches!(cls, K::Int | K::Float);
                                    if cls == K::Float {
                                        self.get(a, K::Float)?;
                                    } else if num {
                                        self.get(a, K::Word)?;
                                        self.ins(Instruction::F64ConvertI64S);
                                    } else {
                                        self.ins(Instruction::F64Const(
                                            0.0f64.into(),
                                        ));
                                    }
                                    self.ins(Instruction::I32Const(num as i32));
                                    self.ins(Instruction::Call(cov.h[0]));
                                }
                                Cond => {
                                    for &a in args.iter().take(3) {
                                        w(self, a)?;
                                    }
                                    self.ins(Instruction::Call(cov.h[2]));
                                }
                                Cmp => {
                                    for &a in args.iter().take(4) {
                                        w(self, a)?;
                                    }
                                    self.ins(Instruction::Call(cov.h[3]));
                                }
                                // dec(d, v): helper writes (d, v, leafbits)
                                Dec => {
                                    self.ins(Instruction::I32Const(ni as i32));
                                    for &a in args.iter().take(2) {
                                        w(self, a)?;
                                    }
                                    self.ins(Instruction::Call(cov.h[4]));
                                }
                                // the generic 64B record kinds would corrupt
                                // the 16B record drain — bail to the interp
                                Hit | Passthru => {
                                    bail!("generic Hit/Passthru sink kinds would corrupt the 16-byte dec ring");
                                }
                            }
                            // sink_result: hit/pass→null, begin→const 1,
                            // the rest passthru their last arg at dk
                            match kind {
                                Point => {
                                    self.ins(Instruction::I64Const(0));
                                    self.store_dst(iid, K::Word)?;
                                }
                                PointPass | Leaf | Cond | Cmp | Dec => {
                                    let last = *args.last().unwrap();
                                    let k = dk.unwrap_or(K::Int);
                                    self.get(last, k)?;
                                    self.store_dst(iid, k)?;
                                }
                                Begin => {
                                    let k = dk.unwrap_or(K::Int);
                                    match k {
                                        K::Float => self.ins(
                                            Instruction::F64Const(1.0f64.into()),
                                        ),
                                        K::Bool => {
                                            self.ins(Instruction::I32Const(1))
                                        }
                                        _ => self.ins(Instruction::I64Const(1)),
                                    }
                                    self.store_dst(iid, k)?;
                                }
                                Hit | Passthru => unreachable!(),
                            }
                            return Ok(());
                        }
                        // classless args pass as raw words — the import sig
                        // is per-call-site so the host sees the same ABI
                        let mut params = Vec::with_capacity(args.len());
                        for a in args {
                            let k = self.stored_k(*a);
                            params.push(k);
                            self.get(*a, k)?;
                        }
                        let ret = if self.class.contains_key(&ix) {
                            self.class[&ix]
                        } else if self.wused.contains(&ix) {
                            K::Word
                        } else {
                            K::Int
                        };
                        let fi = self.natives[&(id.index() as u32, params, Some(ret))];
                        self.ins(Instruction::Call(fi));
                        self.store_dst(iid, ret)?;
                    }
                    // ---------- heap ops: bump-arena objects in linear memory ----------
                    // every object leads with a u32 tag so a host walking
                    // __hp memory can decode it — the "one tag word per
                    // object" contract:
                    //   array    = [tag u32][data u32][len u32][cap u32] (16B)
                    //   string   = same header, data → byte buffer
                    //   instance = [tag u32][adt u32][field words @ +8]
                    //   dict     = [tag u32][size u32][bcap u32][buckets u32]
                    //              [ecap u32][order u32] — buckets are open-
                    //              addressed [key,val,state] slots for lookup;
                    //              order is an insertion-ordered key array so
                    //              d[i] matches the interpreter's IndexMap
                    //   closure  = [tag u32][table idx u32][ncaps u64][caps…]
                    Inst::NewArray => {
                        self.ins(Instruction::I32Const(16));
                        self.alloc();
                        self.ins(Instruction::LocalSet(self.hp));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Const(TAG_ARRAY as i32));
                        self.ins(Instruction::I32Store(mem_arg(0, 2)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Const(0));
                        self.ins(Instruction::I32Store(mem_arg(4, 2)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I64Const(0));
                        self.ins(Instruction::I64Store(mem_arg(8, 3)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I64ExtendI32U);
                        self.store_dst(iid, K::Word)?;
                    }
                    Inst::Push { array, value } => {
                        // hp = array header; tb = len
                        self.get(*array, K::Word)?;
                        self.ins(Instruction::I32WrapI64);
                        self.ins(Instruction::LocalSet(self.hp));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(8, 2)));
                        self.ins(Instruction::LocalSet(self.tb));
                        // grow when len == cap — newcap = cap ? cap*2 : 4
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(8, 2)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(12, 2)));
                        self.ins(Instruction::I32Eq);
                        self.ins(Instruction::If(BlockType::Empty));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(12, 2)));
                        self.ins(Instruction::I32Eqz);
                        self.ins(Instruction::If(BlockType::Result(ValType::I32)));
                        self.ins(Instruction::I32Const(4));
                        self.ins(Instruction::Else);
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(12, 2)));
                        self.ins(Instruction::I32Const(2));
                        self.ins(Instruction::I32Mul);
                        self.ins(Instruction::End);
                        self.ins(Instruction::LocalSet(self.tc));
                        // new data block: copy old elements across
                        self.ins(Instruction::LocalGet(self.tc));
                        self.ins(Instruction::I32Const(8));
                        self.ins(Instruction::I32Mul);
                        self.alloc();
                        self.ins(Instruction::LocalSet(self.sz));
                        self.ins(Instruction::LocalGet(self.sz));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(4, 2)));
                        self.ins(Instruction::LocalGet(self.tb));
                        self.ins(Instruction::I32Const(8));
                        self.ins(Instruction::I32Mul);
                        self.ins(Instruction::MemoryCopy {
                            src_mem: 0,
                            dst_mem: 0,
                        });
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::LocalGet(self.sz));
                        self.ins(Instruction::I32Store(mem_arg(4, 2)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::LocalGet(self.tc));
                        self.ins(Instruction::I32Store(mem_arg(12, 2)));
                        self.ins(Instruction::End);
                        // data[len] = value word; len += 1
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(4, 2)));
                        self.ins(Instruction::LocalGet(self.tb));
                        self.ins(Instruction::I32Const(8));
                        self.ins(Instruction::I32Mul);
                        self.ins(Instruction::I32Add);
                        self.get(*value, K::Word)?;
                        self.ins(Instruction::I64Store(mem_arg(0, 3)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::LocalGet(self.tb));
                        self.ins(Instruction::I32Const(1));
                        self.ins(Instruction::I32Add);
                        self.ins(Instruction::I32Store(mem_arg(8, 2)));
                    }
                    Inst::GetIndex { set, index, kind } => {
                        self.direct(kind)?;
                        self.get(*set, K::Word)?;
                        self.ins(Instruction::I32WrapI64);
                        self.ins(Instruction::LocalSet(self.hp));
                        // tag dispatch: str → char object, dict → slot val,
                        // anything else → word-element array
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(0, 2)));
                        self.ins(Instruction::I32Const(TAG_STR as i32));
                        self.ins(Instruction::I32Eq);
                        self.ins(Instruction::If(BlockType::Result(ValType::I64)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.get(*index, K::Word)?;
                        self.ins(Instruction::I32WrapI64);
                        self.ins(Instruction::Call(self.helpers[H_STR_CHAR]));
                        self.ins(Instruction::I64ExtendI32U);
                        self.ins(Instruction::Else);
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(0, 2)));
                        self.ins(Instruction::I32Const(TAG_DICT as i32));
                        self.ins(Instruction::I32Eq);
                        self.ins(Instruction::If(BlockType::Result(ValType::I64)));
                        if self.stored_k(*index) == K::Int {
                            // d[i] → i-th inserted entry as a [key,val] pair
                            self.ins(Instruction::LocalGet(self.hp));
                            self.get(*index, K::Word)?;
                            self.ins(Instruction::I32WrapI64);
                            self.ins(Instruction::Call(self.helpers[H_DICT_ENTRY]));
                            self.ins(Instruction::I64ExtendI32U);
                        } else {
                            // d[key] → val or Null(0) on miss (interp: unwrap_or)
                            self.ins(Instruction::LocalGet(self.hp));
                            self.get(*index, K::Word)?;
                            self.ins(Instruction::Call(self.helpers[H_DICT_FIND]));
                            self.ins(Instruction::LocalSet(self.sz));
                            self.ins(Instruction::LocalGet(self.sz));
                            self.ins(Instruction::I32Eqz);
                            self.ins(Instruction::If(BlockType::Result(ValType::I64)));
                            self.ins(Instruction::I64Const(0));
                            self.ins(Instruction::Else);
                            self.ins(Instruction::LocalGet(self.sz));
                            self.ins(Instruction::I64Load(mem_arg(8, 3)));
                            self.ins(Instruction::End);
                        }
                        self.ins(Instruction::Else);
                        // array: idx >= len → trap, then data[idx]
                        self.get(*index, K::Word)?;
                        self.ins(Instruction::I32WrapI64);
                        self.ins(Instruction::LocalSet(self.tb));
                        self.ins(Instruction::LocalGet(self.tb));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(8, 2)));
                        self.ins(Instruction::I32GeU);
                        self.trap_if();
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(4, 2)));
                        self.ins(Instruction::LocalGet(self.tb));
                        self.ins(Instruction::I32Const(8));
                        self.ins(Instruction::I32Mul);
                        self.ins(Instruction::I32Add);
                        self.ins(Instruction::I64Load(mem_arg(0, 3)));
                        self.ins(Instruction::End);
                        self.ins(Instruction::End);
                        self.store_dst(iid, K::Word)?;
                    }
                    Inst::SetIndex {
                        set,
                        index,
                        value,
                    } => {
                        self.get(*set, K::Word)?;
                        self.ins(Instruction::I32WrapI64);
                        self.ins(Instruction::LocalSet(self.hp));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(0, 2)));
                        self.ins(Instruction::I32Const(TAG_DICT as i32));
                        self.ins(Instruction::I32Eq);
                        self.ins(Instruction::If(BlockType::Empty));
                        // d[key] = v through the dict helper
                        self.ins(Instruction::LocalGet(self.hp));
                        self.get(*index, K::Word)?;
                        self.get(*value, K::Word)?;
                        self.ins(Instruction::Call(self.helpers[H_DICT_SET]));
                        self.ins(Instruction::Else);
                        // array: bounds check then data[idx] = v
                        self.get(*index, K::Word)?;
                        self.ins(Instruction::I32WrapI64);
                        self.ins(Instruction::LocalSet(self.tb));
                        self.ins(Instruction::LocalGet(self.tb));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(8, 2)));
                        self.ins(Instruction::I32GeU);
                        self.trap_if();
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(4, 2)));
                        self.ins(Instruction::LocalGet(self.tb));
                        self.ins(Instruction::I32Const(8));
                        self.ins(Instruction::I32Mul);
                        self.ins(Instruction::I32Add);
                        self.get(*value, K::Word)?;
                        self.ins(Instruction::I64Store(mem_arg(0, 3)));
                        self.ins(Instruction::End);
                    }
                    // arrays and strings share the +8 len field; a dict's
                    // count is its +4 size word
                    Inst::Len(src) => {
                        self.get(*src, K::Word)?;
                        self.ins(Instruction::I32WrapI64);
                        self.ins(Instruction::LocalSet(self.hp));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(0, 2)));
                        self.ins(Instruction::I32Const(TAG_DICT as i32));
                        self.ins(Instruction::I32Eq);
                        self.ins(Instruction::If(BlockType::Result(ValType::I32)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(4, 2)));
                        self.ins(Instruction::Else);
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(8, 2)));
                        self.ins(Instruction::End);
                        self.ins(Instruction::I64ExtendI32U);
                        self.store_dst(iid, K::Int)?;
                    }
                    Inst::NewInstance { adt, fields } => {
                        let n = fields.len() as i32;
                        self.ins(Instruction::I32Const(8 + n * 8));
                        self.alloc();
                        self.ins(Instruction::LocalSet(self.hp));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Const(TAG_INSTANCE as i32));
                        self.ins(Instruction::I32Store(mem_arg(0, 2)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Const(adt.index() as i32));
                        self.ins(Instruction::I32Store(mem_arg(4, 2)));
                        for (fi, f) in fields.iter().enumerate() {
                            self.ins(Instruction::LocalGet(self.hp));
                            self.get(*f, K::Word)?;
                            self.ins(Instruction::I64Store(mem_arg(8 + fi as u32 * 8, 3)));
                        }
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I64ExtendI32U);
                        self.store_dst(iid, K::Word)?;
                    }
                    Inst::GetField { src, slot, kind } => {
                        self.direct(kind)?;
                        self.get(*src, K::Word)?;
                        self.ins(Instruction::I32WrapI64);
                        self.ins(Instruction::I64Load(mem_arg(8 + *slot * 8, 3)));
                        self.store_dst(iid, K::Word)?;
                    }
                    Inst::SetField {
                        receiver,
                        slot,
                        value,
                    } => {
                        self.get(*receiver, K::Word)?;
                        self.ins(Instruction::I32WrapI64);
                        self.get(*value, K::Word)?;
                        self.ins(Instruction::I64Store(mem_arg(8 + *slot * 8, 3)));
                    }
                    Inst::IsInstance { src, adt } => {
                        // word 0 is Null — never an instance; anything else
                        // is solver-guaranteed a heap object, tag at +0 and
                        // the adt id one word in
                        self.get(*src, K::Word)?;
                        self.ins(Instruction::I32WrapI64);
                        self.ins(Instruction::LocalSet(self.hp));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Eqz);
                        self.ins(Instruction::If(BlockType::Result(ValType::I32)));
                        self.ins(Instruction::I32Const(0));
                        self.ins(Instruction::Else);
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(0, 2)));
                        self.ins(Instruction::I32Const(TAG_INSTANCE as i32));
                        self.ins(Instruction::I32Eq);
                        self.ins(Instruction::If(BlockType::Result(ValType::I32)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Load(mem_arg(4, 2)));
                        self.ins(Instruction::I32Const(adt.index() as i32));
                        self.ins(Instruction::I32Eq);
                        self.ins(Instruction::Else);
                        self.ins(Instruction::I32Const(0));
                        self.ins(Instruction::End);
                        self.ins(Instruction::End);
                        self.store_dst(iid, K::Bool)?;
                    }
                    Inst::In(needle, haystack, condition) => {
                        // tag dispatch on the haystack: dict → key probe,
                        // str → substring scan, else → word-element scan
                        self.get(*needle, K::Word)?;
                        self.ins(Instruction::LocalSet(self.tmp));
                        self.get(*haystack, K::Word)?;
                        self.ins(Instruction::I32WrapI64);
                        self.ins(Instruction::LocalSet(self.sz));
                        self.ins(Instruction::LocalGet(self.sz));
                        self.ins(Instruction::I32Load(mem_arg(0, 2)));
                        self.ins(Instruction::I32Const(TAG_DICT as i32));
                        self.ins(Instruction::I32Eq);
                        self.ins(Instruction::If(BlockType::Result(ValType::I32)));
                        self.ins(Instruction::LocalGet(self.sz));
                        self.ins(Instruction::LocalGet(self.tmp));
                        self.ins(Instruction::Call(self.helpers[H_DICT_FIND]));
                        self.ins(Instruction::I32Const(0));
                        self.ins(Instruction::I32Ne);
                        self.ins(Instruction::Else);
                        self.ins(Instruction::LocalGet(self.sz));
                        self.ins(Instruction::I32Load(mem_arg(0, 2)));
                        self.ins(Instruction::I32Const(TAG_STR as i32));
                        self.ins(Instruction::I32Eq);
                        self.ins(Instruction::If(BlockType::Result(ValType::I32)));
                        self.ins(Instruction::LocalGet(self.tmp));
                        self.ins(Instruction::I32WrapI64);
                        self.ins(Instruction::LocalGet(self.sz));
                        self.ins(Instruction::Call(self.helpers[H_STR_IN]));
                        self.ins(Instruction::Else);
                        // array scan — element compare through __key_eq so
                        // str elements match by content, not pointer
                        self.ins(Instruction::LocalGet(self.sz));
                        self.ins(Instruction::I32Load(mem_arg(8, 2)));
                        self.ins(Instruction::LocalSet(self.tb));
                        self.ins(Instruction::I32Const(0));
                        self.ins(Instruction::LocalSet(self.tc));
                        self.ins(Instruction::I32Const(0));
                        self.ins(Instruction::LocalSet(self.hp));
                        self.ins(Instruction::Block(BlockType::Empty));
                        self.ins(Instruction::Loop(BlockType::Empty));
                        self.ins(Instruction::LocalGet(self.tc));
                        self.ins(Instruction::LocalGet(self.tb));
                        self.ins(Instruction::I32GeU);
                        self.ins(Instruction::BrIf(1));
                        self.ins(Instruction::LocalGet(self.sz));
                        self.ins(Instruction::I32Load(mem_arg(4, 2)));
                        self.ins(Instruction::LocalGet(self.tc));
                        self.ins(Instruction::I32Const(3));
                        self.ins(Instruction::I32Shl);
                        self.ins(Instruction::I32Add);
                        self.ins(Instruction::I64Load(mem_arg(0, 3)));
                        self.ins(Instruction::LocalGet(self.tmp));
                        self.ins(Instruction::Call(self.helpers[H_KEY_EQ]));
                        self.ins(Instruction::If(BlockType::Empty));
                        self.ins(Instruction::I32Const(1));
                        self.ins(Instruction::LocalSet(self.hp));
                        self.ins(Instruction::Br(2));
                        self.ins(Instruction::End);
                        self.ins(Instruction::LocalGet(self.tc));
                        self.ins(Instruction::I32Const(1));
                        self.ins(Instruction::I32Add);
                        self.ins(Instruction::LocalSet(self.tc));
                        self.ins(Instruction::Br(0));
                        self.ins(Instruction::End);
                        self.ins(Instruction::End);
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::End);
                        self.ins(Instruction::End);
                        if !*condition {
                            self.ins(Instruction::I32Eqz);
                        }
                        self.store_dst(iid, K::Bool)?;
                    }
                    // ---------- error-as-value ops: Raised words are negative ----------
                    Inst::Unwrap(src) => {
                        // dst = src word; Null (0) / Raised (neg) → trap
                        self.get(*src, K::Word)?;
                        self.ins(Instruction::LocalSet(self.tmp));
                        self.ins(Instruction::LocalGet(self.tmp));
                        self.ins(Instruction::I64Const(0));
                        self.ins(Instruction::I64LeS);
                        self.trap_if();
                        self.ins(Instruction::LocalGet(self.tmp));
                        self.store_dst(iid, K::Word)?;
                    }
                    Inst::UnwrapUnit(src) => {
                        // dst = Null; only Raised faults
                        self.get(*src, K::Word)?;
                        self.ins(Instruction::I64Const(0));
                        self.ins(Instruction::I64LtS);
                        self.trap_if();
                        self.ins(Instruction::I64Const(0));
                        self.store_dst(iid, K::Word)?;
                    }
                    Inst::UnwrapRaised(src) => {
                        // payload = low 63 bits of the raised word
                        self.get(*src, K::Word)?;
                        self.ins(Instruction::I64Const(0x7fff_ffff_ffff_ffffu64 as i64));
                        self.ins(Instruction::I64And);
                        self.store_dst(iid, K::Word)?;
                    }
                    Inst::IsRaised(src) => {
                        self.get(*src, K::Word)?;
                        self.ins(Instruction::I64Const(0));
                        self.ins(Instruction::I64LtS);
                        self.store_dst(iid, K::Bool)?;
                    }
                    // a raise/panic inside a wasm body aborts the export —
                    // the host sees a trap, matching an unwound fault
                    Inst::Raise(_) | Inst::Panic => {
                        self.ins(Instruction::Unreachable);
                    }
                    // f-strings: each part formats to a tagged str object,
                    // folded left-to-right through __str_cat
                    Inst::Format(parts) => {
                        let mut first = true;
                        for p in parts {
                            match p {
                                FormatPart::Literal(s) => {
                                    let a = self
                                        .ctx
                                        .str_objs
                                        .get(&(s.index() as u32))
                                        .copied()
                                        .ok_or("format literal not laid out")?;
                                    self.ins(Instruction::I32Const(a as i32));
                                }
                                FormatPart::Value(v) => {
                                    self.format_part(*v)?;
                                }
                            }
                            if first {
                                first = false;
                            } else {
                                self.ins(Instruction::Call(self.helpers[H_STR_CAT]));
                            }
                        }
                        if first {
                            self.ins(Instruction::I32Const(self.ctx.obj_obj as i32));
                        }
                        self.ins(Instruction::I64ExtendI32U);
                        self.store_dst(iid, K::Word)?;
                    }
                    Inst::NewDict => {
                        self.ins(Instruction::I32Const(24));
                        self.alloc();
                        self.ins(Instruction::LocalSet(self.hp));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Const(TAG_DICT as i32));
                        self.ins(Instruction::I32Store(mem_arg(0, 2)));
                        for off in [4u32, 12] {
                            self.ins(Instruction::LocalGet(self.hp));
                            self.ins(Instruction::I64Const(0));
                            self.ins(Instruction::I64Store(mem_arg(off, 3)));
                        }
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Const(0));
                        self.ins(Instruction::I32Store(mem_arg(20, 2)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I64ExtendI32U);
                        self.store_dst(iid, K::Word)?;
                    }
                    // `d.key = v` — the key is a static str object
                    Inst::Insert { dict, key, value } => {
                        self.get(*dict, K::Word)?;
                        self.ins(Instruction::I32WrapI64);
                        let a = self
                            .ctx
                            .str_objs
                            .get(&(key.index() as u32))
                            .copied()
                            .ok_or("insert key not laid out")?;
                        self.ins(Instruction::I64Const(a as i64));
                        self.get(*value, K::Word)?;
                        self.ins(Instruction::Call(self.helpers[H_DICT_SET]));
                    }
                    Inst::RefBody(t) => {
                        let b = t.index();
                        if !self.emitted[b] {
                            bail!("RefBody to skipped body {b}");
                        }
                        let ti = *self.tramp.get(&b).ok_or("no trampoline for body")?;
                        self.ins(Instruction::I32Const(16));
                        self.alloc();
                        self.ins(Instruction::LocalSet(self.hp));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Const(TAG_CLOSURE as i32));
                        self.ins(Instruction::I32Store(mem_arg(0, 2)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Const(ti as i32));
                        self.ins(Instruction::I32Store(mem_arg(4, 2)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I64Const(0));
                        self.ins(Instruction::I64Store(mem_arg(8, 3)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I64ExtendI32U);
                        self.store_dst(iid, K::Word)?;
                    }
                    Inst::MakeClosure { body: t, captures } => {
                        let b = t.index();
                        if !self.emitted[b] {
                            bail!("MakeClosure to skipped body {b}");
                        }
                        let ti = *self.tramp.get(&b).ok_or("no trampoline for body")?;
                        self.ins(Instruction::I32Const(16 + 8 * captures.len() as i32));
                        self.alloc();
                        self.ins(Instruction::LocalSet(self.hp));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Const(TAG_CLOSURE as i32));
                        self.ins(Instruction::I32Store(mem_arg(0, 2)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I32Const(ti as i32));
                        self.ins(Instruction::I32Store(mem_arg(4, 2)));
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I64Const(captures.len() as i64));
                        self.ins(Instruction::I64Store(mem_arg(8, 3)));
                        for (i, c) in captures.iter().enumerate() {
                            self.ins(Instruction::LocalGet(self.hp));
                            self.get(*c, K::Word)?;
                            self.ins(Instruction::I64Store(mem_arg(16 + i as u32 * 8, 3)));
                        }
                        self.ins(Instruction::LocalGet(self.hp));
                        self.ins(Instruction::I64ExtendI32U);
                        self.store_dst(iid, K::Word)?;
                    }
                    // ---------- entry-frame locals: shared raw words ----------
                    // `GetEntry`/`SetEntry` address body 0's locals — a fixed
                    // memory region so any body can read module-level lets
                    Inst::GetEntry(l) => {
                        self.ins(Instruction::I32Const((self.entry_base + l.index() as u32 * 8) as i32));
                        self.ins(Instruction::I64Load(mem_arg(0, 3)));
                        self.store_dst(iid, K::Word)?;
                    }
                    Inst::SetEntry(l, v) => {
                        self.ins(Instruction::I32Const((self.entry_base + l.index() as u32 * 8) as i32));
                        self.get(*v, K::Word)?;
                        self.ins(Instruction::I64Store(mem_arg(0, 3)));
                    }
                    Inst::ToFloat(v) => {
                        self.get(*v, K::Int)?;
                        self.ins(Instruction::F64ConvertI64S);
                        self.store_dst(iid, K::Float)?;
                    }
                    Inst::Sqrt(v) => {
                        self.get(*v, K::Float)?;
                        self.ins(Instruction::F64Sqrt);
                        self.store_dst(iid, K::Float)?;
                    }
                    Inst::Return(v) => {
                        if let Some(k) = self.ret_k {
                            self.get(*v, k)?;
                        }
                        self.ins(Instruction::Return);
                    }
                    other => bail!("unsupported inst {other:?}"),
                }
        }
        Ok(())
    }
}

// ---------- module ----------

/// Emit the module from `ir`. Skipped bodies are reported, not fatal — the
/// caller keeps them on the interpreter lane.
/// The `(id, params, ret)` import key one `CallNative` emits under — args
/// pass their stored class (`Word` when classless), the result is the
/// inst's class, `Word` when only word-read, `Int` when dead.
pub(crate) fn native_key(body: &IrBody, ana: &AnaI, iid: InstId) -> (u32, Vec<K>, Option<K>) {
    let Inst::CallNative { id, args } = &body.instructions[iid] else {
        unreachable!()
    };
    let params = args
        .iter()
        .map(|a| match &body.instructions[*a] {
            Inst::GetLocal(l) => ana
                .lclass
                .get(&(l.index() as u32))
                .copied()
                .unwrap_or(K::Word),
            _ => ana
                .class
                .get(&(a.index() as u32))
                .copied()
                .unwrap_or(K::Word),
        })
        .collect();
    let ix = iid.index() as u32;
    let ret = Some(if let Some(&k) = ana.class.get(&ix) {
        k
    } else if ana.wused.contains(&ix) {
        K::Word
    } else {
        K::Int
    });
    (id.index() as u32, params, ret)
}

/// `sink`: `cov::*` natives recorded into the linear-memory sink instead of
/// imported (see [`crate::SinkKind`]) — the host drains via `__covbuf`/`__covp`.
/// `math`: pure float natives inlined as f64 ops. `cov_points`/`cov_decs` size
/// the `__ptmap` bitmap and `__dcov` decision cells. Passing them turns on the
/// Go-Explore memory contract (`__sp`/`__status`/`__ptmap`/…) that
/// `wgame::WasmGame` snapshots and drains.
pub fn emit_ir(
    ir: &Ir,
    strs: &StrInterner,
    opts: &Opts,
    sink: Option<&crate::CovSink>,
    math: Option<&crate::MathNatives>,
    cov_points: u32,
    cov_decs: u32,
) -> Result<Wasmgen, Bail> {
    let empty_sink = crate::CovSink::new();
    let sink = sink.unwrap_or(&empty_sink);
    let empty_math = crate::MathNatives::new();
    let math = math.unwrap_or(&empty_math);
    let nbodies = ir.bodies.len();
    // local indices addressed by GetEntry/SetEntry anywhere — those entry
    // locals live in a shared linear-memory region so cross-body module
    // state works (the resume lane can't emit a live LoadEntry at all)
    let mut entry_locals: HashSet<u32> = HashSet::new();
    for (_, body) in ir.bodies.iter() {
        for (_, block) in body.blocks.iter() {
            for &iid in &block.stream {
                match &body.instructions[iid] {
                    Inst::GetEntry(l) | Inst::SetEntry(l, _) => {
                        entry_locals.insert(l.index() as u32);
                    }
                    _ => {}
                }
            }
        }
    }
    let bodies: Vec<&IrBody> = ir.bodies.iter().map(|(_, b)| b).collect();
    let ninsts: Vec<usize> = bodies
        .iter()
        .map(|b| {
            b.blocks
                .iter()
                .map(|(_, bl)| bl.stream.len())
                .sum::<usize>()
        })
        .collect();

    // cross-body fixpoint on return classes (call dsts class off callee rets)
    let mut ret: Vec<Option<K>> = vec![None; nbodies];
    let mut anas: Vec<Option<AnaI>> = (0..nbodies).map(|_| None).collect();
    for _ in 0..16 {
        let mut stable = true;
        for b in 0..nbodies {
            match analyze_body(bodies[b], &ret) {
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

    // ---------- static objects: string constants live as tagged objects ----------
    // [tag u32][data u32][len u32][cap u32][bytes] laid out in a data
    // segment; the Str const's word is the object's absolute address. Also
    // emit the fixed display strings Format needs ("true"/"false"/"<obj>").
    let mut str_objs: HashMap<u32, u32> = HashMap::new();
    let mut statics: Vec<(u32, Vec<u8>)> = Vec::new();
    {
        let mut ids: Vec<u32> = Vec::new();
        for (_, body) in ir.bodies.iter() {
            for (_, block) in body.blocks.iter() {
                for &iid in &block.stream {
                    match &body.instructions[iid] {
                        Inst::Constant(Constant::Str(s)) => ids.push(s.index() as u32),
                        Inst::Format(parts) => {
                            for p in parts {
                                if let FormatPart::Literal(s) = p {
                                    ids.push(s.index() as u32);
                                }
                            }
                        }
                        Inst::Insert { key, .. } => ids.push(key.index() as u32),
                        _ => {}
                    }
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            str_objs.insert(id, 0);
        }
    }

    let mut skipped: Vec<Skip> = Vec::new();
    let mut sigs: Vec<Option<Sig>> = (0..nbodies).map(|_| None).collect();
    for b in 0..nbodies {
        let body = bodies[b];
        let Some(ana) = &anas[b] else {
            skipped.push(Skip {
                body: b,
                reason: "analysis failed".into(),
            });
            continue;
        };
        // params without a scalar class pass as raw words — heap handles,
        // strids, and payload words all fit the i64 ABI. A closure body's
        // captures prepend its params: the call_indirect trampoline unmarshal
        // them out of the env word before the real args.
        let mut params: Vec<K> = body
            .captures
            .iter()
            .map(|p| {
                ana.lclass
                    .get(&(p.index() as u32))
                    .copied()
                    .unwrap_or(K::Word)
            })
            .collect();
        params.extend(body.params.iter().map(|p| {
            ana.lclass
                .get(&(p.index() as u32))
                .copied()
                .unwrap_or(K::Word)
        }));
        sigs[b] = Some(Sig {
            params,
            ret: ana.ret,
        });
    }

    // ---------- static memory layout, fixed before trials emit ----------
    // The wgame snapshot contract (was resume.rs's layout):
    //   [0, nb*8)             pc_table — zeros; [0,__sp) is a snapshot range
    //   [nb*8, +entry_bytes)  entry locals — under __sp so snapshots image
    //                         module-level `let`s and live_runs seeds them
    //   [__sp, __sp+1MB)      stack hole — statics + scratch (immutable or
    //                         transient, so unsnapshotted is correct)
    //   [cov0, sink_base)     coverage: [op bitmap][ptmap][decv][opstk]
    //   [sink_base, +8MB)     dec record ring (only when the sink is used)
    //   [heap_base, ...)      __hp arena
    let cov_total: u32 = if opts.coverage {
        (0..nbodies)
            .filter(|b| sigs[*b].is_some())
            .map(|b| ninsts[b] as u32)
            .sum()
    } else {
        0
    };
    let entry_base = nbodies as u32 * 8;
    let entry_bytes = bodies[0].locals.len() as u32 * 8;
    let sp_init = entry_base + entry_bytes; // __sp init — [0,__sp) snapshots+seeds
    let mut off = (sp_init + 15) & !15;
    let put_str = |statics: &mut Vec<(u32, Vec<u8>)>, off: &mut u32, s: &str| {
        let addr = *off;
        let mut b = Vec::with_capacity(16 + s.len());
        b.extend(TAG_STR.to_le_bytes());
        b.extend((addr + 16).to_le_bytes());
        b.extend((s.len() as u32).to_le_bytes());
        b.extend((s.len() as u32).to_le_bytes());
        b.extend(s.as_bytes());
        while b.len() % 8 != 0 {
            b.push(0);
        }
        *off += b.len() as u32;
        statics.push((addr, b));
        addr
    };
    let true_obj = put_str(&mut statics, &mut off, "true");
    let false_obj = put_str(&mut statics, &mut off, "false");
    let obj_obj = put_str(&mut statics, &mut off, "<obj>");
    let null_obj = put_str(&mut statics, &mut off, "null");
    let str_ids: Vec<u32> = str_objs.keys().copied().collect();
    for id in str_ids {
        let addr = put_str(&mut statics, &mut off, strs.get(StrId::from(id)));
        *str_objs.get_mut(&id).unwrap() = addr;
    }
    // Call marshals args as i64 words into a static scratch region
    let max_call_args = ir
        .bodies
        .iter()
        .flat_map(|(_, body)| {
            body.blocks.iter().flat_map(|(_, block)| {
                block.stream.iter().filter_map(|&iid| match &body.instructions[iid] {
                    Inst::Call { args, .. } => Some(args.len()),
                    _ => None,
                })
            })
        })
        .max()
        .unwrap_or(0)
        .max(8) as u32;
    let call_scratch = off;
    off += (max_call_args * 8 + 15) & !15;
    let i64_scratch = off;
    off += 32;
    // coverage region anchored at __sp + 1MB — `wgame` derives it as
    // `stack_base + (1<<20)`; statics+scratch must fit inside the hole
    if off > sp_init + STACK_CAP {
        return Err("static data overflows the 1MB stack hole".into());
    }
    let cov0 = sp_init + STACK_CAP;
    // op bitmap first (resume parity: bitmap lives at cov region start),
    // then the point map, decision cells, and the cmp operand stack
    let ptmap_base = (cov0 + cov_total + 7) & !7;
    let decv_base = (ptmap_base + cov_points + 7) & !7;
    let opstk_base = decv_base + cov_decs * DCELL;
    let sink_base = (opstk_base + OPSTK_N * 16 + 7) & !7;
    let heap_base = sink_base + if sink.is_empty() { 0 } else { SINK_CAP };
    let ctx = Statics {
        str_objs,
        true_obj,
        false_obj,
        obj_obj,
        null_obj,
        call_scratch,
        i64_scratch,
    };

    // native call sites across all sig'd bodies — the trial-emit map
    let mut trial_natives: HashMap<(u32, Vec<K>, Option<K>), u32> = HashMap::new();
    for b in 0..nbodies {
        let Some(ana) = &anas[b] else { continue };
        for (_, block) in bodies[b].blocks.iter() {
            for &iid in &block.stream {
                if let Inst::CallNative { id, .. } = &bodies[b].instructions[iid] {
                    if sink.contains_key(&(id.index() as u32)) {
                        continue; // sink natives record to memory, not imports
                    }
                    trial_natives
                        .entry(native_key(bodies[b], ana, iid))
                        .or_insert(0);
                }
            }
        }
    }
    let dummy_func_map: HashMap<usize, u32> = (0..nbodies).map(|b| (b, b as u32)).collect();

    // settle the emitted set: a body survives iff its callees are emitted, its
    // scopes nest, and a trial emission succeeds
    let mut emitted: Vec<bool> = sigs.iter().map(|s| s.is_some()).collect();
    let mut reasons: Vec<String> = (0..nbodies).map(|_| String::new()).collect();
    // trial emission validates, never runs — helper/tramp indices may be 0
    let trial_helpers: HashMap<&'static str, u32> =
        HELPER_NAMES.iter().map(|&n| (n, 0)).collect();
    let trial_tramp: HashMap<usize, u32> = (0..nbodies).map(|b| (b, 0)).collect();
    let trial_cov = (!sink.is_empty()).then(|| CovCtx {
        ptmap_base,
        decv_base,
        opstk_base,
        sink_base,
        g_covp: 0,
        g_osp: 0,
        h: [0; 8],
    });
    loop {
        let mut changed = false;
        for b in 0..nbodies {
            if !emitted[b] {
                continue;
            }
            let ana = anas[b].as_ref().unwrap();
            let mut why = String::new();
            if let Some(c) = ana.callee.values().find(|c| !emitted[**c]) {
                why = format!("calls skipped body {c}");
            } else if let Err(e) = try_ir_body(
                b,
                bodies[b],
                ana,
                &sigs,
                &dummy_func_map,
                &trial_natives,
                &ProbeGlobals {
                    cov_base: u32::MAX,
                    ..Default::default()
                },
                &entry_locals,
                16, // trial: region base value doesn't affect validation
                &ctx,
                &trial_helpers,
                &trial_tramp,
                0,
                &emitted,
                sink,
                math,
                trial_cov.as_ref(),
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
        for (_, block) in bodies[b].blocks.iter() {
            for &iid in &block.stream {
                if let Inst::CallNative { id, .. } = &bodies[b].instructions[iid] {
                    if sink.contains_key(&(id.index() as u32)) {
                        continue; // records go to the sink region, not imports
                    }
                    let key = native_key(bodies[b], ana, iid);
                    if !natives.contains_key(&key) {
                        let fi = native_list.len() as u32;
                        natives.insert(key.clone(), fi);
                        native_list.push(key);
                    }
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
    let nemitted = func_map.len() as u32;

    // internal helpers ride after the emitted bodies; the cov helpers follow
    // them (only when the sink is live), then trampolines
    let ncov = if sink.is_empty() {
        0
    } else {
        COV_HELPER_NAMES.len()
    } as u32;
    let mut helpers: HashMap<&'static str, u32> = HELPER_NAMES
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, nimports + nemitted + i as u32))
        .collect();
    for (i, &n) in COV_HELPER_NAMES.iter().enumerate().take(ncov as usize) {
        helpers.insert(n, nimports + nemitted + HELPER_NAMES.len() as u32 + i as u32);
    }
    let helper_base = nimports + nemitted;
    let tramp_base = helper_base + HELPER_NAMES.len() as u32 + ncov;

    // bodies a dynamic call may reach: every emitted body that a RefBody or
    // MakeClosure materializes as a word gets a call_indirect trampoline.
    // `has_dyn_call` means some body emits call_indirect — the funcref
    // table must then exist even when no trampoline targets do (a callee
    // word can arrive from a param, a native, or another module).
    let mut tramp_targets: Vec<usize> = Vec::new();
    let mut has_dyn_call = false;
    for b in 0..nbodies {
        if !emitted[b] {
            continue;
        }
        for (_, block) in bodies[b].blocks.iter() {
            for &iid in &block.stream {
                match &bodies[b].instructions[iid] {
                    Inst::RefBody(t) | Inst::MakeClosure { body: t, .. } => {
                        let t = t.index();
                        if emitted[t] && !tramp_targets.contains(&t) {
                            tramp_targets.push(t);
                        }
                    }
                    Inst::Call { .. } => has_dyn_call = true,
                    _ => {}
                }
            }
        }
    }
    let needs_table = has_dyn_call || !tramp_targets.is_empty();
    let tramp: HashMap<usize, u32> = tramp_targets
        .iter()
        .enumerate()
        .map(|(i, &b)| (b, i as u32))
        .collect();

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
    let mut cov_next = cov0; // op bitmap at the coverage region start

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
                K::Word => 'w',
            })
            .collect::<String>();
        // the ret char is part of the host ABI — wgame's dispatcher
        // parses `n{id}_{params}_{ret}` to pick the result channel
        let ret_desc = match ret {
            Some(K::Int) => 'i',
            Some(K::Float) => 'f',
            Some(K::Bool) => 'b',
            Some(K::Word) => 'w',
            None => 'v',
        };
        imports.import(
            "env",
            &format!("n{id}_{sig_desc}_{ret_desc}"),
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
    let fuel_g = opts
        .fuel
        .then(|| import_global(&mut imports, "__fuel", ValType::I64));
    let pause_g = opts
        .pause
        .then(|| import_global(&mut imports, "__pause", ValType::I32));
    // __hp is the first defined global — the module declares it below,
    // followed by the coverage-sink contract globals wgame reads
    let hp_g = gnext;
    let g_status = gnext + 1;
    let g_sp = gnext + 2;
    let g_covp = gnext + 3;
    let g_covbuf = gnext + 4;
    let g_ptmap = gnext + 5;
    let g_osp = gnext + 6;
    let g_dcov = gnext + 7;
    let covctx = (!sink.is_empty()).then(|| CovCtx {
        ptmap_base,
        decv_base,
        opstk_base,
        sink_base,
        g_covp,
        g_osp,
        h: [
            helpers[H_COV_LEAF],
            helpers[H_COV_BEGIN],
            helpers[H_COV_COND],
            helpers[H_COV_CMP],
            helpers[H_COV_DEC],
            helpers[H_COV_GAP],
            helpers[H_COV_NEAR],
            helpers[H_COV_BIT],
        ],
    });
    let mut globs = ProbeGlobals {
        fuel_g,
        pause_g,
        cov_base: u32::MAX,
    };
    let nimport_entries = imports.len();

    // uniform trampoline signature for call_indirect — (i32,i32,i32)->i64
    // (env obj, args scratch ptr, nargs)
    let tramp_ty = {
        let key = (vec![K::Bool, K::Bool, K::Bool], Some(K::Int));
        let ntypes = types.len();
        let ty = *type_ids.entry(key).or_insert(ntypes);
        if ty == ntypes {
            types
                .ty()
                .function([ValType::I32, ValType::I32, ValType::I32], [ValType::I64]);
        }
        ty
    };

    for b in 0..nbodies {
        let Some(sig) = &sigs[b] else { continue };
        let ana = anas[b].as_ref().unwrap();
        let body = bodies[b];
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
            cov_next += ninsts[b] as u32;
        }
        let (f, sm) = try_ir_body(
            b,
            body,
            ana,
            &sigs,
            &func_map,
            &natives,
            &globs,
            &entry_locals,
            entry_base,
            &ctx,
            &helpers,
            &tramp,
            tramp_ty,
            &emitted,
            sink,
            math,
            covctx.as_ref(),
        )
        .map_err(|e| format!("body {b} passed trial but failed emit: {e}"))?;
        let sm_loc: Vec<(u32, u32)> = sm
            .iter()
            .map(|&(off, i)| {
                let loc = body.locs[InstId::from(i)];
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

    // ---------- appended runtime: helpers, then call_indirect trampolines ----------
    let mut htype_ids: HashMap<(Vec<ValType>, Vec<ValType>), u32> = HashMap::new();
    let hty = |types: &mut TypeSection,
                   htype_ids: &mut HashMap<(Vec<ValType>, Vec<ValType>), u32>,
                   p: &[ValType],
                   r: &[ValType]| {
        let key = (p.to_vec(), r.to_vec());
        let ntypes = types.len();
        let ty = *htype_ids.entry(key).or_insert(ntypes);
        if ty == ntypes {
            types.ty().function(p.iter().copied(), r.iter().copied());
        }
        ty
    };
    for h in emit_helpers(&ctx, &helpers, hp_g, covctx.as_ref()) {
        let ty = hty(&mut types, &mut htype_ids, h.params, h.rets);
        funcs.function(ty);
        code.function(&h.f);
        names.append(helpers[h.name], h.name);
    }
    for &b in &tramp_targets {
        let body = bodies[b];
        let sig = sigs[b].as_ref().unwrap();
        let f = emit_trampoline(body, sig, func_map[&b]);
        funcs.function(tramp_ty);
        code.function(&f);
        names.append(tramp_base + tramp[&b], &format!("__tr_b{b}"));
    }

    let mut module = Module::new();
    module.section(&types);
    if nimport_entries > 0 {
        module.section(&imports);
    }
    module.section(&funcs);
    // function table for call_indirect — whenever a body can emit the
    // instruction (dynamic call) or a body can be a dynamic callee
    if needs_table {
        let mut tables = TableSection::new();
        tables.table(TableType {
            element_type: RefType::FUNCREF,
            table64: false,
            minimum: tramp_targets.len() as u64,
            maximum: None,
            shared: false,
        });
        module.section(&tables);
    }
    // linear memory always exists — sized to cover the heap base
    let pages = (heap_base as u64 + 65535) / 65536;
    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: pages.max(1),
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    module.section(&mems);
    // defined globals — index order must match the g_* declarations above
    let mut globals = GlobalSection::new();
    let mut gdef = |mutable: bool, init: i32| {
        globals.global(
            GlobalType {
                val_type: ValType::I32,
                mutable,
                shared: false,
            },
            &ConstExpr::i32_const(init),
        );
    };
    gdef(true, heap_base as i32); // __hp
    gdef(true, 0); // __status — always 0: the structured lane never suspends
    gdef(true, sp_init as i32); // __sp — constant; [0,__sp) is a snapshot range
    gdef(true, 0); // __covp — sink record cursor
    gdef(false, sink_base as i32); // __covbuf
    gdef(false, ptmap_base as i32); // __ptmap
    gdef(true, 0); // __osp — cmp operand-stack depth (internal)
    gdef(false, decv_base as i32); // __dcov
    module.section(&globals);
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("__status", ExportKind::Global, g_status);
    exports.export("__sp", ExportKind::Global, g_sp);
    exports.export("__hp", ExportKind::Global, hp_g);
    exports.export("__covp", ExportKind::Global, g_covp);
    exports.export("__covbuf", ExportKind::Global, g_covbuf);
    exports.export("__ptmap", ExportKind::Global, g_ptmap);
    exports.export("__dcov", ExportKind::Global, g_dcov);
    module.section(&exports);
    // trampoline table indices → func indices, matching the tag field
    // closure objects store at +4
    if !tramp_targets.is_empty() {
        let mut elems = ElementSection::new();
        let func_idxs: Vec<u32> = tramp_targets
            .iter()
            .map(|&b| tramp_base + tramp[&b])
            .collect();
        elems.active(
            None,
            &ConstExpr::i32_const(0),
            Elements::Functions(func_idxs.into()),
        );
        module.section(&elems);
    }
    module.section(&code);
    // static str objects (tag + header + bytes) at their laid-out addrs
    if !statics.is_empty() {
        let mut data = DataSection::new();
        for (addr, bytes) in &statics {
            data.active(
                0,
                &ConstExpr::i32_const(*addr as i32),
                bytes.iter().copied(),
            );
        }
        module.section(&data);
    }
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
fn try_ir_body(
    b: usize,
    body: &IrBody,
    ana: &AnaI,
    sigs: &[Option<Sig>],
    func_map: &HashMap<usize, u32>,
    natives: &HashMap<(u32, Vec<K>, Option<K>), u32>,
    globs: &ProbeGlobals,
    entry_locals: &HashSet<u32>,
    entry_base: u32,
    ctx: &Statics,
    helpers: &HashMap<&'static str, u32>,
    tramp: &HashMap<usize, u32>,
    tramp_ty: u32,
    emitted: &[bool],
    sink: &crate::CovSink,
    math: &crate::MathNatives,
    cov: Option<&CovCtx>,
) -> Result<(Function, Vec<(u32, u32)>), Bail> {
    if sigs[b].is_none() {
        bail!("no sig for body {b}");
    }
    let (order, pos, rt) = layout(body);
    let sc = scopes_ir(body, &order, &pos, &rt)?;
    let copies = phi_copies(body)?;

    // wasm locals: captures then params occupy the leading slots (the
    // trampoline fills captures from the env word), then classed locals,
    // then classed insts, then the i64 scratch
    let mut llocal: HashMap<u32, u32> = HashMap::new();
    for (i, p) in body.captures.iter().enumerate() {
        llocal.insert(p.index() as u32, i as u32);
    }
    for (i, p) in body.params.iter().enumerate() {
        llocal.insert(p.index() as u32, (body.captures.len() + i) as u32);
    }
    let mut groups: Vec<(u32, ValType)> = Vec::new();
    let mut next = (body.captures.len() + body.params.len()) as u32;
    fn alloc(groups: &mut Vec<(u32, ValType)>, next: &mut u32, k: K) -> u32 {
        let ix = *next;
        *next += 1;
        if let Some(g) = groups.last_mut()
            && g.1 == k.val_type()
        {
            g.0 += 1;
            return ix;
        }
        groups.push((1, k.val_type()));
        ix
    }
    // every demanded local gets a slot — classed ones store their class,
    // the rest store raw words
    let mut locals_sorted: Vec<u32> = ana.lused.iter().copied().collect();
    locals_sorted.sort();
    for l in locals_sorted {
        if llocal.contains_key(&l) {
            continue;
        }
        let k = ana.lclass.get(&l).copied().unwrap_or(K::Word);
        let ix = alloc(&mut groups, &mut next, k);
        llocal.insert(l, ix);
    }
    let mut ilocal: HashMap<u32, u32> = HashMap::new();
    let mut insts_sorted: Vec<u32> = ana
        .class
        .keys()
        .chain(ana.wused.iter())
        .copied()
        .collect();
    insts_sorted.sort();
    insts_sorted.dedup();
    for ix in insts_sorted {
        let iid = InstId::from(ix);
        // GetLocal aliases its local's slot; consts inline at use sites
        if matches!(
            body.instructions[iid],
            Inst::GetLocal(_) | Inst::Constant(_)
        ) {
            continue;
        }
        let k = ana.class.get(&ix).copied().unwrap_or(K::Word);
        let l = alloc(&mut groups, &mut next, k);
        ilocal.insert(ix, l);
    }
    // scratch: one i64 (tmp) then four i32s (sz/hp/tb/tc)
    let tmp = alloc(&mut groups, &mut next, K::Int);
    let sz = alloc(&mut groups, &mut next, K::Bool);
    let hp = alloc(&mut groups, &mut next, K::Bool);
    let tb = alloc(&mut groups, &mut next, K::Bool);
    let tc = alloc(&mut groups, &mut next, K::Bool);

    let mut em = Emi {
        body,
        class: &ana.class,
        lclass: &ana.lclass,
        ilocal: &ilocal,
        wused: &ana.wused,
        llocal: &llocal,
        sigs,
        func_map,
        callee: &ana.callee,
        natives,
        ret_k: ana.ret,
        tmp,
        sz,
        hp,
        tb,
        tc,
        is_entry: b == 0,
        entry_locals,
        entry_base,
        ctx,
        helpers,
        tramp,
        tramp_ty,
        emitted,
        rt: &rt,
        copies: &copies,
        stack: Vec::new(),
        f: Function::new(groups),
        code_off: 0,
        srcmap: Vec::new(),
        fuel_g: globs.fuel_g,
        pause_g: globs.pause_g,
        cov_base: globs.cov_base,
        sink,
        math,
        cov,
    };
    em.emit_body(&order, &pos, sc)?;
    Ok((em.f, em.srcmap))
}

// ---------- internal helpers: hand-coded runtime functions ----------

/// A hand-coded runtime function appended after the emitted bodies.
pub(crate) struct HelperFn {
    pub name: &'static str,
    pub params: &'static [ValType],
    pub rets: &'static [ValType],
    pub f: Function,
}

/// Emit every helper in `HELPER_NAMES`. They implement the tagged-heap
/// runtime — str concat/compare/index, int/float ascii, dict hashing and
/// probing — inside the module so the lane needs no host runtime services.
pub(crate) fn emit_helpers(
    ctx: &Statics,
    helpers: &HashMap<&'static str, u32>,
    hp_g: u32,
    cov: Option<&CovCtx>,
) -> Vec<HelperFn> {
    use Instruction as I;
    use ValType as V;
    let mut out: Vec<HelperFn> = Vec::new();
    let i64_scratch = ctx.i64_scratch as i32;
    let obj_obj = ctx.obj_obj as i32;
    let null_obj = ctx.null_obj as i32;
    macro_rules! push {
        ($name:expr, $params:expr, $rets:expr, $f:expr) => {{
            let mut f = $f;
            f.instruction(&I::End);
            out.push(HelperFn {
                name: $name,
                params: $params,
                rets: $rets,
                f,
            });
        }};
    }

    // ---- __alloc(sz i32) -> i32: bump __hp, grow memory on demand ----
    {
        let mut f = Function::new(vec![]);
        for i in [
            // sz = (sz + 7) & ~7 — keep word alignment for hosts
            I::LocalGet(0),
            I::I32Const(7),
            I::I32Add,
            I::I32Const(-8),
            I::I32And,
            I::LocalSet(0),
            // if __hp + sz > memory.size << 16 → grow
            I::GlobalGet(hp_g),
            I::LocalGet(0),
            I::I32Add,
            I::MemorySize(0),
            I::I32Const(16),
            I::I32Shl,
            I::I32GtU,
            I::If(BlockType::Empty),
            I::GlobalGet(hp_g),
            I::LocalGet(0),
            I::I32Add,
            I::MemorySize(0),
            I::I32Const(16),
            I::I32Shl,
            I::I32Sub,
            I::I32Const(65535),
            I::I32Add,
            I::I32Const(16),
            I::I32ShrU,
            I::MemoryGrow(0),
            I::I32Const(-1),
            I::I32Eq,
            I::If(BlockType::Empty),
            I::Unreachable,
            I::End,
            I::End,
            // ret = __hp; __hp += sz
            I::GlobalGet(hp_g),
            I::GlobalGet(hp_g),
            I::LocalGet(0),
            I::I32Add,
            I::GlobalSet(hp_g),
        ] {
            f.instruction(&i);
        }
        push!(H_ALLOC, &[V::I32], &[V::I32], f);
    }

    // ---- __is_str(w i64) -> i32: plausible tagged str object? ----
    // aligned && >= 16 && < __hp && tag == TAG_STR — statics pass (they
    // live above 16 and below __hp); raw ints fail somewhere cheap.
    {
        let mut f = Function::new(vec![]);
        for i in [
            I::LocalGet(0),
            I::I64Const(7),
            I::I64And,
            I::I64Eqz,
            I::If(BlockType::Result(V::I32)),
            I::LocalGet(0),
            I::I64Const(16),
            I::I64GeU,
            I::If(BlockType::Result(V::I32)),
            I::LocalGet(0),
            I::GlobalGet(hp_g),
            I::I64ExtendI32U,
            I::I64LtU,
            I::If(BlockType::Result(V::I32)),
            I::LocalGet(0),
            I::I32WrapI64,
            I::I32Load(mem_arg(0, 2)),
            I::I32Const(TAG_STR as i32),
            I::I32Eq,
            I::Else,
            I::I32Const(0),
            I::End,
            I::Else,
            I::I32Const(0),
            I::End,
            I::Else,
            I::I32Const(0),
            I::End,
        ] {
            f.instruction(&i);
        }
        push!(H_IS_STR, &[V::I64], &[V::I32], f);
    }

    // ---- __str_eq(a, b) -> i32: byte equality on tagged strs ----
    // locals 2=da,3=db,4=len,5=i,6=ret
    {
        let mut f = Function::new(vec![(5, V::I32)]);
        for i in [
            I::LocalGet(0),
            I::LocalGet(1),
            I::I32Eq,
            I::If(BlockType::Result(V::I32)),
            I::I32Const(1),
            I::Else,
            I::LocalGet(0),
            I::I32Load(mem_arg(8, 2)),
            I::LocalGet(1),
            I::I32Load(mem_arg(8, 2)),
            I::I32Ne,
            I::If(BlockType::Result(V::I32)),
            I::I32Const(0),
            I::Else,
            I::LocalGet(0),
            I::I32Load(mem_arg(4, 2)),
            I::LocalSet(2),
            I::LocalGet(1),
            I::I32Load(mem_arg(4, 2)),
            I::LocalSet(3),
            I::LocalGet(0),
            I::I32Load(mem_arg(8, 2)),
            I::LocalSet(4),
            I::I32Const(1),
            I::LocalSet(6),
            I::Block(BlockType::Empty),
            I::Loop(BlockType::Empty),
            I::LocalGet(5),
            I::LocalGet(4),
            I::I32GeU,
            I::BrIf(1),
            I::LocalGet(2),
            I::LocalGet(5),
            I::I32Add,
            I::I32Load8U(mem_arg(0, 0)),
            I::LocalGet(3),
            I::LocalGet(5),
            I::I32Add,
            I::I32Load8U(mem_arg(0, 0)),
            I::I32Ne,
            I::If(BlockType::Empty),
            I::I32Const(0),
            I::LocalSet(6),
            I::Br(2),
            I::End,
            I::LocalGet(5),
            I::I32Const(1),
            I::I32Add,
            I::LocalSet(5),
            I::Br(0),
            I::End,
            I::End,
            I::LocalGet(6),
            I::End,
            I::End,
        ] {
            f.instruction(&i);
        }
        push!(H_STR_EQ, &[V::I32, V::I32], &[V::I32], f);
    }

    // ---- __str_cat(a, b) -> i32: fresh concat object ----
    // locals 2=la,3=lb,4=out
    {
        let mut f = Function::new(vec![(3, V::I32)]);
        for i in [
            I::LocalGet(0),
            I::I32Load(mem_arg(8, 2)),
            I::LocalSet(2),
            I::LocalGet(1),
            I::I32Load(mem_arg(8, 2)),
            I::LocalSet(3),
            I::I32Const(16),
            I::LocalGet(2),
            I::I32Add,
            I::LocalGet(3),
            I::I32Add,
            I::Call(helpers[H_ALLOC]),
            I::LocalSet(4),
            I::LocalGet(4),
            I::I32Const(TAG_STR as i32),
            I::I32Store(mem_arg(0, 2)),
            I::LocalGet(4),
            I::LocalGet(4),
            I::I32Const(16),
            I::I32Add,
            I::I32Store(mem_arg(4, 2)),
            I::LocalGet(4),
            I::LocalGet(2),
            I::LocalGet(3),
            I::I32Add,
            I::I32Store(mem_arg(8, 2)),
            I::LocalGet(4),
            I::LocalGet(2),
            I::LocalGet(3),
            I::I32Add,
            I::I32Store(mem_arg(12, 2)),
            I::LocalGet(4),
            I::I32Const(16),
            I::I32Add,
            I::LocalGet(0),
            I::I32Load(mem_arg(4, 2)),
            I::LocalGet(2),
            I::MemoryCopy {
                src_mem: 0,
                dst_mem: 0,
            },
            I::LocalGet(4),
            I::I32Const(16),
            I::I32Add,
            I::LocalGet(2),
            I::I32Add,
            I::LocalGet(1),
            I::I32Load(mem_arg(4, 2)),
            I::LocalGet(3),
            I::MemoryCopy {
                src_mem: 0,
                dst_mem: 0,
            },
            I::LocalGet(4),
        ] {
            f.instruction(&i);
        }
        push!(H_STR_CAT, &[V::I32, V::I32], &[V::I32], f);
    }

    // ---- __str_cmp(a, b) -> i32: byte-lex compare → -1/0/1 ----
    // locals 2=da,3=db,4=la,5=lb,6=n,7=i
    {
        let mut f = Function::new(vec![(6, V::I32)]);
        for i in [
            I::LocalGet(0),
            I::I32Load(mem_arg(4, 2)),
            I::LocalSet(2),
            I::LocalGet(1),
            I::I32Load(mem_arg(4, 2)),
            I::LocalSet(3),
            I::LocalGet(0),
            I::I32Load(mem_arg(8, 2)),
            I::LocalSet(4),
            I::LocalGet(1),
            I::I32Load(mem_arg(8, 2)),
            I::LocalSet(5),
            I::LocalGet(4),
            I::LocalGet(5),
            I::I32LtU,
            I::If(BlockType::Result(V::I32)),
            I::LocalGet(4),
            I::Else,
            I::LocalGet(5),
            I::End,
            I::LocalSet(6),
            I::Block(BlockType::Empty),
            I::Loop(BlockType::Empty),
            I::LocalGet(7),
            I::LocalGet(6),
            I::I32GeU,
            I::BrIf(1),
            I::LocalGet(2),
            I::LocalGet(7),
            I::I32Add,
            I::I32Load8U(mem_arg(0, 0)),
            I::LocalGet(3),
            I::LocalGet(7),
            I::I32Add,
            I::I32Load8U(mem_arg(0, 0)),
            I::I32LtU,
            I::If(BlockType::Empty),
            I::I32Const(-1),
            I::Return,
            I::End,
            I::LocalGet(2),
            I::LocalGet(7),
            I::I32Add,
            I::I32Load8U(mem_arg(0, 0)),
            I::LocalGet(3),
            I::LocalGet(7),
            I::I32Add,
            I::I32Load8U(mem_arg(0, 0)),
            I::I32GtU,
            I::If(BlockType::Empty),
            I::I32Const(1),
            I::Return,
            I::End,
            I::LocalGet(7),
            I::I32Const(1),
            I::I32Add,
            I::LocalSet(7),
            I::Br(0),
            I::End,
            I::End,
            I::LocalGet(4),
            I::LocalGet(5),
            I::I32LtU,
            I::If(BlockType::Empty),
            I::I32Const(-1),
            I::Return,
            I::End,
            I::LocalGet(4),
            I::LocalGet(5),
            I::I32GtU,
            I::If(BlockType::Empty),
            I::I32Const(1),
            I::Return,
            I::End,
            I::I32Const(0),
        ] {
            f.instruction(&i);
        }
        push!(H_STR_CMP, &[V::I32, V::I32], &[V::I32], f);
    }

    // ---- __i64_str(v i64) -> i32: decimal ascii object ----
    // locals 1=neg,2=pos,3=out (i32); 4=n (i64)
    {
        let mut f = Function::new(vec![(3, V::I32), (1, V::I64)]);
        for i in [
            I::LocalGet(0),
            I::I64Const(0),
            I::I64LtS,
            I::If(BlockType::Empty),
            I::I32Const(1),
            I::LocalSet(1),
            I::I64Const(0),
            I::LocalGet(0),
            I::I64Sub,
            I::LocalSet(4),
            I::Else,
            I::LocalGet(0),
            I::LocalSet(4),
            I::End,
            I::I32Const(32),
            I::LocalSet(2),
            I::Loop(BlockType::Empty),
            I::LocalGet(2),
            I::I32Const(1),
            I::I32Sub,
            I::LocalSet(2),
            I::I32Const(i64_scratch),
            I::LocalGet(2),
            I::I32Add,
            I::LocalGet(4),
            I::I64Const(10),
            I::I64RemU,
            I::I64Const(48),
            I::I64Add,
            I::I32WrapI64,
            I::I32Store8(mem_arg(0, 0)),
            I::LocalGet(4),
            I::I64Const(10),
            I::I64DivU,
            I::LocalSet(4),
            I::LocalGet(4),
            I::I64Const(0),
            I::I64Ne,
            I::BrIf(0),
            I::End,
            I::LocalGet(1),
            I::If(BlockType::Empty),
            I::LocalGet(2),
            I::I32Const(1),
            I::I32Sub,
            I::LocalSet(2),
            I::I32Const(i64_scratch),
            I::LocalGet(2),
            I::I32Add,
            I::I32Const(45),
            I::I32Store8(mem_arg(0, 0)),
            I::End,
            // out = __alloc(16 + (32 - pos))
            I::I32Const(16),
            I::I32Const(32),
            I::LocalGet(2),
            I::I32Sub,
            I::I32Add,
            I::Call(helpers[H_ALLOC]),
            I::LocalSet(3),
            I::LocalGet(3),
            I::I32Const(TAG_STR as i32),
            I::I32Store(mem_arg(0, 2)),
            I::LocalGet(3),
            I::LocalGet(3),
            I::I32Const(16),
            I::I32Add,
            I::I32Store(mem_arg(4, 2)),
            I::LocalGet(3),
            I::I32Const(32),
            I::LocalGet(2),
            I::I32Sub,
            I::I32Store(mem_arg(8, 2)),
            I::LocalGet(3),
            I::I32Const(32),
            I::LocalGet(2),
            I::I32Sub,
            I::I32Store(mem_arg(12, 2)),
            I::LocalGet(3),
            I::I32Const(16),
            I::I32Add,
            I::I32Const(i64_scratch),
            I::LocalGet(2),
            I::I32Add,
            I::I32Const(32),
            I::LocalGet(2),
            I::I32Sub,
            I::MemoryCopy {
                src_mem: 0,
                dst_mem: 0,
            },
            I::LocalGet(3),
        ] {
            f.instruction(&i);
        }
        push!(H_I64_STR, &[V::I64], &[V::I32], f);
    }

    // ---- __f64_str(v f64) -> i32: integral or %.6f ascii ----
    // locals 1=pos,2=out,3=cnt,4=i (i32); 5=v2,6=frac (f64); 7=ip,8=t (i64)
    {
        let mut f = Function::new(vec![(4, V::I32), (2, V::F64), (2, V::I64)]);
        for i in [
            // nan or |v| >= 2^62 → "<obj>"
            I::LocalGet(0),
            I::LocalGet(0),
            I::F64Ne,
            I::If(BlockType::Empty),
            I::I32Const(obj_obj),
            I::Return,
            I::End,
            I::LocalGet(0),
            I::F64Abs,
            I::F64Const(4611686018427387904.0f64.into()),
            I::F64Ge,
            I::If(BlockType::Empty),
            I::I32Const(obj_obj),
            I::Return,
            I::End,
            // integral → decimal via __i64_str
            I::LocalGet(0),
            I::F64Trunc,
            I::LocalGet(0),
            I::F64Eq,
            I::If(BlockType::Result(V::I32)),
            I::LocalGet(0),
            I::I64TruncF64S,
            I::Call(helpers[H_I64_STR]),
            I::Else,
            // %.6f into the scratch buffer, forward
            I::I32Const(0),
            I::LocalSet(1),
            I::LocalGet(0),
            I::LocalSet(5),
            I::LocalGet(5),
            I::F64Const(0.0f64.into()),
            I::F64Lt,
            I::If(BlockType::Empty),
            I::I32Const(i64_scratch),
            I::LocalGet(1),
            I::I32Add,
            I::I32Const(45),
            I::I32Store8(mem_arg(0, 0)),
            I::LocalGet(1),
            I::I32Const(1),
            I::I32Add,
            I::LocalSet(1),
            I::LocalGet(5),
            I::F64Neg,
            I::LocalSet(5),
            I::End,
            I::LocalGet(5),
            I::I64TruncF64S,
            I::LocalSet(7),
            I::LocalGet(5),
            I::LocalGet(7),
            I::F64ConvertI64S,
            I::F64Sub,
            I::LocalSet(6),
            // count digits of ip
            I::LocalGet(7),
            I::LocalSet(8),
            I::Loop(BlockType::Empty),
            I::LocalGet(3),
            I::I32Const(1),
            I::I32Add,
            I::LocalSet(3),
            I::LocalGet(8),
            I::I64Const(10),
            I::I64DivU,
            I::LocalSet(8),
            I::LocalGet(8),
            I::I64Const(0),
            I::I64Ne,
            I::BrIf(0),
            I::End,
            // write digits backward from pos+cnt
            I::LocalGet(1),
            I::LocalGet(3),
            I::I32Add,
            I::LocalSet(4),
            I::Loop(BlockType::Empty),
            I::LocalGet(4),
            I::I32Const(1),
            I::I32Sub,
            I::LocalSet(4),
            I::I32Const(i64_scratch),
            I::LocalGet(4),
            I::I32Add,
            I::LocalGet(7),
            I::I64Const(10),
            I::I64RemU,
            I::I64Const(48),
            I::I64Add,
            I::I32WrapI64,
            I::I32Store8(mem_arg(0, 0)),
            I::LocalGet(7),
            I::I64Const(10),
            I::I64DivU,
            I::LocalSet(7),
            I::LocalGet(4),
            I::LocalGet(1),
            I::I32GtU,
            I::BrIf(0),
            I::End,
            I::LocalGet(1),
            I::LocalGet(3),
            I::I32Add,
            I::LocalSet(1),
            I::I32Const(i64_scratch),
            I::LocalGet(1),
            I::I32Add,
            I::I32Const(46),
            I::I32Store8(mem_arg(0, 0)),
            I::LocalGet(1),
            I::I32Const(1),
            I::I32Add,
            I::LocalSet(1),
            // 6 fractional digits
            I::I32Const(6),
            I::LocalSet(3),
            I::Loop(BlockType::Empty),
            I::LocalGet(6),
            I::F64Const(10.0f64.into()),
            I::F64Mul,
            I::LocalSet(6),
            I::LocalGet(6),
            I::I64TruncF64S,
            I::LocalSet(7),
            I::I32Const(i64_scratch),
            I::LocalGet(1),
            I::I32Add,
            I::LocalGet(7),
            I::I64Const(48),
            I::I64Add,
            I::I32WrapI64,
            I::I32Store8(mem_arg(0, 0)),
            I::LocalGet(6),
            I::LocalGet(7),
            I::F64ConvertI64S,
            I::F64Sub,
            I::LocalSet(6),
            I::LocalGet(1),
            I::I32Const(1),
            I::I32Add,
            I::LocalSet(1),
            I::LocalGet(3),
            I::I32Const(1),
            I::I32Sub,
            I::LocalSet(3),
            I::LocalGet(3),
            I::I32Const(0),
            I::I32GtS,
            I::BrIf(0),
            I::End,
            // out = __alloc(16 + pos); header; copy
            I::I32Const(16),
            I::LocalGet(1),
            I::I32Add,
            I::Call(helpers[H_ALLOC]),
            I::LocalSet(2),
            I::LocalGet(2),
            I::I32Const(TAG_STR as i32),
            I::I32Store(mem_arg(0, 2)),
            I::LocalGet(2),
            I::LocalGet(2),
            I::I32Const(16),
            I::I32Add,
            I::I32Store(mem_arg(4, 2)),
            I::LocalGet(2),
            I::LocalGet(1),
            I::I32Store(mem_arg(8, 2)),
            I::LocalGet(2),
            I::LocalGet(1),
            I::I32Store(mem_arg(12, 2)),
            I::LocalGet(2),
            I::I32Const(16),
            I::I32Add,
            I::I32Const(i64_scratch),
            I::LocalGet(1),
            I::MemoryCopy {
                src_mem: 0,
                dst_mem: 0,
            },
            I::LocalGet(2),
            I::End,
        ] {
            f.instruction(&i);
        }
        push!(H_F64_STR, &[V::F64], &[V::I32], f);
    }

    // ---- __str_charat(s i32, i i32) -> i32: i-th utf-8 char object ----
    // locals 2=data,3=len,4=pos,5=ci,6=w,7=out,8=b
    {
        let utf8_len: &[I] = &[
            I::LocalGet(8),
            I::I32Const(0x80),
            I::I32LtU,
            I::If(BlockType::Result(V::I32)),
            I::I32Const(1),
            I::Else,
            I::LocalGet(8),
            I::I32Const(0xF0),
            I::I32GeU,
            I::If(BlockType::Result(V::I32)),
            I::I32Const(4),
            I::Else,
            I::LocalGet(8),
            I::I32Const(0xE0),
            I::I32GeU,
            I::If(BlockType::Result(V::I32)),
            I::I32Const(3),
            I::Else,
            I::I32Const(2),
            I::End,
            I::End,
            I::End,
        ];
        let mut f = Function::new(vec![(7, V::I32)]);
        for i in [
            I::LocalGet(0),
            I::I32Load(mem_arg(4, 2)),
            I::LocalSet(2),
            I::LocalGet(0),
            I::I32Load(mem_arg(8, 2)),
            I::LocalSet(3),
            I::Loop(BlockType::Empty),
            // bounds: pos >= len → trap
            I::LocalGet(4),
            I::LocalGet(3),
            I::I32GeU,
            I::If(BlockType::Empty),
            I::Unreachable,
            I::End,
            // found?
            I::LocalGet(5),
            I::LocalGet(1),
            I::I32Eq,
            I::If(BlockType::Empty),
            I::LocalGet(2),
            I::LocalGet(4),
            I::I32Add,
            I::I32Load8U(mem_arg(0, 0)),
            I::LocalSet(8),
        ] {
            f.instruction(&i);
        }
        for i in utf8_len {
            f.instruction(i);
        }
        for i in [
            I::LocalSet(6),
            I::I32Const(16),
            I::LocalGet(6),
            I::I32Add,
            I::Call(helpers[H_ALLOC]),
            I::LocalSet(7),
            I::LocalGet(7),
            I::I32Const(TAG_STR as i32),
            I::I32Store(mem_arg(0, 2)),
            I::LocalGet(7),
            I::LocalGet(7),
            I::I32Const(16),
            I::I32Add,
            I::I32Store(mem_arg(4, 2)),
            I::LocalGet(7),
            I::LocalGet(6),
            I::I32Store(mem_arg(8, 2)),
            I::LocalGet(7),
            I::LocalGet(6),
            I::I32Store(mem_arg(12, 2)),
            I::LocalGet(7),
            I::I32Const(16),
            I::I32Add,
            I::LocalGet(2),
            I::LocalGet(4),
            I::I32Add,
            I::LocalGet(6),
            I::MemoryCopy {
                src_mem: 0,
                dst_mem: 0,
            },
            I::LocalGet(7),
            I::Return,
            I::End,
            // skip the char at pos
            I::LocalGet(2),
            I::LocalGet(4),
            I::I32Add,
            I::I32Load8U(mem_arg(0, 0)),
            I::LocalSet(8),
        ] {
            f.instruction(&i);
        }
        for i in utf8_len {
            f.instruction(i);
        }
        for i in [
            I::LocalSet(6),
            I::LocalGet(4),
            I::LocalGet(6),
            I::I32Add,
            I::LocalSet(4),
            I::LocalGet(5),
            I::I32Const(1),
            I::I32Add,
            I::LocalSet(5),
            I::Br(0),
            I::End,
            // the loop only exits via found-char return or bounds trap
            I::Unreachable,
        ] {
            f.instruction(&i);
        }
        push!(H_STR_CHAR, &[V::I32, V::I32], &[V::I32], f);
    }

    // ---- __str_in(needle i32, s i32) -> i32: substring scan ----
    // locals 2=nd,3=nl,4=sd,5=sl,6=i,7=j
    {
        let mut f = Function::new(vec![(6, V::I32)]);
        for i in [
            I::LocalGet(0),
            I::I32Load(mem_arg(4, 2)),
            I::LocalSet(2),
            I::LocalGet(0),
            I::I32Load(mem_arg(8, 2)),
            I::LocalSet(3),
            I::LocalGet(1),
            I::I32Load(mem_arg(4, 2)),
            I::LocalSet(4),
            I::LocalGet(1),
            I::I32Load(mem_arg(8, 2)),
            I::LocalSet(5),
            I::LocalGet(3),
            I::LocalGet(5),
            I::I32GtU,
            I::If(BlockType::Empty),
            I::I32Const(0),
            I::Return,
            I::End,
            I::Block(BlockType::Empty),
            I::Loop(BlockType::Empty),
            // i > sl - nl → miss
            I::LocalGet(6),
            I::LocalGet(5),
            I::LocalGet(3),
            I::I32Sub,
            I::I32GtU,
            I::BrIf(1),
            I::I32Const(0),
            I::LocalSet(7),
            I::Block(BlockType::Empty),
            I::Loop(BlockType::Empty),
            // j == nl → whole needle matched
            I::LocalGet(7),
            I::LocalGet(3),
            I::I32GeU,
            I::If(BlockType::Empty),
            I::I32Const(1),
            I::Return,
            I::End,
            I::LocalGet(4),
            I::LocalGet(6),
            I::I32Add,
            I::LocalGet(7),
            I::I32Add,
            I::I32Load8U(mem_arg(0, 0)),
            I::LocalGet(2),
            I::LocalGet(7),
            I::I32Add,
            I::I32Load8U(mem_arg(0, 0)),
            I::I32Ne,
            I::If(BlockType::Empty),
            // mismatch → exit the inner Block → advance the outer scan
            I::Br(2),
            I::End,
            I::LocalGet(7),
            I::I32Const(1),
            I::I32Add,
            I::LocalSet(7),
            I::Br(0),
            I::End,
            I::End,
            I::LocalGet(6),
            I::I32Const(1),
            I::I32Add,
            I::LocalSet(6),
            I::Br(0),
            I::End,
            I::End,
            I::I32Const(0),
        ] {
            f.instruction(&i);
        }
        push!(H_STR_IN, &[V::I32, V::I32], &[V::I32], f);
    }

    // ---- __str_or_obj(w i64) -> i32: display object for a word ----
    // str → content, tagged object → "<obj>", anything else → int digits
    // (the word repr can't prove int-ness, but a word that isn't a tagged
    // object is an int/bool — ints are the common Word-classed value)
    // local 1 = tag scratch
    {
        let mut f = Function::new(vec![(1, V::I32)]);
        for i in [
            I::LocalGet(0),
            I::I64Eqz,
            I::If(BlockType::Empty),
            I::I32Const(null_obj),
            I::Return,
            I::End,
            I::LocalGet(0),
            I::Call(helpers[H_IS_STR]),
            I::If(BlockType::Empty),
            I::LocalGet(0),
            I::I32WrapI64,
            I::Return,
            I::End,
            // plausible tagged object: aligned, in-heap, tag in {1,3,4,5}
            I::LocalGet(0),
            I::I64Const(7),
            I::I64And,
            I::I64Eqz,
            I::LocalGet(0),
            I::I64Const(16),
            I::I64GeU,
            I::I32And,
            I::LocalGet(0),
            I::GlobalGet(hp_g),
            I::I64ExtendI32U,
            I::I64LtU,
            I::I32And,
            I::If(BlockType::Result(V::I32)),
            I::LocalGet(0),
            I::I32WrapI64,
            I::I32Load(mem_arg(0, 2)),
            I::Else,
            I::I32Const(-1),
            I::End,
            I::LocalSet(1),
            I::LocalGet(1),
            I::I32Const(TAG_RAW as i32),
            I::I32Eq,
            I::LocalGet(1),
            I::I32Const(TAG_STR as i32),
            I::I32Eq,
            I::I32Or,
            I::I32Eqz,
            I::LocalGet(1),
            I::I32Const(0),
            I::I32GtS,
            I::I32And,
            I::If(BlockType::Empty),
            I::I32Const(obj_obj),
            I::Return,
            I::End,
            // not a tagged object → render as int digits
            I::LocalGet(0),
            I::Call(helpers[H_I64_STR]),
        ] {
            f.instruction(&i);
        }
        push!(H_STR_OR_OBJ, &[V::I64], &[V::I32], f);
    }

    // ---- __key_eq(ka i64, kb i64) -> i32: word eq, str eq by content ----
    {
        let mut f = Function::new(vec![]);
        for i in [
            I::LocalGet(0),
            I::LocalGet(1),
            I::I64Eq,
            I::If(BlockType::Result(V::I32)),
            I::I32Const(1),
            I::Else,
            I::LocalGet(0),
            I::Call(helpers[H_IS_STR]),
            I::If(BlockType::Result(V::I32)),
            I::LocalGet(1),
            I::Call(helpers[H_IS_STR]),
            I::If(BlockType::Result(V::I32)),
            I::LocalGet(0),
            I::I32WrapI64,
            I::LocalGet(1),
            I::I32WrapI64,
            I::Call(helpers[H_STR_EQ]),
            I::Else,
            I::I32Const(0),
            I::End,
            I::Else,
            I::I32Const(0),
            I::End,
            I::End,
        ] {
            f.instruction(&i);
        }
        push!(H_KEY_EQ, &[V::I64, V::I64], &[V::I32], f);
    }

    // ---- __key_hash(k i64) -> i32: fnv-1a on str bytes, mix on words ----
    // locals 1=a,2=len,3=i,4=h,5=data
    {
        let mut f = Function::new(vec![(5, V::I32)]);
        for i in [
            I::LocalGet(0),
            I::Call(helpers[H_IS_STR]),
            I::If(BlockType::Empty),
            I::LocalGet(0),
            I::I32WrapI64,
            I::LocalSet(1),
            I::LocalGet(1),
            I::I32Load(mem_arg(4, 2)),
            I::LocalSet(5),
            I::LocalGet(1),
            I::I32Load(mem_arg(8, 2)),
            I::LocalSet(2),
            I::I32Const(-2128831035i32),
            I::LocalSet(4),
            I::Block(BlockType::Empty),
            I::Loop(BlockType::Empty),
            I::LocalGet(3),
            I::LocalGet(2),
            I::I32GeU,
            I::BrIf(1),
            I::LocalGet(4),
            I::LocalGet(5),
            I::LocalGet(3),
            I::I32Add,
            I::I32Load8U(mem_arg(0, 0)),
            I::I32Xor,
            I::I32Const(16777619),
            I::I32Mul,
            I::LocalSet(4),
            I::LocalGet(3),
            I::I32Const(1),
            I::I32Add,
            I::LocalSet(3),
            I::Br(0),
            I::End,
            I::End,
            I::LocalGet(4),
            I::Return,
            I::End,
            // raw word mix
            I::LocalGet(0),
            I::LocalGet(0),
            I::I64Const(32),
            I::I64ShrU,
            I::I64Xor,
            I::I32WrapI64,
            I::I32Const(-1640531535i32),
            I::I32Mul,
            I::LocalSet(4),
            I::LocalGet(4),
            I::I32Const(13),
            I::I32ShrU,
            I::LocalGet(4),
            I::I32Xor,
        ] {
            f.instruction(&i);
        }
        push!(H_KEY_HASH, &[V::I64], &[V::I32], f);
    }

    // ---- __dict_find(d i32, key i64) -> i32: slot ptr or 0 ----
    // slots: [key i64][val i64][state i32][pad] = 24B in a TAG_RAW region.
    // locals 2=cap,3=bk,4=i,5=start,6=slot,7=r
    {
        let mut f = Function::new(vec![(6, V::I32)]);
        for i in [
            I::LocalGet(0),
            I::I32Load(mem_arg(8, 2)),
            I::LocalSet(2),
            I::LocalGet(2),
            I::I32Eqz,
            I::If(BlockType::Empty),
            I::I32Const(0),
            I::Return,
            I::End,
            I::LocalGet(0),
            I::I32Load(mem_arg(12, 2)),
            I::I32Const(8),
            I::I32Add,
            I::LocalSet(3),
            I::LocalGet(1),
            I::Call(helpers[H_KEY_HASH]),
            I::LocalGet(2),
            I::I32Const(1),
            I::I32Sub,
            I::I32And,
            I::LocalSet(4),
            I::LocalGet(4),
            I::LocalSet(5),
            I::Block(BlockType::Empty),
            I::Loop(BlockType::Empty),
            I::LocalGet(3),
            I::LocalGet(4),
            I::I32Const(24),
            I::I32Mul,
            I::I32Add,
            I::LocalSet(6),
            I::LocalGet(6),
            I::I32Load(mem_arg(16, 2)),
            I::I32Eqz,
            I::If(BlockType::Empty),
            // empty slot → key absent → break out of the Block
            I::Br(2),
            I::End,
            I::LocalGet(6),
            I::I32Load(mem_arg(16, 2)),
            I::I32Const(1),
            I::I32Eq,
            I::If(BlockType::Empty),
            I::LocalGet(6),
            I::I64Load(mem_arg(0, 3)),
            I::LocalGet(1),
            I::Call(helpers[H_KEY_EQ]),
            I::If(BlockType::Empty),
            I::LocalGet(6),
            I::LocalSet(7),
            // two Ifs deep — Br(3) exits the Block
            I::Br(3),
            I::End,
            I::End,
            I::LocalGet(4),
            I::I32Const(1),
            I::I32Add,
            I::LocalGet(2),
            I::I32Const(1),
            I::I32Sub,
            I::I32And,
            I::LocalSet(4),
            I::LocalGet(4),
            I::LocalGet(5),
            I::I32Ne,
            I::BrIf(0),
            I::End,
            I::End,
            I::LocalGet(7),
        ] {
            f.instruction(&i);
        }
        push!(H_DICT_FIND, &[V::I32, V::I64], &[V::I32], f);
    }

    // ---- __dict_set(d i32, key i64, val i64): insert or overwrite ----
    // grows at 70% load; buckets region [TAG_RAW][pad][slots…]
    // locals 3=slot,4=cap,5=bk,6=i,7=_,8=nb,9=ncap,10=oi,11=obk,12=nbk,13=os,14=s2
    {
        let mut f = Function::new(vec![(12, V::I32)]);
        for i in [
            // existing key → overwrite
            I::LocalGet(0),
            I::LocalGet(1),
            I::Call(helpers[H_DICT_FIND]),
            I::LocalSet(3),
            I::LocalGet(3),
            I::If(BlockType::Empty),
            I::LocalGet(3),
            I::LocalGet(2),
            I::I64Store(mem_arg(8, 3)),
            I::Return,
            I::End,
            I::LocalGet(0),
            I::I32Load(mem_arg(8, 2)),
            I::LocalSet(4),
            // grow when cap==0 or (size+1)*10 > cap*7
            I::LocalGet(4),
            I::I32Eqz,
            I::LocalGet(0),
            I::I32Load(mem_arg(4, 2)),
            I::I32Const(1),
            I::I32Add,
            I::I32Const(10),
            I::I32Mul,
            I::LocalGet(4),
            I::I32Const(7),
            I::I32Mul,
            I::I32GtU,
            I::I32Or,
            I::If(BlockType::Empty),
            // ncap = cap ? cap*2 : 8
            I::LocalGet(4),
            I::I32Eqz,
            I::If(BlockType::Result(V::I32)),
            I::I32Const(8),
            I::Else,
            I::LocalGet(4),
            I::I32Const(2),
            I::I32Mul,
            I::End,
            I::LocalSet(9),
            I::I32Const(8),
            I::LocalGet(9),
            I::I32Const(24),
            I::I32Mul,
            I::I32Add,
            I::Call(helpers[H_ALLOC]),
            I::LocalSet(8),
            I::LocalGet(8),
            I::I32Const(TAG_RAW as i32),
            I::I32Store(mem_arg(0, 2)),
            // reinsert non-empty slots from the old table
            I::LocalGet(4),
            I::If(BlockType::Empty),
            I::LocalGet(0),
            I::I32Load(mem_arg(12, 2)),
            I::I32Const(8),
            I::I32Add,
            I::LocalSet(11),
            I::LocalGet(8),
            I::I32Const(8),
            I::I32Add,
            I::LocalSet(12),
            I::I32Const(0),
            I::LocalSet(10),
            I::Block(BlockType::Empty),
            I::Loop(BlockType::Empty),
            I::LocalGet(10),
            I::LocalGet(4),
            I::I32GeU,
            I::BrIf(1),
            I::LocalGet(11),
            I::LocalGet(10),
            I::I32Const(24),
            I::I32Mul,
            I::I32Add,
            I::LocalSet(13),
            I::LocalGet(13),
            I::I32Load(mem_arg(16, 2)),
            I::I32Const(1),
            I::I32Eq,
            I::If(BlockType::Empty),
            I::LocalGet(13),
            I::I64Load(mem_arg(0, 3)),
            I::Call(helpers[H_KEY_HASH]),
            I::LocalGet(9),
            I::I32Const(1),
            I::I32Sub,
            I::I32And,
            I::LocalSet(6),
            I::Block(BlockType::Empty),
            I::Loop(BlockType::Empty),
            I::LocalGet(12),
            I::LocalGet(6),
            I::I32Const(24),
            I::I32Mul,
            I::I32Add,
            I::LocalSet(14),
            I::LocalGet(14),
            I::I32Load(mem_arg(16, 2)),
            I::I32Eqz,
            I::If(BlockType::Empty),
            I::LocalGet(14),
            I::LocalGet(13),
            I::I64Load(mem_arg(0, 3)),
            I::I64Store(mem_arg(0, 3)),
            I::LocalGet(14),
            I::LocalGet(13),
            I::I64Load(mem_arg(8, 3)),
            I::I64Store(mem_arg(8, 3)),
            I::LocalGet(14),
            I::I32Const(1),
            I::I32Store(mem_arg(16, 2)),
            // Br(2) exits the inner probe Block → next old entry
            I::Br(2),
            I::End,
            I::LocalGet(6),
            I::I32Const(1),
            I::I32Add,
            I::LocalGet(9),
            I::I32Const(1),
            I::I32Sub,
            I::I32And,
            I::LocalSet(6),
            I::Br(0),
            I::End,
            I::End,
            I::End,
            I::LocalGet(10),
            I::I32Const(1),
            I::I32Add,
            I::LocalSet(10),
            I::Br(0),
            I::End,
            I::End,
            I::End,
            I::LocalGet(0),
            I::LocalGet(9),
            I::I32Store(mem_arg(8, 2)),
            I::LocalGet(0),
            I::LocalGet(8),
            I::I32Store(mem_arg(12, 2)),
            I::End,
            // probe for an empty slot, write key/val, size += 1
            I::LocalGet(0),
            I::I32Load(mem_arg(8, 2)),
            I::LocalSet(4),
            I::LocalGet(0),
            I::I32Load(mem_arg(12, 2)),
            I::I32Const(8),
            I::I32Add,
            I::LocalSet(5),
            I::LocalGet(1),
            I::Call(helpers[H_KEY_HASH]),
            I::LocalGet(4),
            I::I32Const(1),
            I::I32Sub,
            I::I32And,
            I::LocalSet(6),
            I::Loop(BlockType::Empty),
            I::LocalGet(5),
            I::LocalGet(6),
            I::I32Const(24),
            I::I32Mul,
            I::I32Add,
            I::LocalSet(3),
            I::LocalGet(3),
            I::I32Load(mem_arg(16, 2)),
            I::I32Eqz,
            I::If(BlockType::Empty),
            I::LocalGet(3),
            I::LocalGet(1),
            I::I64Store(mem_arg(0, 3)),
            I::LocalGet(3),
            I::LocalGet(2),
            I::I64Store(mem_arg(8, 3)),
            I::LocalGet(3),
            I::I32Const(1),
            I::I32Store(mem_arg(16, 2)),
            // record the key in the insertion-ordered `order` array so
            // d[i] can replay the interpreter's IndexMap order; grow it
            // (8B per key) when size == ecap
            I::LocalGet(0),
            I::I32Load(mem_arg(4, 2)),
            I::LocalGet(0),
            I::I32Load(mem_arg(16, 2)),
            I::I32Eq,
            I::If(BlockType::Empty),
            I::LocalGet(0),
            I::I32Load(mem_arg(16, 2)),
            I::If(BlockType::Result(V::I32)),
            I::LocalGet(0),
            I::I32Load(mem_arg(16, 2)),
            I::I32Const(2),
            I::I32Mul,
            I::Else,
            I::I32Const(8),
            I::End,
            I::LocalSet(9),
            I::LocalGet(9),
            I::I32Const(8),
            I::I32Mul,
            I::Call(helpers[H_ALLOC]),
            I::LocalSet(8),
            I::LocalGet(8),
            I::LocalGet(0),
            I::I32Load(mem_arg(20, 2)),
            I::LocalGet(0),
            I::I32Load(mem_arg(4, 2)),
            I::I32Const(8),
            I::I32Mul,
            I::MemoryCopy {
                src_mem: 0,
                dst_mem: 0,
            },
            I::LocalGet(0),
            I::LocalGet(9),
            I::I32Store(mem_arg(16, 2)),
            I::LocalGet(0),
            I::LocalGet(8),
            I::I32Store(mem_arg(20, 2)),
            I::End,
            // order[size] = key; size += 1
            I::LocalGet(0),
            I::I32Load(mem_arg(20, 2)),
            I::LocalGet(0),
            I::I32Load(mem_arg(4, 2)),
            I::I32Const(8),
            I::I32Mul,
            I::I32Add,
            I::LocalGet(1),
            I::I64Store(mem_arg(0, 3)),
            I::LocalGet(0),
            I::LocalGet(0),
            I::I32Load(mem_arg(4, 2)),
            I::I32Const(1),
            I::I32Add,
            I::I32Store(mem_arg(4, 2)),
            I::Return,
            I::End,
            I::LocalGet(6),
            I::I32Const(1),
            I::I32Add,
            I::LocalGet(4),
            I::I32Const(1),
            I::I32Sub,
            I::I32And,
            I::LocalSet(6),
            I::Br(0),
            I::End,
            I::Unreachable,
        ] {
            f.instruction(&i);
        }
        push!(H_DICT_SET, &[V::I32, V::I64, V::I64], &[], f);
    }

    // ---- __dict_entry(d i32, i i32) -> i32: i-th entry as a [key,val] ----
    // pair array — the interpreter's entry_at(i) over an IndexMap, so the
    // insertion-ordered `order` array is what preserves that ordering.
    // locals 2=slot,3=pair
    {
        let mut f = Function::new(vec![(2, V::I32)]);
        for i in [
            // bounds: i >= size → trap
            I::LocalGet(1),
            I::LocalGet(0),
            I::I32Load(mem_arg(4, 2)),
            I::I32GeU,
            I::If(BlockType::Empty),
            I::Unreachable,
            I::End,
            // slot = __dict_find(d, order[i]) — always hits
            I::LocalGet(0),
            I::LocalGet(0),
            I::I32Load(mem_arg(20, 2)),
            I::LocalGet(1),
            I::I32Const(8),
            I::I32Mul,
            I::I32Add,
            I::I64Load(mem_arg(0, 3)),
            I::Call(helpers[H_DICT_FIND]),
            I::LocalSet(2),
            // pair = alloc(32) tagged array of 2 words
            I::I32Const(32),
            I::Call(helpers[H_ALLOC]),
            I::LocalSet(3),
            I::LocalGet(3),
            I::I32Const(TAG_ARRAY as i32),
            I::I32Store(mem_arg(0, 2)),
            I::LocalGet(3),
            I::LocalGet(3),
            I::I32Const(16),
            I::I32Add,
            I::I32Store(mem_arg(4, 2)),
            I::LocalGet(3),
            I::I32Const(2),
            I::I32Store(mem_arg(8, 2)),
            I::LocalGet(3),
            I::I32Const(2),
            I::I32Store(mem_arg(12, 2)),
            // pair[0] = slot.key; pair[1] = slot.val
            I::LocalGet(3),
            I::I32Load(mem_arg(4, 2)),
            I::LocalGet(2),
            I::I64Load(mem_arg(0, 3)),
            I::I64Store(mem_arg(0, 3)),
            I::LocalGet(3),
            I::I32Load(mem_arg(4, 2)),
            I::LocalGet(2),
            I::I64Load(mem_arg(8, 3)),
            I::I64Store(mem_arg(8, 3)),
            I::LocalGet(3),
        ] {
            f.instruction(&i);
        }
        push!(H_DICT_ENTRY, &[V::I32, V::I32], &[V::I32], f);
    }

    // ---- cov::* sink helpers (emitted only when the sink is live) ----
    // Ported from the resumable lane: pure memory ops on the decision
    // cells, the cmp operand stack, and the dec record ring.
    if let Some(c) = cov {
        let decv = c.decv_base as i32;
        let opstk = c.opstk_base as i32;
        let sinkb = c.sink_base as i32;
        let (g_osp, g_covp) = (c.g_osp, c.g_covp);
        let (f_gap, f_near, f_bit) = (c.h[5], c.h[6], c.h[7]);

        // __cov_leaf(x f64, n i32): opstk[__osp] = (x, n); __osp++
        {
            let mut f = Function::new(vec![(1, V::I32)]);
            for i in [
                I::I32Const(opstk),
                I::GlobalGet(g_osp),
                I::I32Const(16),
                I::I32Mul,
                I::I32Add,
                I::LocalSet(2),
                I::LocalGet(2),
                I::LocalGet(0),
                I::F64Store(mem_arg(0, 3)),
                I::LocalGet(2),
                I::LocalGet(1),
                I::I32Store8(mem_arg(8, 0)),
                I::GlobalGet(g_osp),
                I::I32Const(1),
                I::I32Add,
                I::GlobalSet(g_osp),
            ] {
                f.instruction(&i);
            }
            push!(H_COV_LEAF, &[V::F64, V::I32], &[], f);
        }
        // __cov_begin(d): leafbits[d] = 0
        {
            let mut f = Function::new(vec![]);
            for i in [
                I::I32Const(decv),
                I::LocalGet(0),
                I::I32Const(DCELL as i32),
                I::I32Mul,
                I::I32Add,
                I::I64Const(0),
                I::I64Store(mem_arg(0, 3)),
            ] {
                f.instruction(&i);
            }
            push!(H_COV_BEGIN, &[V::I32], &[], f);
        }
        // __cov_cond(d, k, v): flag-near + leafbit
        {
            let mut f = Function::new(vec![]);
            for i in [
                I::LocalGet(0),
                I::LocalGet(1),
                I::LocalGet(2),
                I::If(BlockType::Result(V::F64)),
                I::F64Const(0.0f64.into()),
                I::Else,
                I::F64Const(F64_INF.into()),
                I::End,
                I::LocalGet(2),
                I::If(BlockType::Result(V::F64)),
                I::F64Const(F64_INF.into()),
                I::Else,
                I::F64Const(0.0f64.into()),
                I::End,
                I::Call(f_near),
                I::LocalGet(0),
                I::LocalGet(1),
                I::LocalGet(2),
                I::Call(f_bit),
            ] {
                f.instruction(&i);
            }
            push!(H_COV_COND, &[V::I32; 3], &[], f);
        }
        // __cov_cmp(d, k, op, v): pop 2 operands; numeric -> gap->cellnear,
        // else flag-cellnear; then leafbit
        {
            // locals l4=addr i32, l5=a_num i32, l6=b_num i32, l7=a, l8=b
            let mut f = Function::new(vec![(3, V::I32), (2, V::F64)]);
            for i in [
                I::GlobalGet(g_osp),
                I::I32Const(2),
                I::I32Sub,
                I::GlobalSet(g_osp),
                I::I32Const(opstk),
                I::GlobalGet(g_osp),
                I::I32Const(16),
                I::I32Mul,
                I::I32Add,
                I::LocalSet(4),
                I::LocalGet(4),
                I::F64Load(mem_arg(0, 3)),
                I::LocalSet(7),
                I::LocalGet(4),
                I::I32Load8U(mem_arg(8, 0)),
                I::LocalSet(5),
                I::LocalGet(4),
                I::F64Load(mem_arg(16, 3)),
                I::LocalSet(8),
                I::LocalGet(4),
                I::I32Load8U(mem_arg(24, 0)),
                I::LocalSet(6),
                I::LocalGet(5),
                I::LocalGet(6),
                I::I32And,
                I::If(BlockType::Empty),
                I::LocalGet(0),
                I::LocalGet(1),
                I::LocalGet(7),
                I::LocalGet(8),
                I::LocalGet(2),
                I::Call(f_gap),
                I::Call(f_near),
                I::Else,
                I::LocalGet(0),
                I::LocalGet(1),
                I::LocalGet(3),
                I::If(BlockType::Result(V::F64)),
                I::F64Const(0.0f64.into()),
                I::Else,
                I::F64Const(F64_INF.into()),
                I::End,
                I::LocalGet(3),
                I::If(BlockType::Result(V::F64)),
                I::F64Const(F64_INF.into()),
                I::Else,
                I::F64Const(0.0f64.into()),
                I::End,
                I::Call(f_near),
                I::End,
                I::LocalGet(0),
                I::LocalGet(1),
                I::LocalGet(3),
                I::Call(f_bit),
            ] {
                f.instruction(&i);
            }
            push!(H_COV_CMP, &[V::I32; 4], &[], f);
        }
        // __cov_dec(id, d, v): 16B record [d u32][v u32][leafbits u64] at
        // sink_base + __covp*16; __covp++
        {
            let mut f = Function::new(vec![(1, V::I32)]);
            for i in [
                I::I32Const(sinkb),
                I::GlobalGet(g_covp),
                I::I32Const(16),
                I::I32Mul,
                I::I32Add,
                I::LocalSet(3),
                I::LocalGet(3),
                I::LocalGet(1),
                I::I32Store(mem_arg(0, 2)),
                I::LocalGet(3),
                I::LocalGet(2),
                I::I32Store(mem_arg(4, 2)),
                I::LocalGet(3),
                I::I32Const(decv),
                I::LocalGet(1),
                I::I32Const(DCELL as i32),
                I::I32Mul,
                I::I32Add,
                I::I64Load(mem_arg(0, 3)),
                I::I64Store(mem_arg(8, 3)),
                I::GlobalGet(g_covp),
                I::I32Const(1),
                I::I32Add,
                I::GlobalSet(g_covp),
            ] {
                f.instruction(&i);
            }
            push!(H_COV_DEC, &[V::I32; 3], &[], f);
        }
        // __cov_gap(a, b, op) -> (t, f): coverage.rs's gap() verbatim,
        // results land in locals 3/4 (multi-value if-arms would need a
        // module type index we don't register here)
        {
            // locals: l3=t f64, l4=f f64
            let mut f = Function::new(vec![(2, V::F64)]);
            let one = BlockType::Result(V::F64);
            let pos = |f: &mut Function| {
                f.instruction(&I::F64Const(0.0f64.into()));
                f.instruction(&I::F64Max);
            };
            let cond_eps = |f: &mut Function| {
                f.instruction(&I::If(one));
                f.instruction(&I::F64Const(EPS.into()));
                f.instruction(&I::Else);
                f.instruction(&I::F64Const(0.0f64.into()));
                f.instruction(&I::End);
            };
            // each arm pushes t then f; LocalSet pops f first
            let settle = |f: &mut Function| {
                f.instruction(&I::LocalSet(4));
                f.instruction(&I::LocalSet(3));
            };
            for op in 0..6 {
                if op < 5 {
                    f.instruction(&I::LocalGet(2));
                    f.instruction(&I::I32Const(op));
                    f.instruction(&I::I32Eq);
                    f.instruction(&I::If(BlockType::Empty));
                }
                match op {
                    // Eq: t=|a-b|, f=a==b?eps:0
                    0 => {
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::F64Sub);
                        f.instruction(&I::F64Abs);
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::F64Eq);
                        cond_eps(&mut f);
                        settle(&mut f);
                    }
                    // Ne: t=a==b?eps:0, f=|a-b|
                    1 => {
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::F64Eq);
                        cond_eps(&mut f);
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::F64Sub);
                        f.instruction(&I::F64Abs);
                        settle(&mut f);
                    }
                    // Gt: t=pos(b-a)+(a<=b?eps:0), f=pos(a-b)
                    2 => {
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::F64Sub);
                        pos(&mut f);
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::F64Le);
                        cond_eps(&mut f);
                        f.instruction(&I::F64Add);
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::F64Sub);
                        pos(&mut f);
                        settle(&mut f);
                    }
                    // Ge: t=pos(b-a), f=pos(a-b)+(a>=b?eps:0)
                    3 => {
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::F64Sub);
                        pos(&mut f);
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::F64Sub);
                        pos(&mut f);
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::F64Ge);
                        cond_eps(&mut f);
                        f.instruction(&I::F64Add);
                        settle(&mut f);
                    }
                    // Lt: t=pos(a-b)+(a>=b?eps:0), f=pos(b-a)
                    4 => {
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::F64Sub);
                        pos(&mut f);
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::F64Ge);
                        cond_eps(&mut f);
                        f.instruction(&I::F64Add);
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::F64Sub);
                        pos(&mut f);
                        settle(&mut f);
                    }
                    // Le (default): t=pos(a-b), f=pos(b-a)+(a<=b?eps:0)
                    _ => {
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::F64Sub);
                        pos(&mut f);
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::F64Sub);
                        pos(&mut f);
                        f.instruction(&I::LocalGet(0));
                        f.instruction(&I::LocalGet(1));
                        f.instruction(&I::F64Le);
                        cond_eps(&mut f);
                        f.instruction(&I::F64Add);
                        settle(&mut f);
                    }
                }
                if op < 5 {
                    f.instruction(&I::Else);
                }
            }
            for _ in 0..5 {
                f.instruction(&I::End);
            }
            f.instruction(&I::LocalGet(3));
            f.instruction(&I::LocalGet(4));
            push!(H_COV_GAP, &[V::F64, V::F64, V::I32], &[V::F64; 2], f);
        }
        // __cov_cellnear(d, k, t, f): min-merge into the dist cell pair
        {
            let mut f = Function::new(vec![(1, V::I32)]);
            for i in [
                I::I32Const(decv),
                I::LocalGet(0),
                I::I32Const(DCELL as i32),
                I::I32Mul,
                I::I32Add,
                I::LocalGet(1),
                I::I32Const(8),
                I::I32Mul,
                I::I32Add,
                I::I32Const(8),
                I::I32Add,
                I::LocalSet(4),
                I::LocalGet(4),
                I::LocalGet(4),
                I::F64Load(mem_arg(0, 3)),
                I::LocalGet(2),
                I::F64Min,
                I::F64Store(mem_arg(0, 3)),
                I::LocalGet(4),
                I::LocalGet(4),
                I::F64Load(mem_arg(64, 3)),
                I::LocalGet(3),
                I::F64Min,
                I::F64Store(mem_arg(64, 3)),
            ] {
                f.instruction(&i);
            }
            push!(H_COV_NEAR, &[V::I32, V::I32, V::F64, V::F64], &[], f);
        }
        // __cov_leafbit(d, k, v): leafbits[d] |= (v ? 1 : 2) << (k*2)
        {
            let mut f = Function::new(vec![(1, V::I32)]);
            for i in [
                I::I32Const(decv),
                I::LocalGet(0),
                I::I32Const(DCELL as i32),
                I::I32Mul,
                I::I32Add,
                I::LocalSet(3),
                I::LocalGet(3),
                I::LocalGet(3),
                I::I64Load(mem_arg(0, 3)),
                I::I32Const(2),
                I::LocalGet(2),
                I::I32Sub,
                I::I64ExtendI32U,
                I::LocalGet(1),
                I::I32Const(2),
                I::I32Mul,
                I::I64ExtendI32U,
                I::I64Shl,
                I::I64Or,
                I::I64Store(mem_arg(0, 3)),
            ] {
                f.instruction(&i);
            }
            push!(H_COV_BIT, &[V::I32; 3], &[], f);
        }
    }

    out
}

/// The call_indirect trampoline for body `b`: (env, args_ptr, nargs) -> i64.
/// Unmarshals captures out of the env object then args out of the scratch
/// words, calls the body, and marshals its result back to a word.
pub(crate) fn emit_trampoline(body: &IrBody, sig: &Sig, callee_fi: u32) -> Function {
    use Instruction as I;
    let ncaps = body.captures.len();
    let nparams = body.params.len();
    let mut f = Function::new(vec![]);
    for i in [
        I::LocalGet(2),
        I::I32Const(nparams as i32),
        I::I32Ne,
        I::If(BlockType::Empty),
        I::Unreachable,
        I::End,
    ] {
        f.instruction(&i);
    }
    let unmarshal = |f: &mut Function, k: K| match k {
        K::Int | K::Word => {}
        K::Float => {
            f.instruction(&I::F64ReinterpretI64);
        }
        K::Bool => {
            f.instruction(&I::I32WrapI64);
        }
    };
    for i in 0..ncaps {
        for ins in [
            I::LocalGet(0),
            I::I64Load(mem_arg(16 + i as u32 * 8, 3)),
        ] {
            f.instruction(&ins);
        }
        unmarshal(&mut f, sig.params[i]);
    }
    for i in 0..nparams {
        for ins in [I::LocalGet(1), I::I64Load(mem_arg(i as u32 * 8, 3))] {
            f.instruction(&ins);
        }
        unmarshal(&mut f, sig.params[ncaps + i]);
    }
    f.instruction(&I::Call(callee_fi));
    match sig.ret {
        None => {
            f.instruction(&I::I64Const(0));
        }
        Some(K::Int | K::Word) => {}
        Some(K::Float) => {
            f.instruction(&I::I64ReinterpretF64);
        }
        Some(K::Bool) => {
            f.instruction(&I::I64ExtendI32U);
        }
    }
    f.instruction(&I::End);
    f
}

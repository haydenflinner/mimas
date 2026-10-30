//! `wrt` — the shared runtime substrate for the waffle lane (`wfull`):
//! the class analysis (`analyze_body`), heap-object tags and layout
//! constants, the statics/data-segment bookkeeping, and the hand-coded
//! helper functions (`emit_helpers`) and dynamic-call trampolines
//! (`emit_trampoline`) spliced into the module as `FuncDecl::Compiled`.
//!
//! Classes live on SSA `InstId`s, not bytecode `Reg`s — a register written
//! Int in one arm and Float in another used to union to an unusable mask;
//! the two `InstId`s each get their own class.

use std::collections::{HashMap, HashSet};

use compile::{BinOp, Body as IrBody, Constant, FormatPart, Inst, InstId, OperandKind, UnaryOp};
use wasm_encoder::{BlockType, Function, Instruction, MemArg, ValType};

use crate::{Bail, K, K_BOOL, K_FLOAT, K_INT, Sig, kbit};

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
    /// cursor for lazily-baked const-array objects, past the fixed statics
    pub arr_cur: std::cell::Cell<u32>,
    /// end of the statics hole — const arrays must stay under this
    pub arr_cap: u32,
    /// baked element-words -> object address (dedup)
    pub arr_objs: std::cell::RefCell<HashMap<Vec<u8>, u32>>,
    /// segments appended during body emission; merged into the module's
    /// memory segments after the last body emits
    pub arr_statics: std::cell::RefCell<Vec<(u32, Vec<u8>)>>,
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
                        // "any materialization" like Format — neg/abs pick the
                        // operand's own class, so the demand must not pin it
                        // to one (but the operand must still materialize)
                        UnaryOp::Negative | UnaryOp::Positive => K_INT | K_FLOAT | K_BOOL | K_WORD,
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
                Inst::SetIndex { set, index, value } => {
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
                    Inst::UnaryOp { op, right } => {
                        (match op {
                            UnaryOp::Negative | UnaryOp::Positive => snap_mask[right.index()],
                            UnaryOp::Not => K_BOOL,
                            _ => K_INT,
                        }) | K_WORD
                    }
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
        for ins in [I::LocalGet(0), I::I64Load(mem_arg(16 + i as u32 * 8, 3))] {
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

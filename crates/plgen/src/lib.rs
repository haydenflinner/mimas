//! `.pliron` (pliron LLVM-dialect) emitter for mimas bytecode.
//!
//! One `llvm.func` per `Program` chunk at the [`vm::bc::BodyFn`] ABI plus a
//! `@mimas_pliron_bodies`/`@mimas_pliron_nbodies` global pair the host reads
//! after linking and feeds to `Vm::install_bc`. The emitted code is a text
//! port of the Cranelift JIT tier (`jit/src/emit.rs`): scalar register
//! shadows, the batched `bcn` quota, the shared exit blocks and the
//! `step_at` op fallback all mirror it 1:1. `Switch` and `Constant::Array`/
//! `Dec` still go through `step` — everything else either lowers inline or
//! delegates to the same `mj_*` extern-C shims.
//!
//! Helper calls come out as direct `llvm.call @mj_*` against bodiless
//! declarations, so the object links by name against any companion runtime
//! exporting the `vm::bc::nat` symbols (`mimas-natrt`'s staticlib/cdylib) —
//! no baked addresses, a real symbol seam.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;

use compile::{AccessKind, BlockTarget, Constant, Op, OpFormatPart, Program, Reg};
use vm::bc::{BodyEnv, INLINE_CALL_DEPTH, jit::Layout};

// `RtErr` codes, matching `jit/src/emit.rs` (vm::bc::RtErr discriminants).
const ERR_PANIC: i64 = 0; // MatchPanicReached
const ERR_MOD0: i64 = 2; // ModByZero
const ERR_OVFW: i64 = 7; // IntegerOverflow
const ERR_OOF: i64 = 10; // OutOfFuel

// ---- pliron type spellings ----
const I64: &str = "builtin.integer i64";
const I32: &str = "builtin.integer i32";
const I16: &str = "builtin.integer i16";
const I8: &str = "builtin.integer i8";
const I1: &str = "builtin.integer i1";
const F64: &str = "builtin.fp64";
const PTR: &str = "llvm.ptr (0)";
const VOID: &str = "llvm.void";

/// The exported name of the function table the host feeds `install_bc`.
pub const BODIES_GLOBAL: &str = "mimas_pliron_bodies";
/// The exported name of the `i64` body count global.
pub const NBODIES_GLOBAL: &str = "mimas_pliron_nbodies";
/// Exported initializer: fills `mimas_pliron_bodies` from `llvm.addressof`
/// — a real address natively, the `GOT.func` table index on wasm, where
/// no host-side `dlsym` fill exists.
pub const INIT_FN: &str = "mimas_pliron_init";
/// Exported `() -> i64`: the `Library::fingerprint_install` of the
/// install the emitted `NativeId`s were numbered against. A runtime
/// whose own install fingerprints differently must refuse to run — the
/// baked ids would index a shifted native table.
pub const NSIG_FN: &str = "mimas_pliron_nsig";
/// `mimas_pliron_init`'s single entry block.
const INIT_ENTRY: &str = "init_entry";
/// `mimas_body_{index}` — the per-index entry trampoline. Raw emitted
/// bodies also carry this name before [`emit_bodies`] renames them to
/// content-hash symbols; in the assembled module the name belongs to
/// the trampoline alone, and the table's slot `i` points at it.
pub fn body_name(i: usize) -> String {
    format!("mimas_body_{i}")
}

/// `mimas_body_x{h:016x}` — the content-addressed body symbol a chunk
/// declares and call sites name. The `mimas_body_off_x{h}` /
/// `mimas_body_idx_x{h}` globals are hash-named too, defined once per
/// unique body with the canonical index's values; `mimas_body_{i}`
/// survives only as the per-index entry trampoline the init module
/// emits (see `assemble_module_ex`).
pub fn body_sym(h: u64) -> String {
    format!("mimas_body_x{h:016x}")
}

/// Every `mimas_body_*` name normalized to one token, so hashing sees
/// chunk content only — never a program-assigned index or another
/// body's hash.
fn normalize_names(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(p) = rest.find("mimas_body_") {
        let (head, tail) = rest.split_at(p + "mimas_body_".len());
        out.push_str(head);
        let end = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(tail.len());
        out.push('#');
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

/// Content hash of an emitted body — same source, same hash, wherever
/// `compile_parts` placed the chunk.
pub fn body_hash(text: &str) -> u64 {
    use rustc_hash::FxHasher;
    use std::hash::{Hash, Hasher};
    let mut h = FxHasher::default();
    normalize_names(text).hash(&mut h);
    h.finish()
}

/// `mimas_body_{i}` / `mimas_body_off_{i}` / `mimas_body_idx_{i}` →
/// their content-hash names, one pass. Hash names carry an `x` prefix
/// (`mimas_body_x{h:016x}`) so a hex tail can never re-parse as an
/// index. Content-identical chunks alias to one canonical index — the
/// `off`/`idx` globals are defined once per unique hash with that
/// index's values, and `tbl[i]` fills every index with its entry
/// trampoline's address.
fn rename_body_names(text: &str, hashes: &[u64]) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(p) = rest.find("mimas_body_") {
        out.push_str(&rest[..p]);
        let tail = &rest[p + "mimas_body_".len()..];
        let end = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(tail.len());
        let tok = &tail[..end];
        rest = &tail[end..];
        let mapped = if let Some(d) = tok.strip_prefix("off_") {
            d.parse::<usize>()
                .ok()
                .map(|i| format!("mimas_body_off_x{:016x}", hashes[i]))
        } else if let Some(d) = tok.strip_prefix("idx_") {
            d.parse::<usize>()
                .ok()
                .map(|i| format!("mimas_body_idx_x{:016x}", hashes[i]))
        } else {
            tok.parse::<usize>()
                .ok()
                .map(|i| body_sym(hashes[i]))
        };
        match mapped {
            Some(m) => out.push_str(&m),
            None => {
                out.push_str("mimas_body_");
                out.push_str(tok);
            }
        }
    }
    out.push_str(rest);
    out
}

// ---- `vm::bc::jit` helper ABI table (the subset this emitter calls) ----
//
// Mirrors `jit/src/lib.rs::SPECS`: `P` is a pointer-width word (ptr or
// usize), `Ctx` expands to the two `Ctx` words, `I8`/`I32`/`F64` are real
// narrow params. The emitted callee type uses `llvm.ptr (0)` for `P` —
// `usize`s cross as `inttoptr`'d words, identical in the C ABI.

#[derive(Clone, Copy)]
enum Pt {
    P,
    I8,
    I32,
    /// helpers this emitter doesn't call yet (`mj_wr_f`) — kept so the
    /// table's shape mirrors `jit::SPECS`.
    #[allow(dead_code)]
    F64,
    Ctx,
}

#[derive(Clone, Copy)]
enum Hr {
    Void,
    U8,
    P,
}

#[derive(Clone, Copy, PartialEq)]
enum H {
    StepAt,
    OutNext,
    OutErr,
    OutReturn,
    NewArray,
    NewDict,
    NewInstance,
    NewClosure,
    Push,
    Insert,
    Format,
    SetIndex,
    GetIndex,
    SetField,
    GetField,
    ContainsOp,
    IsInstance,
    IsRaised,
    UnwrapRaised,
    Unwrap,
    UnwrapUnit,
    Len,
    Bin,
    Unary,
    CallNative,
    Raise,
    LoadConstStr,
    BinStr,
    CallBody,
    CallDyn,
}

use Pt::Ctx;

/// `(id, spec name, params, ret)` — names match `jit::SPECS` (`mj_*`) so the
/// driver can map `jit::X as *const u8 as usize` per entry.
static SPECS: &[(H, &str, &[Pt], Hr)] = &[
    (
        H::StepAt,
        "mj_step_at",
        &[Pt::P, Pt::P, Pt::P, Pt::P, Ctx, Pt::P, Pt::P],
        Hr::U8,
    ),
    (H::OutNext, "mj_out_next", &[Pt::P], Hr::Void),
    (H::OutErr, "mj_out_err", &[Pt::P, Pt::I8], Hr::Void),
    (H::OutReturn, "mj_out_return", &[Pt::P, Pt::P], Hr::Void),
    (H::NewArray, "mj_new_array", &[Pt::P, Pt::P, Ctx], Hr::Void),
    (H::NewDict, "mj_new_dict", &[Pt::P, Pt::P, Ctx], Hr::Void),
    (
        H::NewInstance,
        "mj_new_instance",
        &[Pt::P, Pt::P, Pt::I32, Pt::P, Pt::P, Ctx],
        Hr::Void,
    ),
    (
        H::NewClosure,
        "mj_new_closure",
        &[Pt::P, Pt::P, Pt::I32, Pt::P, Pt::P, Ctx],
        Hr::Void,
    ),
    (H::Push, "mj_push", &[Pt::P, Pt::P, Pt::P, Ctx], Hr::Void),
    (
        H::Insert,
        "mj_insert",
        &[Pt::P, Pt::P, Pt::I32, Pt::P, Ctx, Pt::P],
        Hr::Void,
    ),
    (
        H::Format,
        "mj_format",
        &[Pt::P, Pt::P, Pt::P, Ctx, Pt::P, Pt::P],
        Hr::U8,
    ),
    (
        H::SetIndex,
        "mj_set_index",
        &[Pt::P, Pt::P, Pt::P, Pt::P, Ctx, Pt::P],
        Hr::U8,
    ),
    (
        H::GetIndex,
        "mj_get_index",
        &[Pt::P, Pt::P, Pt::P, Pt::P, Pt::I8, Ctx, Pt::P],
        Hr::U8,
    ),
    (
        H::SetField,
        "mj_set_field",
        &[Pt::P, Pt::P, Pt::P, Pt::P, Ctx, Pt::P],
        Hr::U8,
    ),
    (
        H::GetField,
        "mj_get_field",
        &[Pt::P, Pt::P, Pt::P, Pt::P, Pt::I8, Pt::P],
        Hr::U8,
    ),
    (
        H::ContainsOp,
        "mj_contains",
        &[Pt::P, Pt::P, Pt::P, Pt::P, Pt::I8],
        Hr::Void,
    ),
    (
        H::IsInstance,
        "mj_is_instance",
        &[Pt::P, Pt::P, Pt::P, Pt::I32],
        Hr::Void,
    ),
    (
        H::IsRaised,
        "mj_is_raised",
        &[Pt::P, Pt::P, Pt::P],
        Hr::Void,
    ),
    (
        H::UnwrapRaised,
        "mj_unwrap_raised",
        &[Pt::P, Pt::P, Pt::P],
        Hr::Void,
    ),
    (
        H::Unwrap,
        "mj_unwrap",
        &[Pt::P, Pt::P, Pt::P, Ctx, Pt::P],
        Hr::U8,
    ),
    (
        H::UnwrapUnit,
        "mj_unwrap_unit",
        &[Pt::P, Pt::P, Pt::P, Ctx, Pt::P],
        Hr::U8,
    ),
    (H::Len, "mj_len", &[Pt::P, Pt::P, Pt::P, Pt::P], Hr::U8),
    (
        H::Bin,
        "mj_bin",
        &[Pt::P, Pt::P, Pt::P, Pt::I8, Pt::P, Ctx, Pt::P],
        Hr::U8,
    ),
    (
        H::Unary,
        "mj_unary",
        &[Pt::P, Pt::P, Pt::I8, Pt::P, Ctx, Pt::P],
        Hr::U8,
    ),
    (
        H::CallNative,
        "mj_call_native",
        // `mj_call_native`'s Rust sig takes `mc`/`st` as two explicit
        // params (not a by-value `Ctx`), so it wants two words on both
        // ABIs — `Pt::P, Pt::P` where the other helpers write `Ctx`.
        &[
            Pt::P,
            Pt::P,
            Pt::P,
            Pt::I32,
            Pt::P,
            Pt::P,
            Pt::P,
            Pt::P,
            Pt::P,
            Pt::P,
        ],
        Hr::U8,
    ),
    (H::Raise, "mj_raise", &[Pt::P, Pt::P, Ctx, Pt::P], Hr::Void),
    (
        H::LoadConstStr,
        "mj_load_const_str",
        &[Pt::P, Pt::P, Pt::I32, Ctx, Pt::P],
        Hr::Void,
    ),
    (
        H::BinStr,
        "mj_bin_str",
        &[Pt::P, Pt::P, Pt::P, Pt::P, Pt::I8],
        Hr::U8,
    ),
    (
        H::CallBody,
        "mj_call_body",
        &[Pt::P, Pt::P, Pt::P, Pt::P, Pt::P, Pt::P],
        Hr::P,
    ),
    (
        H::CallDyn,
        "mj_call_dyn",
        &[Pt::P, Pt::P, Pt::P, Pt::P, Pt::P, Pt::P, Pt::P],
        Hr::P,
    ),
];

/// The spec names this emitter binds — the driver must supply every one.
pub fn spec_names() -> impl Iterator<Item = &'static str> {
    SPECS.iter().map(|s| s.1)
}

fn spec(h: H) -> (&'static str, &'static [Pt], Hr) {
    let s = SPECS.iter().find(|s| s.0 == h).unwrap();
    (s.1, s.2, s.3)
}

// ---- scalar-shadow liveness (jit/src/emit.rs `analyze`, verbatim) ----

#[derive(Default)]
struct Sh {
    int: BTreeSet<u32>,
    float: BTreeSet<u32>,
}

#[derive(Clone, Copy)]
enum W {
    Int,
    Float,
    Copy(u32),
    Dyn,
    DynCheck,
}

fn analyze(ops: &[(usize, Op)], nregs: u32) -> Sh {
    let mut writes: HashMap<u32, Vec<W>> = HashMap::new();
    let mut int_reads: HashSet<u32> = HashSet::new();
    let mut float_reads: HashSet<u32> = HashSet::new();
    let mut put = |r: u32, w: W| {
        if r < nregs {
            writes.entry(r).or_default().push(w);
        }
    };
    let mut iread = |r: u32| {
        if r < nregs {
            int_reads.insert(r);
        }
    };
    let mut fread = |r: u32| {
        if r < nregs {
            float_reads.insert(r);
        }
    };

    for (_, op) in ops {
        match op {
            Op::Move { dst, src } => {
                put(dst.index() as u32, W::Copy(src.index() as u32));
            }
            Op::LoadConst { dst, constant } => match constant {
                Constant::Int(_) => put(dst.index() as u32, W::Int),
                Constant::Float(_) => put(dst.index() as u32, W::Float),
                _ => put(dst.index() as u32, W::Dyn),
            },
            Op::AddInt { dst, left, right }
            | Op::SubInt { dst, left, right }
            | Op::MultInt { dst, left, right }
            | Op::ModInt { dst, left, right } => {
                iread(left.index() as u32);
                iread(right.index() as u32);
                put(dst.index() as u32, W::Int);
            }
            Op::AddIntImm { dst, left, .. }
            | Op::SubIntImm { dst, left, .. }
            | Op::MultIntImm { dst, left, .. }
            | Op::ModIntImm { dst, left, .. } => {
                iread(left.index() as u32);
                put(dst.index() as u32, W::Int);
            }
            Op::IntLt { dst, left, right }
            | Op::IntLe { dst, left, right }
            | Op::IntGt { dst, left, right }
            | Op::IntGe { dst, left, right }
            | Op::IntEq { dst, left, right }
            | Op::IntNe { dst, left, right } => {
                iread(left.index() as u32);
                iread(right.index() as u32);
                put(dst.index() as u32, W::Dyn);
            }
            Op::IntLtImm { dst, left, .. }
            | Op::IntLeImm { dst, left, .. }
            | Op::IntGtImm { dst, left, .. }
            | Op::IntGeImm { dst, left, .. }
            | Op::IntEqImm { dst, left, .. }
            | Op::IntNeImm { dst, left, .. } => {
                iread(left.index() as u32);
                put(dst.index() as u32, W::Dyn);
            }
            Op::AddFloat { dst, left, right }
            | Op::SubFloat { dst, left, right }
            | Op::MultFloat { dst, left, right }
            | Op::DivFloat { dst, left, right } => {
                fread(left.index() as u32);
                fread(right.index() as u32);
                put(dst.index() as u32, W::Float);
            }
            Op::AddFloatImm { dst, left, .. }
            | Op::SubFloatImm { dst, left, .. }
            | Op::MultFloatImm { dst, left, .. }
            | Op::ModFloatImm { dst, left, .. } => {
                fread(left.index() as u32);
                put(dst.index() as u32, W::Float);
            }
            Op::FloatLt { dst, left, right }
            | Op::FloatLe { dst, left, right }
            | Op::FloatGt { dst, left, right }
            | Op::FloatGe { dst, left, right }
            | Op::FloatEq { dst, left, right }
            | Op::FloatNe { dst, left, right } => {
                fread(left.index() as u32);
                fread(right.index() as u32);
                put(dst.index() as u32, W::Dyn);
            }
            Op::FloatLtImm { dst, left, .. }
            | Op::FloatLeImm { dst, left, .. }
            | Op::FloatGtImm { dst, left, .. }
            | Op::FloatGeImm { dst, left, .. }
            | Op::FloatEqImm { dst, left, .. }
            | Op::FloatNeImm { dst, left, .. } => {
                fread(left.index() as u32);
                put(dst.index() as u32, W::Dyn);
            }
            Op::Len { dst, .. } => put(dst.index() as u32, W::Int),
            Op::ToFloat { dst, src } => {
                iread(src.index() as u32);
                put(dst.index() as u32, W::Float);
            }
            Op::Sqrt { dst, src } => {
                fread(src.index() as u32);
                put(dst.index() as u32, W::Float);
            }
            Op::ForNext { idx, bound, .. } => {
                iread(idx.index() as u32);
                iread(bound.index() as u32);
                put(idx.index() as u32, W::Int);
            }
            Op::BIntLt { left, right, .. }
            | Op::BIntLe { left, right, .. }
            | Op::BIntGt { left, right, .. }
            | Op::BIntGe { left, right, .. }
            | Op::BIntEq { left, right, .. }
            | Op::BIntNe { left, right, .. } => {
                iread(left.index() as u32);
                iread(right.index() as u32);
            }
            Op::BIntLtImm { left, .. }
            | Op::BIntLeImm { left, .. }
            | Op::BIntGtImm { left, .. }
            | Op::BIntGeImm { left, .. }
            | Op::BIntEqImm { left, .. }
            | Op::BIntNeImm { left, .. } => iread(left.index() as u32),
            Op::BFloatLt { left, right, .. }
            | Op::BFloatLe { left, right, .. }
            | Op::BFloatGt { left, right, .. }
            | Op::BFloatGe { left, right, .. }
            | Op::BFloatEq { left, right, .. }
            | Op::BFloatNe { left, right, .. } => {
                fread(left.index() as u32);
                fread(right.index() as u32);
            }
            Op::BFloatLtImm { left, .. }
            | Op::BFloatLeImm { left, .. }
            | Op::BFloatGtImm { left, .. }
            | Op::BFloatGeImm { left, .. }
            | Op::BFloatEqImm { left, .. }
            | Op::BFloatNeImm { left, .. } => fread(left.index() as u32),
            // container reads produce a dynamically-typed value, but the
            // helper-op read path refreshes the scalar shadow (`DynCheck`)
            Op::GetField { dst, .. } | Op::GetIndex { dst, .. } => {
                put(dst.index() as u32, W::DynCheck);
            }
            // every other op that carries a dst writes a dynamically-typed value
            Op::LoadBody { dst, .. }
            | Op::LoadEntry { dst, .. }
            | Op::BoolEq { dst, .. }
            | Op::BoolNe { dst, .. }
            | Op::StrEq { dst, .. }
            | Op::StrNe { dst, .. }
            | Op::Bin { dst, .. }
            | Op::Unary { dst, .. }
            | Op::In { dst, .. }
            | Op::IsInstance { dst, .. }
            | Op::IsRaised { dst, .. }
            | Op::UnwrapRaised { dst, .. }
            | Op::Unwrap { dst, .. }
            | Op::UnwrapUnit { dst, .. }
            | Op::NewArray { dst }
            | Op::NewDict { dst }
            | Op::NewInstance { dst, .. }
            | Op::NewClosure { dst, .. }
            | Op::Format { dst, .. }
            | Op::Call { dst, .. }
            | Op::CallDirect { dst, .. }
            | Op::CallNative { dst, .. } => put(dst.index() as u32, W::Dyn),
            Op::Jump { .. }
            | Op::JumpIf { .. }
            | Op::Switch { .. }
            | Op::Push { .. }
            | Op::Insert { .. }
            | Op::SetIndex { .. }
            | Op::SetField { .. }
            | Op::StoreEntry { .. }
            | Op::Return { .. }
            | Op::Panic {}
            | Op::TraceEdge { .. }
            | Op::Shadow { .. }
            | Op::Raise { .. } => {}
        }
    }

    let mut int: BTreeSet<u32> = BTreeSet::new();
    let mut float: BTreeSet<u32> = BTreeSet::new();
    for r in 0..nregs {
        if writes.contains_key(&r) {
            continue;
        }
        match (int_reads.contains(&r), float_reads.contains(&r)) {
            (true, false) => {
                int.insert(r);
            }
            (false, true) => {
                float.insert(r);
            }
            _ => {}
        }
    }
    loop {
        let mut changed = false;
        for (&r, ws) in &writes {
            // a DynCheck write only carries a shadow if `r` is genuinely read
            // as that scalar — otherwise `(sv, ok)` would be pure overhead
            let has_chk = ws.iter().any(|w| matches!(w, W::DynCheck));
            let ok_i = ws.iter().all(|w| match w {
                W::Int | W::DynCheck => true,
                W::Copy(src) => int.contains(src),
                _ => false,
            }) && (!has_chk || int_reads.contains(&r));
            let ok_f = ws.iter().all(|w| match w {
                W::Float | W::DynCheck => true,
                W::Copy(src) => float.contains(src),
                _ => false,
            }) && (!has_chk || float_reads.contains(&r));
            if ok_i != int.contains(&r) {
                if ok_i {
                    int.insert(r);
                } else {
                    int.remove(&r);
                }
                changed = true;
            }
            if ok_f != float.contains(&r) {
                if ok_f {
                    float.insert(r);
                } else {
                    float.remove(&r);
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Sh { int, float }
}

/// bcgen's `may_pause`, verbatim (see `jit/src/emit.rs`).
fn may_pause(op: &Op) -> bool {
    matches!(
        op,
        Op::Bin { .. }
            | Op::Unary { .. }
            | Op::Call { .. }
            | Op::CallDirect { .. }
            | Op::CallNative { .. }
    )
}

// ---- the emitter ----

/// Env-block hoisted values (entry-block SSA results).
#[derive(Default)]
struct Env {
    env: String,
    thread: String,
    code: String,
    ctx0: String,
    ctx1: String,
    /// `&env.mutation` — the two-word `Ctx` pair's address, passed as the
    /// single indirect `Ctx` arg on wasm.
    ctx_p: String,
    strs: String,
    fuel_p: String,
    opip_p: String,
    out: String,
    paused_p: String,
    opsleft_p: String,
    ip_p: String,
    base: String,
    nregs: String,
    bodies_tbl: String,
}

/// Shared exit blocks.
struct Ex {
    dispatch: String,
    edef: String,
    enext: String,
    eoof: String,
    estep: String,
    estep_g: String,
    eerr: String,
    eret: String,
    eend: String,
}

/// One arg value handed to [`Em::hcall`].
enum A {
    /// An `i64` word value — `inttoptr`'d for a `P` param.
    V(String),
    /// A `llvm.ptr (0)` value — used directly.
    P(String),
    /// An `i8`/`i32`/`f64` value used directly.
    I8(String),
    I32(String),
    #[allow(dead_code)]
    F64(String),
}

struct Em<'a> {
    ops: &'a [(usize, Op)],
    prog: &'a Program,
    lyt: &'a Layout,
    sh: &'a Sh,
    /// wasm32 ABI: `Ctx` args pass indirectly (see [`sig`]).
    wasm: bool,

    /// Completed blocks, in creation order: `(label, lines)`.
    done: Vec<(String, Vec<String>)>,
    cur_name: String,
    cur: Vec<String>,
    /// Lines that land at the top of the entry block (allocas + their
    /// constant size operands, reg-list slots).
    prelude: Vec<String>,
    nv: u32,
    nb: u32,

    env: Env,
    ex: Ex,
    blocks: Vec<String>,
    off2idx: HashMap<usize, usize>,

    // alloca names for mutable state
    v_regs: String,
    v_curip: String,
    v_ekind: String,
    v_bcn: String,
    v_bcn0: String,
    int: BTreeMap<u32, (String, String)>,
    float: BTreeMap<u32, (String, String)>,
    /// Scratch i64 cell for `Len`'s out-param.
    slot_i: String,
    /// `reg_list` slots, deduplicated by index list: alloca `[i32 x n]`.
    rl: HashMap<Vec<u32>, String>,

    /// This chunk's index and code offset — the two program-assigned
    /// numbers the emitted text would otherwise bake in. Everything
    /// index-derived goes through `goff`/`chunk_off`/`body_idx`, which
    /// read the `@mimas_body_off_*` / `@mimas_body_idx_*` globals the
    /// init module defines, so the body itself stays content-only.
    self_i: usize,
    chunk_off: usize,
    /// Memoized `*@mimas_body_off_{self_i}` for `goff`.
    v_offbase: String,
}

/// `(ret_ty, ptys)` for a helper signature — the shared spelling used by
/// the extern declarations and by [`Em::hcall`]. On wasm `Ctx` lowers
/// indirectly (one pointer to the two-word pair) instead of the two
/// machine words the native ABI splits it into.
fn sig(params: &[Pt], ret: Hr, wasm: bool) -> (String, String) {
    let ret_ty = match ret {
        Hr::Void => VOID,
        Hr::U8 => I8,
        Hr::P => PTR,
    };
    let mut ptys = String::new();
    for p in params.iter() {
        let n = match p {
            Pt::Ctx if !wasm => 2,
            _ => 1,
        };
        for _ in 0..n {
            if !ptys.is_empty() {
                ptys.push_str(", ");
            }
            ptys.push_str(match p {
                Pt::P | Pt::Ctx => PTR,
                Pt::I8 => I8,
                Pt::I32 => I32,
                Pt::F64 => F64,
            });
        }
    }
    (ret_ty.to_string(), ptys)
}

/// Per-body `.pliron` texts plus their content hashes, one entry per
/// chunk. The returned texts are content-addressed: every
/// `mimas_body_{index}` name — the body's own symbol and its direct
/// call targets — is renamed to [`body_sym`] of the callee's hash, and
/// every program-assigned number (code offsets, table indices) reads an
/// `@mimas_body_off_{i}` / `@mimas_body_idx_{i}` global the init module
/// defines. Identical chunks emit identical text, wherever
/// `compile_parts` placed them — the raw material for `emit_module`,
/// `assemble_module_ex`, and content-keyed object caches.
/// Level-2 hash input: `mimas_body_{i}` self-refs, the `off`/`idx`
/// globals and every non-name token collapse to `#`; `mimas_body_{j}`
/// callee refs resolve to the callee's level-1 content name — what the
/// body calls *content-wise*, never where `compile_parts` placed it.
fn level2_key(text: &str, self_i: usize, lvl1: &[u64]) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(p) = rest.find("mimas_body_") {
        out.push_str(&rest[..p]);
        let tail = &rest[p + "mimas_body_".len()..];
        let end = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(tail.len());
        let tok = &tail[..end];
        rest = &tail[end..];
        let idx = tok
            .strip_prefix("off_")
            .or_else(|| tok.strip_prefix("idx_"))
            .map(|_| None)
            .unwrap_or_else(|| tok.parse::<usize>().ok().map(Some));
        match idx {
            Some(None) | None => out.push('#'),
            Some(Some(j)) if j == self_i => out.push('#'),
            Some(Some(j)) => out.push_str(&body_sym(lvl1[j])),
        }
    }
    out.push_str(rest);
    out
}

/// Two-level content hash, Unison-style. `h'` hashes the text with all
/// `mimas_body_*` names normalized — chunk content modulo call
/// identities. `h` then hashes the text with every `mimas_body_{j}`
/// callee ref resolved to `mimas_body_{h'_j}`: a body hashes by what it
/// calls *content-wise*, not by where `compile_parts` placed the
/// callee. Isomorphic cycles get equal `h`s; a level collision shows up
/// as a renamed-text mismatch and panics, never a silent alias.
pub fn emit_bodies(prog: &Program, lyt: &Layout, wasm: bool) -> (Vec<u64>, Vec<String>) {
    let texts: Vec<String> = (0..prog.chunks.len())
        .map(|b| emit_body(prog, b, lyt, wasm))
        .collect();
    let lvl1: Vec<u64> = texts.iter().map(|t| body_hash(t)).collect();
    // Level-2 keys carry callee identity in `mimas_body_x{lvl1}` names —
    // hash them raw, not through `body_hash`'s name normalizer.
    let hashes: Vec<u64> = {
        use rustc_hash::FxHasher;
        use std::hash::{Hash, Hasher};
        texts
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut h = FxHasher::default();
                level2_key(t, i, &lvl1).hash(&mut h);
                h.finish()
            })
            .collect()
    };
    let renamed: Vec<String> = texts
        .iter()
        .map(|t| rename_body_names(t, &hashes))
        .collect();
    // same hash + different renamed text is a real aliasing bug, not a
    // content-identical dup — panic rather than merge under one symbol.
    let mut seen: HashMap<u64, usize> = HashMap::new();
    for (i, h) in hashes.iter().enumerate() {
        if let Some(&j) = seen.get(h) {
            if renamed[i] != renamed[j] {
                let a = &renamed[j];
                let b = &renamed[i];
                let p = a
                    .bytes()
                    .zip(b.bytes())
                    .position(|(x, y)| x != y)
                    .unwrap_or(a.len().min(b.len()));
                eprintln!("COLLIDE {j} vs {i} @{p}:\n...{}...\n...{}...",
                    &a[p.saturating_sub(80)..(p + 160).min(a.len())],
                    &b[p.saturating_sub(80)..(p + 160).min(b.len())]);
            }
            assert!(
                renamed[i] == renamed[j],
                "body_hash collision between chunks {j} and {i}"
            );
        } else {
            seen.insert(*h, i);
        }
    }
    (hashes, renamed)
}

/// Content hash for every chunk — re-emits the bodies. For hosts that
/// only need the symbol names (`dlsym` fill on the link path).
pub fn body_hashes(prog: &Program, lyt: &Layout, wasm: bool) -> Vec<u64> {
    emit_bodies(prog, lyt, wasm).0
}

/// Per-chunk `(offset, end)` byte spans — all a module assembly reads
/// from the `Program`. Plain data (`Send + Sync`), so hosts that
/// assemble modules on worker threads fetch it once and pass the slice.
pub fn chunk_spans(prog: &Program) -> Vec<(usize, usize)> {
    let offs: Vec<usize> = (0..prog.chunks.len())
        .map(|i| prog.chunks[compile::BodyId::from(i as u32)].offset)
        .collect();
    offs.iter()
        .map(|&o| {
            let end = offs
                .iter()
                .copied()
                .filter(|&x| x > o)
                .min()
                .unwrap_or(prog.bytes.len());
            (o, end)
        })
        .collect()
}

/// Emit `.pliron` text for every chunk of `prog`, the bodiless `mj_*`
/// extern declarations, and the
/// `@mimas_pliron_bodies`/`@mimas_pliron_nbodies` globals.
///
/// Helper references are real `llvm.call @mj_*` symbols: link the object
/// against a runtime that exports them (`mimas-natrt`'s `staticlib`/
/// `cdylib`, or any artifact providing `vm::bc::nat`). Returns the body
/// content hashes alongside — the host resolves `mimas_body_{h}`.
pub fn emit_module(prog: &Program, lyt: &Layout, wasm: bool, nsig: u64) -> (Vec<u64>, String) {
    let (hashes, bodies) = emit_bodies(prog, lyt, wasm);
    let bodies: Vec<(usize, String)> = bodies
        .into_iter()
        .enumerate()
        .collect();
    let text = assemble_module(&chunk_spans(prog), lyt, wasm, &bodies, &hashes, nsig);
    (hashes, text)
}

/// Assemble one module from the shared members plus `bodies` as
/// `(chunk index, emitted text)` pairs. `nbodies` is the whole program's
/// chunk count: chunk indexes not listed get a bodiless decl so
/// cross-body `llvm.call @mimas_body_*`s still resolve — a subset module
/// scans and lowers standalone (the point of the split), and its object
/// carries only real undefined-symbol references for the missing bodies.
/// The table stays sized for the whole program and `init` fills only the
/// listed slots, at their true indexes — so a merged link of the parts
/// stays consistent.
pub fn assemble_module(
    spans: &[(usize, usize)],
    lyt: &Layout,
    wasm: bool,
    bodies: &[(usize, String)],
    hashes: &[u64],
    nsig: u64,
) -> String {
    assemble_module_ex(spans, lyt, wasm, bodies, true, hashes, None, nsig)
}

/// `assemble_module` with the program-tail split out. `tail = false`
/// emits the bodies-table and count as bodiless `llvm.global` decls and
/// drops `mimas_pliron_init`: the resulting object carries only the
/// listed bodies' code plus undefined-symbol refs, so it depends on the
/// body text and lowering config alone — not on `nbodies` — and a
/// content-keyed cache can share kit chunks across programs that differ
/// only in chunk count. The pair to it is `assemble_init_module`, which
/// owns the real definitions and the initializer.
pub fn assemble_module_ex(
    spans: &[(usize, usize)],
    lyt: &Layout,
    wasm: bool,
    bodies: &[(usize, String)],
    tail: bool,
    hashes: &[u64],
    init_fill: Option<&[usize]>,
    nsig: u64,
) -> String {
    let nbodies = spans.len();
    let mut present = vec![false; nbodies];
    for (i, _) in bodies {
        present[*i] = true;
    }
    let mut members: Vec<String> = Vec::new();

    // extern declarations for the helper surface — bodiless `llvm.func`s
    // (Import linkage), resolved by the companion runtime at link time.
    // Each declares `llvm_func_attrs` so the contract facts a bodiless
    // decl can't be IPO-inferred for travel in the text: `willreturn`
    // (every helper returns or aborts on a violated contract) and
    // `nounwind` (a Rust panic hits the extern "C" abort shim, it never
    // unwinds into the body). `nocapture` is a per-parameter fact and
    // the dialect has no declared slot for it — the consumer seeds it.
    // Each decl gets its own `!N` outline index; refs are unique per op.
    let mut outlines: Vec<String> = Vec::new();
    for (i, (_, name, params, ret)) in SPECS.iter().enumerate() {
        let (ret_ty, ptys) = sig(params, *ret, wasm);
        members.push(format!(
            "llvm.func @{name}: llvm.func <{ret_ty} ({ptys}) variadic = false> [] !{i}"
        ));
        outlines.push(format!(
            "!{i} = [llvm_func_attrs = llvm.attributes [\"nounwind\", \"willreturn\"]]"
        ));
    }

    // function-pointer table the body's `call_body`/`call_dyn` `tbl`
    // argument indexes. The text path can't carry symbol initializers
    // (scan zero-fills every defined global), so the table is emitted
    // mutable+zero and the host fills it after dlopen from the
    // `mimas_body_*` symbols — which is also how it learns the table for
    // `install_bc`. Entries are word-sized ints (ptr-sized and
    // scalar-typed, so the aggregate survives the parser): the runtime
    // reads the table as `*const usize`, so elements must step at
    // `word_size` — i64 natively, i32 on wasm32.
    let wt = if lyt.word_size == 4 { I32 } else { I64 };
    let w = lyt.word_size;
    let zeros = vec![format!("builtin.integer <0: i{w}>"); nbodies].join(", ");
    let zw = format!("llvm.array [{nbodies} x {wt}]");
    if tail {
        members.push(format!(
            "llvm.global @{BODIES_GLOBAL} : {zw}\n    [llvm_global_linkage: llvm.linkage ExternalLinkage, llvm_global_constant: builtin.bool false] = llvm.aggregate <[{zeros}] : {zw}>"
        ));
    } else {
        // bodiless decl: the body only takes the table's address, so a
        // `ptr`-typed global suffices and keeps the module — and its
        // object — free of the chunk count.
        members.push(format!(
            "llvm.global @{BODIES_GLOBAL} : {PTR}\n    [llvm_global_linkage: llvm.linkage ExternalLinkage, llvm_global_constant: builtin.bool false]"
        ));
    }
    // Per-chunk index globals, hash-named: the one place
    // program-assigned numbers live — `off_{h}` is the canonical
    // index's code offset, `idx_{h}` its body-table index. Bodies read
    // them through decls, so body text and objects stay content-only;
    // only the init module defines them. Mutable plus an init store for
    // wasm, where data-section initializers do not survive lowering.
    // Content-identical chunks alias: every index i with hash h reads
    // the same global and gets the canonical (lowest) index's values —
    // safe because the code is literally identical, and `tbl[i]` still
    // points at the shared body.
    let mut canon: BTreeMap<u64, usize> = BTreeMap::new();
    for i in 0..nbodies {
        canon.entry(hashes[i]).or_insert(i);
    }
    for (h, i) in &canon {
        let off = spans[*i].0;
        let hn = format!("{h:016x}");
        if tail {
            members.push(format!(
                "llvm.global @mimas_body_off_x{hn} : {I64}\n    [llvm_global_linkage: llvm.linkage ExternalLinkage, llvm_global_constant: builtin.bool false] = builtin.integer <{off}: i64>"
            ));
            members.push(format!(
                "llvm.global @mimas_body_idx_x{hn} : {I64}\n    [llvm_global_linkage: llvm.linkage ExternalLinkage, llvm_global_constant: builtin.bool false] = builtin.integer <{i}: i64>"
            ));
        } else {
            members.push(format!(
                "llvm.global @mimas_body_off_x{hn} : {I64}\n    [llvm_global_linkage: llvm.linkage ExternalLinkage, llvm_global_constant: builtin.bool false]"
            ));
            members.push(format!(
                "llvm.global @mimas_body_idx_x{hn} : {I64}\n    [llvm_global_linkage: llvm.linkage ExternalLinkage, llvm_global_constant: builtin.bool false]"
            ));
        }
    }

    if tail {
        // the pair's scalar half — a standalone host reads this instead of
        // re-deriving the count from the Program. Mutable+zero like the
        // table: the wasm object's data section can't carry the constant
        // initializer, so `mimas_pliron_init` stores the count alongside
        // the table fill (the shim reads it after calling init).
        members.push(format!(
            "llvm.global @{NBODIES_GLOBAL} : {I64}\n    [llvm_global_linkage: llvm.linkage ExternalLinkage, llvm_global_constant: builtin.bool false] = builtin.integer <0: i64>"
        ));

        // `mimas_pliron_init` fills the table from code: `llvm.addressof`
        // yields the callee's address natively and its `GOT.func` table
        // index on wasm (where symbol-initializer relocs can't come from the
        // text path and there is no host `dlsym` to fill the table). The
        // wasm shim calls it once before `install_bc`.
        let mut init = format!(
            "llvm.func @{INIT_FN}: llvm.func <{VOID} () variadic = false>\n    [llvm_function_linkage: llvm.linkage ExternalLinkage]\n  {{\n^{INIT_ENTRY}():\n    a_tbl = llvm.addressof @{BODIES_GLOBAL} : {PTR};\n    a_nb = llvm.addressof @{NBODIES_GLOBAL} : {PTR};\n    v_nb = llvm.constant <builtin.integer <{nbodies}: i64>> : {I64};\n    llvm.store *a_nb <- v_nb [align : 8];\n"
        );
        // `init_fill` defaults to the listed bodies; the init module
        // passes all chunks, so wasm's table and index globals end up
        // complete without a dlsym pass.
        let fill: Vec<usize> = match init_fill {
            Some(f) => f.to_vec(),
            None => bodies.iter().map(|(i, _)| *i).collect(),
        };
        for &i in &fill {
            init.push_str(&format!(
                "    a_f{i} = llvm.addressof @mimas_body_{i} : {PTR};\n    v_f{i} = llvm.ptrtoint a_f{i} to {wt};\n    a_p{i} = llvm.gep <{I8}> (a_tbl)[Constant({ioff})] : {PTR};\n    llvm.store *a_p{i} <- v_f{i} [align : {w}];\n",
                ioff = i * w
            ));
            // off/idx globals: one store pair per unique hash, keyed on
            // the canonical index's values.
            if canon[&hashes[i]] == i {
                let off = spans[i].0;
                let hn = format!("{:016x}", hashes[i]);
                init.push_str(&format!(
                    "    a_o{i} = llvm.addressof @mimas_body_off_x{hn} : {PTR};\n    v_o{i} = llvm.constant <builtin.integer <{off}: i64>> : {I64};\n    llvm.store *a_o{i} <- v_o{i} [align : 8];\n    a_x{i} = llvm.addressof @mimas_body_idx_x{hn} : {PTR};\n    v_x{i} = llvm.constant <builtin.integer <{i}: i64>> : {I64};\n    llvm.store *a_x{i} <- v_x{i} [align : 8];\n"
                ));
            }
        }
        init.push_str("    llvm.return\n  }");
        members.push(init);

        // The native-table fingerprint of the compile-time install —
        // `NativeId`s are registration-order, so a runtime built against
        // a different install must refuse rather than misdispatch.
        members.push(format!(
            "llvm.func @{NSIG_FN}: llvm.func <{I64} () variadic = false>\n    [llvm_function_linkage: llvm.linkage ExternalLinkage]\n  {{\n^nsig_entry():\n    v_sig = llvm.constant <builtin.integer <{nsig}: i64>> : {I64};\n    llvm.return v_sig\n  }}",
            nsig = nsig as i64
        ));

        // Per-index entry trampolines: `tbl[i]` and `bc[frame.chunk]`
        // dispatch to `mimas_body_{i}`, which may be an alias of a
        // canonical `mimas_body_x{h}` whose `off`/`idx` globals hold the
        // canonical chunk's numbers. An interp-pushed frame enters with
        // `code.ip` in the *real* chunk's byte range, so the trampoline
        // shifts it into the canonical range when — and only when — it
        // points there. Reentries see an already-canonical `ip` (bodies
        // write canonical offsets) and pass through. Ranges are
        // disjoint across chunks, so the bounds check can't misfire.
        let wty = if lyt.word_size == 4 { I32 } else { I64 };
        for &i in &fill {
            let (off_i, end_i) = spans[i];
            let delta = spans[canon[&hashes[i]]].0 as i64 - off_i as i64;
            let cw = std::mem::offset_of!(BodyEnv<'_>, code)
                / std::mem::size_of::<usize>()
                * lyt.word_size;
            let adjust = if wasm {
                format!(
                    "    a_cp{i} = llvm.gep <{I8}> (a_env)[Constant({cw})] : {PTR};\n    v_c{i} = llvm.load a_cp{i} [align : 8] : {PTR};\n    a_ip{i} = llvm.gep <{I8}> (v_c{i})[Constant({cip})] : {PTR};\n    v_iw{i} = llvm.load a_ip{i} [align : 4] : {wty};\n    v_ip{i} = llvm.zext <nneg=false> v_iw{i} to {I64};\n    v_lo{i} = llvm.constant <builtin.integer <{off_i}: i64>> : {I64};\n    c_lo{i} = llvm.icmp v_ip{i} <UGE> v_lo{i} : {I1};\n    v_hi{i} = llvm.constant <builtin.integer <{end_i}: i64>> : {I64};\n    c_hi{i} = llvm.icmp v_ip{i} <ULT> v_hi{i} : {I1};\n    c_in{i} = llvm.and c_lo{i}, c_hi{i} : {I1};\n    v_d{i} = llvm.constant <builtin.integer <{delta}: i64>> : {I64};\n    v_adj{i} = llvm.add v_ip{i}, v_d{i} <{{nsw=false,nuw=false}}> : {I64};\n    v_n{i} = llvm.select c_in{i} ? v_adj{i} : v_ip{i} : {I64};\n    v_nw{i} = llvm.trunc v_n{i} to {wty};\n    llvm.store *a_ip{i} <- v_nw{i} [align : 4];\n",
                    cw = cw,
                    cip = lyt.code_ip,
                )
            } else {
                format!(
                    "    a_cp{i} = llvm.gep <{I8}> (a_env)[Constant({cw})] : {PTR};\n    v_c{i} = llvm.load a_cp{i} [align : 8] : {PTR};\n    a_ip{i} = llvm.gep <{I8}> (v_c{i})[Constant({cip})] : {PTR};\n    v_ip{i} = llvm.load a_ip{i} [align : 8] : {I64};\n    v_lo{i} = llvm.constant <builtin.integer <{off_i}: i64>> : {I64};\n    c_lo{i} = llvm.icmp v_ip{i} <UGE> v_lo{i} : {I1};\n    v_hi{i} = llvm.constant <builtin.integer <{end_i}: i64>> : {I64};\n    c_hi{i} = llvm.icmp v_ip{i} <ULT> v_hi{i} : {I1};\n    c_in{i} = llvm.and c_lo{i}, c_hi{i} : {I1};\n    v_d{i} = llvm.constant <builtin.integer <{delta}: i64>> : {I64};\n    v_adj{i} = llvm.add v_ip{i}, v_d{i} <{{nsw=false,nuw=false}}> : {I64};\n    v_n{i} = llvm.select c_in{i} ? v_adj{i} : v_ip{i} : {I64};\n    llvm.store *a_ip{i} <- v_n{i} [align : 8];\n",
                    cw = cw,
                    cip = lyt.code_ip,
                )
            };
            members.push(format!(
                "llvm.func @mimas_body_{i}: llvm.func <{VOID} ({PTR}) variadic = false>\n    [llvm_function_linkage: llvm.linkage ExternalLinkage]\n  {{\n^entry{i}(a_env: {PTR}):\n{adjust}    r{i} = llvm.call @{} (a_env) : llvm.func <{VOID} ({PTR}) variadic = false>;\n    llvm.return\n  }}",
                body_sym(hashes[i])
            ));
        }
    }

    // Content-identical chunks share a hash — one definition wins.
    // Present bodies mark `seen` first: an absent chunk that happens to
    // be content-identical to a present body must NOT shadow it with a
    // bodiless decl.
    let mut decl_seen: HashSet<u64> = HashSet::new();
    for (i, text) in bodies {
        if decl_seen.insert(hashes[*i]) {
            members.push(text.clone());
        }
    }
    // Bodies outside this subset: bodiless decls so the present bodies'
    // direct `llvm.call @mimas_body_*`s resolve. They land as ordinary
    // undefined symbols in the object.
    for i in 0..nbodies {
        if !present[i] && decl_seen.insert(hashes[i]) {
            members.push(format!(
                "llvm.func @{}: llvm.func <{VOID} ({PTR}) variadic = false> []",
                body_sym(hashes[i])
            ));
        }
    }

    format!(
        "builtin.module @mimas_pliron {{\n^block0():\n  {}\n}}\n\noutlined_attributes:\n{}\n",
        members.join(";\n  "),
        outlines.join("\n")
    )
}

/// The program-tail module: helper decls, the real bodies-table,
/// count and per-chunk index globals, `mimas_pliron_init`, and bodiless
/// decls for every `mimas_body_{h}`. Pairs with
/// `assemble_module_ex(.., tail = false)` chunk modules — together they
/// cover one whole program.
pub fn assemble_init_module(
    spans: &[(usize, usize)],
    lyt: &Layout,
    wasm: bool,
    hashes: &[u64],
    nsig: u64,
) -> String {
    let all: Vec<usize> = (0..spans.len()).collect();
    assemble_module_ex(spans, lyt, wasm, &[], true, hashes, Some(&all), nsig)
}

/// `.pliron` text for one chunk — `emit_bodies`'s per-index form, for
/// callers that parallelize emission themselves.
pub fn emit_body(prog: &Program, body: usize, lyt: &Layout, wasm: bool) -> String {
    let body_id = compile::BodyId::from(body as u32);
    let ops = prog.ops(body_id);
    let chunk = &prog.chunks[body_id];
    let sh = analyze(&ops, chunk.regs as u32);

    let mut em = Em {
        ops: &ops,
        prog,
        lyt,
        sh: &sh,
        wasm,
        done: Vec::new(),
        cur_name: String::new(),
        cur: Vec::new(),
        prelude: Vec::new(),
        nv: 0,
        nb: 0,
        env: Env::default(),
        ex: Ex {
            dispatch: "dispatch".into(),
            edef: "edef".into(),
            enext: "enext".into(),
            eoof: "eoof".into(),
            estep: "estep".into(),
            estep_g: "estep_g".into(),
            eerr: "eerr".into(),
            eret: "eret".into(),
            eend: "eend".into(),
        },
        blocks: ops
            .iter()
            .enumerate()
            .map(|(i, _)| format!("op{i}"))
            .collect(),
        off2idx: ops.iter().enumerate().map(|(i, (o, _))| (*o, i)).collect(),
        v_regs: String::new(),
        v_curip: String::new(),
        v_ekind: String::new(),
        v_bcn: String::new(),
        v_bcn0: String::new(),
        int: BTreeMap::new(),
        float: BTreeMap::new(),
        slot_i: String::new(),
        rl: HashMap::new(),
        self_i: body,
        chunk_off: chunk.offset,
        v_offbase: String::new(),
    };

    // the alloca size operand (a single shared `i32 1` const)
    em.prelude.push(format!(
        "a_one = llvm.constant <builtin.integer <1: i32>> : {I32}"
    ));

    // ---- allocas (entry prelude) ----
    em.v_regs = em.alloca(PTR, "regs");
    em.v_curip = em.alloca(I64, "curip");
    em.v_ekind = em.alloca(I8, "ekind");
    em.v_bcn = em.alloca(I64, "bcn");
    em.v_bcn0 = em.alloca(I64, "bcn0");
    for &r in &sh.int {
        let sv = em.alloca(I64, &format!("isv{r}"));
        let ok = em.alloca(I8, &format!("iok{r}"));
        em.int.insert(r, (sv, ok));
    }
    for &r in &sh.float {
        let sv = em.alloca(F64, &format!("fsv{r}"));
        let ok = em.alloca(I8, &format!("fok{r}"));
        em.float.insert(r, (sv, ok));
    }
    em.slot_i = em.alloca(I64, "slot_i");

    // ---- entry ----
    em.start_block("entry");
    let env_p = "env".to_string();
    // `BodyEnv` is `#[repr(C)]` with pointer fields — a byte-offset load each.
    let mut env = Env {
        env: env_p.clone(),
        ..Env::default()
    };
    // `BodyEnv` is `repr(C)` with all-pointer fields, so a field offset
    // is `index * word_size`: deriving the index from the host
    // `offset_of!` (a compile-checked field reference) and rescaling by
    // the probed word size keeps this correct on wasm32, where the same
    // struct packs at a 4-byte stride.
    let envoff = |off: usize| off / std::mem::size_of::<usize>() * lyt.word_size;
    env.thread = em.envld(&env_p, envoff(std::mem::offset_of!(BodyEnv<'_>, thread)));
    env.code = em.envld(&env_p, envoff(std::mem::offset_of!(BodyEnv<'_>, code)));
    env.ctx0 = em.envld(&env_p, envoff(std::mem::offset_of!(BodyEnv<'_>, mutation)));
    env.ctx1 = em.envld(&env_p, envoff(std::mem::offset_of!(BodyEnv<'_>, state)));
    if wasm {
        // `mutation`/`state` are adjacent `repr(C)` fields — a gep to
        // `mutation` is the `{mutation, state}` pair's address, the
        // single indirect `Ctx` arg the wasm ABI wants.
        let c = em.iconst(envoff(std::mem::offset_of!(BodyEnv<'_>, mutation)) as i64);
        env.ctx_p = em.gep_dyn(&env_p, &c);
    }
    env.strs = em.envld(&env_p, envoff(std::mem::offset_of!(BodyEnv<'_>, strs)));
    env.fuel_p = em.envld(&env_p, envoff(std::mem::offset_of!(BodyEnv<'_>, fuel)));
    env.opip_p = em.envld(&env_p, envoff(std::mem::offset_of!(BodyEnv<'_>, op_ip)));
    env.out = em.envld(&env_p, envoff(std::mem::offset_of!(BodyEnv<'_>, out)));
    {
        let c = em.iconst(lyt.state_paused as i64);
        env.paused_p = em.gep_dyn(&env.ctx1, &c);
        let c = em.iconst(lyt.ops_left_off as i64);
        env.opsleft_p = em.gep_dyn(&env.thread, &c);
        let c = em.iconst(lyt.code_ip as i64);
        env.ip_p = em.gep_dyn(&env.code, &c);
    }
    // `thread.frames.last().unwrap().base`
    let fptr = {
        let c = em.iconst((lyt.frames_off + lyt.vec_ptr) as i64);
        let a = em.gep_dyn(&env.thread, &c);
        em.ld(PTR, &a, 0, 8)
    };
    let flen = {
        let c = em.iconst((lyt.frames_off + lyt.vec_len) as i64);
        let a = em.gep_dyn(&env.thread, &c);
        em.ld_word(&a, 0)
    };
    let one = em.iconst(1);
    let fm1 = em.sub(&flen, &one);
    let fsz = em.iconst(lyt.frame_size as i64);
    let foff = em.mul(&fm1, &fsz);
    let faddr = em.gep_dyn(&fptr, &foff);
    env.base = {
        let c = em.iconst(lyt.frame_base as i64);
        let a = em.gep_dyn(&faddr, &c);
        em.ld_word(&a, 0)
    };
    // `regs = thread.regs.as_mut_ptr() + base * val_size`
    let rp = {
        let c = em.iconst((lyt.regs_off + lyt.vec_ptr) as i64);
        let a = em.gep_dyn(&env.thread, &c);
        em.ld(PTR, &a, 0, 8)
    };
    let vs = em.iconst(lyt.val_size as i64);
    let boff = em.mul(&env.base, &vs);
    let regs0 = em.gep_dyn(&rp, &boff);
    env.nregs = em.iconst(chunk.regs as i64);
    env.bodies_tbl = {
        let v = em.v();
        em.ln(format!("{v} = llvm.addressof @{BODIES_GLOBAL} : {PTR}"));
        v
    };
    em.env = env;
    let a = em.v_regs.clone();
    em.st(&a, &regs0);
    let z64 = em.iconst(0);
    let a = em.v_curip.clone();
    em.st(&a, &z64);
    let z8 = em.iconst8(0);
    let a = em.v_ekind.clone();
    em.st(&a, &z8);
    // arm the batched quota: `bcn = min(*fuel, ops_left)` (bcgen's entry arm).
    // The driver already checked paused/fuel/ops_left before invoking the
    // body, so bcn >= 1 here.
    em.rearm_seq();
    // shadow init: `(v, ok) = match regs[r] { Int(v) => (v, 1), _ => (0, 0) }`
    let int_regs: Vec<u32> = {
        let mut v: Vec<u32> = em.sh.int.iter().copied().collect();
        v.sort_unstable();
        v
    };
    for r in int_regs {
        let a = em.vaddr(&regs0, r);
        let t = em.ld_tag(&a);
        let want = em.tconst(em.lyt.t_int);
        let ty = em.tag_ty();
        let k = em.icmp(ty, "EQ", &t, &want);
        let pv = em.ld(I64, &a, em.lyt.val_pay as u32, 8);
        let sv0 = em.sel(I64, &k, &pv, &z64);
        let (sv, ok) = em.int[&r].clone();
        em.st(&sv, &sv0);
        let k8 = em.i1_to_i8(&k);
        em.st(&ok, &k8);
    }
    let float_regs: Vec<u32> = {
        let mut v: Vec<u32> = em.sh.float.iter().copied().collect();
        v.sort_unstable();
        v
    };
    let zf = em.f64const(0.0);
    for r in float_regs {
        let a = em.vaddr(&regs0, r);
        let t = em.ld_tag(&a);
        let want = em.tconst(em.lyt.t_float);
        let ty = em.tag_ty();
        let k = em.icmp(ty, "EQ", &t, &want);
        let pv = em.ld(F64, &a, em.lyt.val_pay as u32, 8);
        let sv0 = em.sel(F64, &k, &pv, &zf);
        let (sv, ok) = em.float[&r].clone();
        em.st(&sv, &sv0);
        let k8 = em.i1_to_i8(&k);
        em.st(&ok, &k8);
    }
    let d = em.ex.dispatch.clone();
    em.br(&d);

    // ---- dispatch: `llvm.switch` on `code.ip` (absolute byte offsets) ----
    em.start_block(&em.ex.dispatch.clone());
    if ops.is_empty() {
        let e = em.ex.edef.clone();
        em.br(&e);
        em.start_block(&em.ex.edef.clone());
        let a = em.v_curip.clone();
        em.st(&a, &z64);
        let g = em.ex.estep_g.clone();
        em.br(&g);
    } else {
        let ip = em.ld_word(&em.env.ip_p.clone(), 0);
        // dispatch on `code.ip - @off_{self}` so the case labels stay
        // chunk-local constants — the body's only index-free form.
        let base = em.off_base();
        let rel = em.sub(&ip, &base);
        let mut cases = String::new();
        for (i, (off, _)) in ops.iter().enumerate() {
            write!(
                cases,
                "\n      {{ <{}: i64>: ^op{i}() }},",
                off - em.chunk_off
            )
            .unwrap();
        }
        cases.pop();
        let edef = em.ex.edef.clone();
        em.ln(format!("llvm.switch {rel}, ^{edef}() [{cases}\n    ]"));
        em.start_block(&em.ex.edef.clone());
        let a = em.v_curip.clone();
        em.st(&a, &ip);
        let g = em.ex.estep_g.clone();
        em.br(&g);
    }

    // ---- shared exits ----
    // `estep_g`: quota gate before `edef`/`eend` fallbacks (see jit emit).
    em.start_block(&em.ex.estep_g.clone());
    let g_cont = em.create_block();
    let g_tramp = em.create_block();
    let pv = em.ld(I8, &em.env.paused_p.clone(), 0, 1);
    let p1 = em.icmp_ne0(I8, &pv);
    em.cbr(&p1, &g_tramp, &g_cont);
    em.start_block(&g_cont);
    let bcnv = em.bcn();
    let z = em.iconst(0);
    let bz = em.icmp(I64, "EQ", &bcnv, &z);
    let g2 = em.create_block();
    em.cbr(&bz, &g_tramp, &g2);
    em.start_block(&g2);
    let cur = em.bcn();
    let one = em.iconst(1);
    let b1 = em.sub(&cur, &one);
    let a = em.v_bcn.clone();
    em.st(&a, &b1);
    let es = em.ex.estep.clone();
    em.br(&es);
    em.fill_gate_tramp(&g_tramp, &g2);

    em.start_block(&em.ex.enext.clone());
    em.settle_seq();
    em.flush_seq();
    let ip = em.cur_ip();
    em.store_ip(&ip);
    em.hcall(H::OutNext, &[A::P(em.env.out.clone())]);
    em.ret();

    em.start_block(&em.ex.eoof.clone());
    em.settle_seq();
    em.flush_seq();
    let ip = em.cur_ip();
    em.store_opip(&ip);
    let k = em.iconst8(ERR_OOF);
    em.hcall(H::OutErr, &[A::P(em.env.out.clone()), A::I8(k)]);
    em.ret();

    em.start_block(&em.ex.estep.clone());
    em.settle_seq();
    em.flush_seq();
    let ip = em.cur_ip();
    em.store_ip(&ip);
    em.store_opip(&ip);
    let regs = em.regs();
    let k = em
        .hcall(
            H::StepAt,
            &[
                A::P(em.env.thread.clone()),
                A::P(regs),
                A::V(em.env.nregs.clone()),
                A::P(em.env.code.clone()),
                A::P(em.env.ctx0.clone()),
                A::P(em.env.ctx1.clone()),
                A::P(em.env.strs.clone()),
                A::P(em.env.out.clone()),
            ],
        )
        .unwrap();
    let k1 = em.icmp_ne0(I8, &k);
    let (er, dp) = (em.ex.eret.clone(), em.ex.dispatch.clone());
    em.cbr(&k1, &er, &dp);

    em.start_block(&em.ex.eerr.clone());
    em.settle_seq();
    em.flush_seq();
    let ip = em.cur_ip();
    em.store_opip(&ip);
    let a = em.v_ekind.clone();
    let kk = em.ld(I8, &a, 0, 1);
    em.hcall(H::OutErr, &[A::P(em.env.out.clone()), A::I8(kk)]);
    em.ret();

    em.start_block(&em.ex.eret.clone());
    em.settle_seq();
    em.ret();

    em.start_block(&em.ex.eend.clone());
    let max = em.iconst(-1); // usize::MAX — garbage decode, same as bcgen's `_ =>`
    let a = em.v_curip.clone();
    em.st(&a, &max);
    let g = em.ex.estep_g.clone();
    em.br(&g);

    // ---- op blocks ----
    for (i, (off, op)) in ops.iter().enumerate() {
        let next = ops.get(i + 1).map(|(o, _)| *o).unwrap_or(usize::MAX);
        em.emit_op(i, *off, next, op);
    }

    // ---- serialize ----
    // ops inside a block are `;`-separated (no trailing `;`); blocks and
    // module members are whitespace-separated.
    em.finish();
    let mut out = format!(
        "llvm.func @{}: llvm.func <{VOID} ({PTR}) variadic = false>\n    [llvm_function_linkage: llvm.linkage ExternalLinkage]\n  {{",
        body_name(body)
    );
    for (name, lines) in &em.done {
        if name == "entry" {
            write!(out, "\n  ^entry(env: {PTR}):").unwrap();
            if !em.prelude.is_empty() || !lines.is_empty() {
                write!(out, "\n    ").unwrap();
                write!(out, "{}", em.prelude.join(";\n    ")).unwrap();
                if !lines.is_empty() {
                    write!(out, ";\n    ").unwrap();
                }
            }
        } else {
            write!(out, "\n  ^{name}():").unwrap();
            if !lines.is_empty() {
                write!(out, "\n    ").unwrap();
            }
        }
        write!(out, "{}", lines.join(";\n    ")).unwrap();
        out.push('\n');
    }
    out.push_str("  }");
    out
}

impl Em<'_> {
    // ---- raw emission ----

    fn v(&mut self) -> String {
        let s = format!("v{}", self.nv);
        self.nv += 1;
        s
    }

    fn ln(&mut self, s: String) {
        self.cur.push(s);
    }

    /// Emit `v = <op>` returning the value name.
    fn ev(&mut self, op: String) -> String {
        let v = self.v();
        self.ln(format!("{v} = {op}"));
        v
    }

    fn create_block(&mut self) -> String {
        let s = format!("b{}", self.nb);
        self.nb += 1;
        s
    }

    /// Close the current block and start `name` (all blocks take no args).
    fn start_block(&mut self, name: &str) {
        if !self.cur_name.is_empty() {
            let done = std::mem::replace(&mut self.cur_name, name.to_string());
            let lines = std::mem::take(&mut self.cur);
            self.done.push((done, lines));
        } else {
            self.cur_name = name.to_string();
        }
    }

    fn finish(&mut self) {
        assert!(!self.cur_name.is_empty());
        let name = std::mem::take(&mut self.cur_name);
        let lines = std::mem::take(&mut self.cur);
        self.done.push((name, lines));
    }

    /// `alloca` into the entry prelude; returns the pointer value name.
    /// `a_one` is the shared `i32 1` size operand emitted first.
    fn alloca(&mut self, elem: &str, hint: &str) -> String {
        let v = format!("a_{hint}");
        self.prelude
            .push(format!("{v} = llvm.alloca [{elem} x a_one] : {PTR}"));
        v
    }

    /// `BodyEnv` field load: `gep env, off` → `load ptr`.
    fn envld(&mut self, env_p: &str, off: usize) -> String {
        let c = self.iconst(off as i64);
        let a = self.gep_dyn(env_p, &c);
        self.ld(PTR, &a, 0, self.lyt.word_size as u32)
    }

    // ---- value helpers ----

    fn iconst_ty(&mut self, ty: &str, v: i64) -> String {
        let w = match ty {
            I8 => v as i8 as i64,
            I16 => v as i16 as i64,
            I32 => v as i32 as i64,
            _ => v,
        };
        let ity = ty.strip_prefix("builtin.integer ").unwrap_or("i64");
        self.ev(format!(
            "llvm.constant <builtin.integer <{w}: {ity}>> : {ty}"
        ))
    }

    fn iconst(&mut self, v: i64) -> String {
        self.iconst_ty(I64, v)
    }

    fn iconst32(&mut self, v: i64) -> String {
        self.iconst_ty(I32, v)
    }

    fn iconst8(&mut self, v: i64) -> String {
        self.iconst_ty(I8, v)
    }

    fn f64const(&mut self, f: f64) -> String {
        // bitcast keeps the text path free of float-literal edge cases
        let i = self.iconst(f.to_bits() as i64);
        self.ev(format!("llvm.bitcast {i} to {F64}"))
    }

    /// `p + off` (byte offset), opaque-pointer `gep i8`.
    fn gep(&mut self, p: &str, off: u32) -> String {
        self.ev(format!("llvm.gep <{I8}> ({p})[Constant({off})] : {PTR}"))
    }

    fn gep_dyn(&mut self, p: &str, off: &str) -> String {
        self.ev(format!(
            "llvm.gep <{I8}> ({p}, {off})[OperandIdx(1)] : {PTR}"
        ))
    }

    fn ld(&mut self, ty: &str, p: &str, off: u32, align: u32) -> String {
        let a = if off == 0 {
            p.to_string()
        } else {
            self.gep(p, off)
        };
        self.ev(format!("llvm.load {a} [align : {align}] : {ty}"))
    }

    fn ldf(&mut self, p: &str) -> String {
        self.ld(F64, p, 0, 8)
    }

    fn ldp(&mut self, p: &str) -> String {
        self.ld(PTR, p, 0, 8)
    }

    fn st(&mut self, p: &str, v: &str) {
        self.st_al(p, v, 8);
    }

    fn st_al(&mut self, p: &str, v: &str, align: u32) {
        self.ln(format!("llvm.store *{p} <- {v} [align : {align}]"));
    }

    // `llvm.add`/`sub`/`mul` carry a mandatory IntegerOverflowFlagsAttr —
    // plain wraps get `<{nsw=false,nuw=false}>`.
    fn add(&mut self, a: &str, b: &str) -> String {
        self.ev(format!(
            "llvm.add {a}, {b} <{{nsw=false,nuw=false}}> : {I64}"
        ))
    }

    fn sub(&mut self, a: &str, b: &str) -> String {
        self.ev(format!(
            "llvm.sub {a}, {b} <{{nsw=false,nuw=false}}> : {I64}"
        ))
    }

    fn mul(&mut self, a: &str, b: &str) -> String {
        self.ev(format!(
            "llvm.mul {a}, {b} <{{nsw=false,nuw=false}}> : {I64}"
        ))
    }

    /// `a & b` on `i1` flags (the gate conjunctions in the inline call
    /// path — `llvm.and` takes no overflow-flags attr).
    fn band1(&mut self, a: &str, b: &str) -> String {
        self.ev(format!("llvm.and {a}, {b} : {I1}"))
    }

    /// Zero-extend to `i64` (`i32` fields like `Frame::return_reg`).
    fn zext64(&mut self, v: &str) -> String {
        self.ev(format!("llvm.zext <nneg=false> {v} to {I64}"))
    }

    fn sext64(&mut self, v: &str) -> String {
        self.ev(format!("llvm.sext {v} to {I64}"))
    }

    fn trunc_to(&mut self, ty: &str, v: &str) -> String {
        self.ev(format!("llvm.trunc {v} to {ty}"))
    }

    /// `usize` field load, zero-extended to i64 — the probed `word_size`
    /// is 4 on wasm32, where an i64 load would read the next field too.
    /// Arithmetic stays i64 everywhere downstream.
    fn ld_word(&mut self, p: &str, off: u32) -> String {
        let wty = self.wty(self.lyt.word_size);
        let v = self.ld(wty, p, off, self.lyt.word_size as u32);
        if self.lyt.word_size < 8 {
            self.zext64(&v)
        } else {
            v
        }
    }

    /// `isize` borrow-flag load — sign-extended (negative = live
    /// `borrow_mut`, which the fast paths route to the slow call).
    fn ld_flag(&mut self, p: &str, off: u32) -> String {
        let wty = self.wty(self.lyt.word_size);
        let v = self.ld(wty, p, off, self.lyt.word_size as u32);
        if self.lyt.word_size < 8 {
            self.sext64(&v)
        } else {
            v
        }
    }

    /// `usize` field store — truncates the i64 working value on wasm32.
    fn st_word(&mut self, p: &str, v: &str, off: u32) {
        let owned;
        let v = if self.lyt.word_size < 8 {
            let wty = self.wty(self.lyt.word_size);
            owned = self.trunc_to(wty, v);
            &owned
        } else {
            v
        };
        self.st_off(p, v, off, self.lyt.word_size as u32);
    }

    /// `a <cc> b` as `i1`, for `_oty`-typed operands. The trailing type in
    /// `llvm.icmp`/`llvm.fcmp` is the *result* type (`type($0)` binds
    /// `result[0]`, not the operand type like LLVM's syntax) — annotating
    /// the operand width here gave every compare an `oty`-typed result,
    /// which then fed `i1` slots (`cond_br`, `and : i1`, `select`).
    fn icmp(&mut self, _oty: &str, cc: &str, a: &str, b: &str) -> String {
        self.ev(format!("llvm.icmp {a} <{cc}> {b} : {I1}"))
    }

    /// `a != 0` as `i1`, for `oty`-typed `a`.
    fn icmp_ne0(&mut self, oty: &str, a: &str) -> String {
        let z = self.iconst_ty(oty, 0);
        self.icmp(oty, "NE", a, &z)
    }

    fn fcmp(&mut self, cc: &str, a: &str, b: &str) -> String {
        self.ev(format!("llvm.fcmp {a} <{cc}> {b} : {I1}"))
    }

    /// `i1 → i8` for flag/`Val::Bool` payloads (`c ? 1 : 0`).
    fn i1_to_i8(&mut self, c: &str) -> String {
        let one = self.iconst8(1);
        let z = self.iconst8(0);
        self.sel(I8, c, &one, &z)
    }

    fn i2p(&mut self, v: &str) -> String {
        self.ev(format!("llvm.inttoptr {v} to {PTR}"))
    }

    fn p2i(&mut self, v: &str) -> String {
        self.ev(format!("llvm.ptrtoint {v} to {I64}"))
    }

    fn sel(&mut self, ty: &str, c: &str, t: &str, f: &str) -> String {
        self.ev(format!("llvm.select {c} ? {t} : {f} : {ty}"))
    }

    fn br(&mut self, t: &str) {
        self.ln(format!("llvm.br ^{t}()"));
    }

    fn cbr(&mut self, c: &str, t: &str, f: &str) {
        self.ln(format!("llvm.cond_br if {c} ^{t}() else ^{f}()"));
    }

    fn ret(&mut self) {
        self.ln("llvm.return".into());
    }

    /// Call a `vm::bc::jit` helper as a direct `llvm.call @mj_*` symbol
    /// (resolved at link time against the companion runtime); returns the
    /// `u8` status or pointer result if the signature has one.
    fn hcall(&mut self, h: H, args: &[A]) -> Option<String> {
        let (name, params, ret) = spec(h);
        // expand the flat arg list over `Ctx` (two P params)
        let mut av: Vec<String> = Vec::new();
        let mut ai = args.iter();
        for p in params.iter() {
            match p {
                Pt::Ctx => {
                    if self.wasm {
                        // Two arg slots but one ABI word: `&env.mutation`
                        // addresses the live `{mutation, state}` pair
                        // (adjacent `repr(C)` fields) — the indirect
                        // `Ctx` the callee dereferences.
                        ai.next().unwrap();
                        ai.next().unwrap();
                        av.push(self.env.ctx_p.clone());
                    } else {
                        for _ in 0..2 {
                            match ai.next().unwrap() {
                                A::V(v) => {
                                    let v = v.clone();
                                    let p = self.i2p(&v);
                                    av.push(p);
                                }
                                A::P(v) => av.push(v.clone()),
                                _ => panic!("Ctx param needs a word/ptr arg"),
                            }
                        }
                    }
                }
                Pt::P => match ai.next().unwrap() {
                    A::V(v) => {
                        let v = v.clone();
                        let p = self.i2p(&v);
                        av.push(p);
                    }
                    A::P(v) => av.push(v.clone()),
                    _ => panic!("P param needs a word/ptr arg"),
                },
                Pt::I8 => match ai.next().unwrap() {
                    A::I8(v) => av.push(v.clone()),
                    _ => panic!("I8 param needs an i8 arg"),
                },
                Pt::I32 => match ai.next().unwrap() {
                    A::I32(v) => av.push(v.clone()),
                    _ => panic!("I32 param needs an i32 arg"),
                },
                Pt::F64 => match ai.next().unwrap() {
                    A::F64(v) => av.push(v.clone()),
                    _ => panic!("F64 param needs an f64 arg"),
                },
            }
        }
        assert!(ai.next().is_none(), "extra args for {name}");
        let (ret_ty, ptys) = sig(params, ret, self.wasm);
        let fty = format!("llvm.func <{ret_ty} ({ptys}) variadic = false>");
        // `llvm.call` is one-result even for `llvm.void` — always bind.
        let v = self.v();
        self.ln(format!(
            "{v} = llvm.call @{name} ({}) : {fty}",
            av.join(", ")
        ));
        match ret {
            Hr::Void => None,
            _ => Some(v),
        }
    }

    // ---- alloca-variable plumbing (`use_var`/`def_var` analogues) ----

    fn regs(&mut self) -> String {
        self.ldp(&self.v_regs.clone())
    }

    fn cur_ip(&mut self) -> String {
        self.ld(I64, &self.v_curip.clone(), 0, 8)
    }

    fn bcn(&mut self) -> String {
        self.ld(I64, &self.v_bcn.clone(), 0, 8)
    }

    fn sh_int(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.sh.int.iter().copied().collect();
        v.sort_unstable();
        v
    }

    fn sh_float(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.sh.float.iter().copied().collect();
        v.sort_unstable();
        v
    }

    // ---- inline `Val` access over the probed layout (`lyt`) ----

    /// `thread.frames.len()`.
    fn frames_len(&mut self) -> String {
        let c = self.iconst((self.lyt.frames_off + self.lyt.vec_len) as i64);
        let a = self.gep_dyn(&self.env.thread.clone(), &c);
        self.ld_word(&a, 0)
    }

    /// `regs + r*val_size` — `&regs[r]` for a window base `regs`.
    fn vaddr(&mut self, regs: &str, r: u32) -> String {
        self.gep(regs, r * self.lyt.val_size as u32)
    }

    /// The integer IR type of byte width `w` (any probed tag/flag width).
    fn wty(&self, w: usize) -> &'static str {
        match w {
            1 => I8,
            2 => I16,
            4 => I32,
            8 => I64,
            d => unreachable!("bad tag width {d}"),
        }
    }

    /// The discriminant's IR type (its probed byte width).
    fn tag_ty(&self) -> &'static str {
        self.wty(self.lyt.tag_size)
    }

    /// A discriminant constant at the probed width.
    fn tconst(&mut self, t: u64) -> String {
        self.iconst_ty(self.tag_ty(), t as i64)
    }

    /// `regs[r]`'s discriminant.
    fn ld_tag(&mut self, a: &str) -> String {
        self.ld(
            self.tag_ty(),
            a,
            self.lyt.val_tag as u32,
            self.lyt.tag_size as u32,
        )
    }

    /// `regs[r]` as `Val::Int`: inline tag probe → payload load; a miss
    /// routes to `estep` (the `ri` helper's `0` return, inlined).
    fn ld_int(&mut self, regs: &str, r: u32) -> String {
        let good = self.create_block();
        let a = self.vaddr(regs, r);
        let t = self.ld_tag(&a);
        let want = self.tconst(self.lyt.t_int);
        let ty = self.tag_ty();
        let hit = self.icmp(ty, "EQ", &t, &want);
        let es = self.ex.estep.clone();
        self.cbr(&hit, &good, &es);
        self.start_block(&good);
        self.ld(I64, &a, self.lyt.val_pay as u32, 8)
    }

    /// `regs[r]` as `Val::Float`.
    fn ld_float(&mut self, regs: &str, r: u32) -> String {
        let good = self.create_block();
        let a = self.vaddr(regs, r);
        let t = self.ld_tag(&a);
        let want = self.tconst(self.lyt.t_float);
        let ty = self.tag_ty();
        let hit = self.icmp(ty, "EQ", &t, &want);
        let es = self.ex.estep.clone();
        self.cbr(&hit, &good, &es);
        self.start_block(&good);
        self.ld(F64, &a, self.lyt.val_pay as u32, 8)
    }

    /// `*a = Val::Int(v)` — tag byte plus the 8-byte union slot.
    fn st_int(&mut self, a: &str, v: &str) {
        let t = self.tconst(self.lyt.t_int);
        self.st_off(a, &t, self.lyt.val_tag as u32, self.lyt.tag_size as u32);
        self.st_off(a, v, self.lyt.val_pay as u32, 8);
    }

    /// `*a = Val::Float(v)` (ditto, `f64` store).
    fn st_float(&mut self, a: &str, v: &str) {
        let t = self.tconst(self.lyt.t_float);
        self.st_off(a, &t, self.lyt.val_tag as u32, self.lyt.tag_size as u32);
        self.st_off(a, v, self.lyt.val_pay as u32, 8);
    }

    /// `*a = Val::Bool(v8)` — tag plus the `u8` union slot.
    fn st_bool(&mut self, a: &str, v8: &str) {
        let t = self.tconst(self.lyt.t_bool);
        self.st_off(a, &t, self.lyt.val_tag as u32, self.lyt.tag_size as u32);
        self.st_off(a, v8, self.lyt.bool_pay as u32, 1);
    }

    /// `*a = Val::Null` — only the tag byte is ever read.
    fn st_null(&mut self, a: &str) {
        let t = self.tconst(self.lyt.t_null);
        self.st_off(a, &t, self.lyt.val_tag as u32, self.lyt.tag_size as u32);
    }

    /// `*a = Val::Fn(body)` — tag plus the `u32` index in the union slot.
    fn st_fn(&mut self, a: &str, body32: &str) {
        let t = self.tconst(self.lyt.t_fn);
        self.st_off(a, &t, self.lyt.val_tag as u32, self.lyt.tag_size as u32);
        self.st_off(a, body32, self.lyt.fn_pay as u32, 4);
    }

    fn st_off(&mut self, a: &str, v: &str, off: u32, align: u32) {
        let p = if off == 0 {
            a.to_string()
        } else {
            self.gep(a, off)
        };
        self.st_al(&p, v, align);
    }

    /// `val_size`-byte `Val` copy `*d = *s` (eight-byte chunks; the probe
    /// asserts the stride is a multiple of 8).
    fn cpy_val(&mut self, d: &str, s: &str) {
        for k in 0..self.lyt.val_size as u32 / 8 {
            let w = self.ld(I64, s, k * 8, 8);
            let p = self.gep(d, k * 8);
            self.st_al(&p, &w, 8);
        }
    }

    /// Fast-path tail for an inline container read: `regs[dst] = *elem`,
    /// re-derive dst's shadow from the authoritative slot, then `next`.
    fn copy_to_dst(&mut self, i: usize, dst: Reg, elem: &str) {
        let regs = self.regs();
        let d = self.vaddr(&regs, dst.index() as u32);
        self.cpy_val(&d, elem);
        self.refresh_shadow(dst.index() as u32);
        let nb = self.next_blk(i);
        self.br(&nb);
    }

    /// `llvm.switch idx, ^dflt() [{ <j: i64>: ^tgt_j() }]` — one case per
    /// table entry; an `idx` that matches no case (negative, past the end,
    /// wrapped) lands on `dflt`, which is what the interpreter's
    /// bounds check does with it.
    fn emit_switch_tbl(&mut self, idx: &str, dflt: &str, table: &[BlockTarget]) {
        let mut cases = String::new();
        for (j, t) in table.iter().enumerate() {
            let b = self.tgt_blk(t);
            write!(cases, "\n      {{ <{j}: i64>: ^{b}() }},").unwrap();
        }
        cases.pop();
        self.ln(format!("llvm.switch {idx}, ^{dflt}() [{cases}\n    ]"));
    }

    /// The block implementing the op at byte offset `t` — jump targets
    /// always land on op starts, so the map is exact.
    fn tgt_blk(&self, t: &BlockTarget) -> String {
        let BlockTarget::ByteOffset(o) = t else {
            panic!("unresolved BlockTarget in compiled program")
        };
        self.blocks[self.off2idx[o]].clone()
    }

    /// Successor block for fallthrough (`next` op, or `eend` off the end).
    fn next_blk(&self, i: usize) -> String {
        if i + 1 < self.ops.len() {
            self.blocks[i + 1].clone()
        } else {
            self.ex.eend.clone()
        }
    }

    /// "Flush shadows, then jump to `cont`" — used when control rejoins a
    /// shared block rather than continuing in the current chain.
    fn flush_jump(&mut self, cont: &str) {
        self.flush_seq();
        self.br(cont);
    }

    /// The inline shadow-writeback sequence — see `jit/src/emit.rs`
    /// `flush_seq`; a `select` per field keeps it branch-free.
    fn flush_seq(&mut self) {
        if self.sh.int.is_empty() && self.sh.float.is_empty() {
            return;
        }
        let regs = self.regs();
        let tint = self.tconst(self.lyt.t_int);
        let tag_ty = self.tag_ty();
        for r in self.sh_int() {
            let (sv_a, ok_a) = self.int[&r].clone();
            let okv = self.ld(I8, &ok_a, 0, 1);
            let ok1 = self.icmp_ne0(I8, &okv);
            let v = self.ld(I64, &sv_a, 0, 8);
            let a = self.vaddr(&regs, r);
            let old = self.ld_tag(&a);
            let nt = self.sel(tag_ty, &ok1, &tint, &old);
            self.st_off(&a, &nt, self.lyt.val_tag as u32, self.lyt.tag_size as u32);
            let oldp = self.ld(I64, &a, self.lyt.val_pay as u32, 8);
            let np = self.sel(I64, &ok1, &v, &oldp);
            self.st_off(&a, &np, self.lyt.val_pay as u32, 8);
        }
        let tflt = self.tconst(self.lyt.t_float);
        for r in self.sh_float() {
            let (sv_a, ok_a) = self.float[&r].clone();
            let okv = self.ld(I8, &ok_a, 0, 1);
            let ok1 = self.icmp_ne0(I8, &okv);
            let v = self.ldf(&sv_a);
            let a = self.vaddr(&regs, r);
            let old = self.ld_tag(&a);
            let nt = self.sel(tag_ty, &ok1, &tflt, &old);
            self.st_off(&a, &nt, self.lyt.val_tag as u32, self.lyt.tag_size as u32);
            let oldp = self.ld(F64, &a, self.lyt.val_pay as u32, 8);
            let np = self.sel(F64, &ok1, &v, &oldp);
            self.st_off(&a, &np, self.lyt.val_pay as u32, 8);
        }
    }

    /// `code.ip = v` through the hoisted cell pointer.
    fn store_ip(&mut self, v: &str) {
        let p = self.env.ip_p.clone();
        self.st_word(&p, v, 0);
    }

    /// `*op_ip = v` through the hoisted cell pointer.
    fn store_opip(&mut self, v: &str) {
        let p = self.env.opip_p.clone();
        self.st_word(&p, v, 0);
    }

    /// `code.ip = next; *op_ip = off` — bcgen's pre-helper/pre-call idiom.
    /// `off`/`next` are program code offsets; `goff` keeps them
    /// index-free in the emitted text.
    fn mark_op(&mut self, off: usize, next: usize) {
        let n = self.goff(next);
        let o = self.goff(off);
        self.store_ip(&n);
        self.store_opip(&o);
    }

    /// Global code offset for one of this chunk's ops: `@off_{self}` +
    /// `(off - chunk.offset)`. The `off` globals are defined in the init
    /// module; a chunk module sees bodiless decls, so the emitted code
    /// carries no program-assigned constant.
    fn goff(&mut self, off: usize) -> String {
        let b = self.off_base();
        let l = self.iconst((off - self.chunk_off) as i64);
        self.add(&b, &l)
    }

    /// `*@mimas_body_off_{self_i}`, loaded once.
    fn off_base(&mut self) -> String {
        if self.v_offbase.is_empty() {
            let v = self.v();
            self.ln(format!(
                "{v} = llvm.addressof @mimas_body_off_{} : {PTR}",
                self.self_i
            ));
            self.v_offbase = self.ld_word(&v, 0);
        }
        self.v_offbase.clone()
    }

    /// `*@mimas_body_off_{callee}` — a callee chunk's code offset
    /// (`Chunk::offset`), read from the per-program global.
    fn chunk_off(&mut self, callee: usize) -> String {
        let v = self.v();
        self.ln(format!("{v} = llvm.addressof @mimas_body_off_{callee} : {PTR}"));
        self.ld_word(&v, 0)
    }

    /// `*@mimas_body_idx_{callee}` — a callee chunk's body-table index,
    /// read from the per-program global.
    fn body_idx(&mut self, callee: usize) -> String {
        let v = self.v();
        self.ln(format!("{v} = llvm.addressof @mimas_body_idx_{callee} : {PTR}"));
        self.ld_word(&v, 0)
    }

    /// Route an `RtErr` kind to `eerr` through a trampoline that just sets
    /// `ekind` — created empty, filled by [`Em::fill_err`].
    fn err_tramp(&mut self) -> String {
        self.create_block()
    }

    /// Fill a trampoline created by [`Em::err_tramp`]: `ekind = kind; →eerr`.
    fn fill_err(&mut self, t: &str, kind: i64) {
        self.start_block(t);
        let k = self.iconst8(kind);
        let a = self.v_ekind.clone();
        self.st(&a, &k);
        let e = self.ex.eerr.clone();
        self.br(&e);
    }

    /// Read `regs[r]` as an i64 — shadow when eligible, else the inline tag
    /// probe; a miss routes to `estep` (see `jit/src/emit.rs::int_opnd`).
    fn int_opnd(&mut self, r: u32) -> String {
        if self.int.contains_key(&r) {
            let good = self.create_block();
            let ok_a = self.int[&r].1.clone();
            let okv = self.ld(I8, &ok_a, 0, 1);
            let ok1 = self.icmp_ne0(I8, &okv);
            let es = self.ex.estep.clone();
            self.cbr(&ok1, &good, &es);
            self.start_block(&good);
            let a = self.int[&r].0.clone();
            self.ld(I64, &a, 0, 8)
        } else {
            let regs = self.regs();
            self.ld_int(&regs, r)
        }
    }

    fn float_opnd(&mut self, r: u32) -> String {
        if self.float.contains_key(&r) {
            let good = self.create_block();
            let ok_a = self.float[&r].1.clone();
            let okv = self.ld(I8, &ok_a, 0, 1);
            let ok1 = self.icmp_ne0(I8, &okv);
            let es = self.ex.estep.clone();
            self.cbr(&ok1, &good, &es);
            self.start_block(&good);
            let a = self.float[&r].0.clone();
            self.ldf(&a)
        } else {
            let regs = self.regs();
            self.ld_float(&regs, r)
        }
    }

    /// Scalar write into `dst`: shadow-var update when shadowed, else an
    /// inline `Val` store.
    fn wr_int_dst(&mut self, d: u32, v: &str) {
        if let Some((sv, ok)) = self.int.get(&d).cloned() {
            self.st(&sv, v);
            let one = self.iconst8(1);
            self.st(&ok, &one);
        } else {
            let regs = self.regs();
            let a = self.vaddr(&regs, d);
            self.st_int(&a, v);
        }
    }

    fn wr_float_dst(&mut self, d: u32, v: &str) {
        if let Some((sv, ok)) = self.float.get(&d).cloned() {
            self.st(&sv, v);
            let one = self.iconst8(1);
            self.st(&ok, &one);
        } else {
            let regs = self.regs();
            let a = self.vaddr(&regs, d);
            self.st_float(&a, v);
        }
    }

    /// Bool write into `dst` — eval dsts are always `W::Dyn` (unshadowed).
    fn wr_bool_dst(&mut self, d: u32, v8: &str) {
        let regs = self.regs();
        let a = self.vaddr(&regs, d);
        self.st_bool(&a, v8);
    }

    /// A `*const Val` for reg `s`, materializing a live shadow first — after
    /// this `regs[s]` is authoritative for the read.
    fn val_ptr(&mut self, s: u32) -> String {
        if let Some((sv_a, ok_a)) = self.int.get(&s).cloned() {
            let c = self.create_block();
            let m = self.create_block();
            let okv = self.ld(I8, &ok_a, 0, 1);
            let ok1 = self.icmp_ne0(I8, &okv);
            self.cbr(&ok1, &m, &c);
            self.start_block(&m);
            let regs = self.regs();
            let a = self.vaddr(&regs, s);
            let v = self.ld(I64, &sv_a, 0, 8);
            self.st_int(&a, &v);
            self.br(&c);
            self.start_block(&c);
        } else if let Some((sv_a, ok_a)) = self.float.get(&s).cloned() {
            let c = self.create_block();
            let m = self.create_block();
            let okv = self.ld(I8, &ok_a, 0, 1);
            let ok1 = self.icmp_ne0(I8, &okv);
            self.cbr(&ok1, &m, &c);
            self.start_block(&m);
            let regs = self.regs();
            let a = self.vaddr(&regs, s);
            let v = self.ldf(&sv_a);
            self.st_float(&a, &v);
            self.br(&c);
            self.start_block(&c);
        }
        let regs = self.regs();
        self.vaddr(&regs, s)
    }

    /// Helper-op shape: inline shadow writeback, then stores `code.ip`/
    /// `*op_ip`, calls the `u8` helper, then `1`→`eret` / `0`→fallthrough.
    /// `args` are built by the caller before this runs — the writeback
    /// blocks are all dominated by the current block, so they stay usable.
    fn helper_op(&mut self, i: usize, off: usize, next: usize, h: H, args: &[A]) {
        self.flush_seq();
        self.mark_op(off, next);
        let k = self.hcall(h, args).unwrap();
        let k1 = self.icmp_ne0(I8, &k);
        let nb = self.next_blk(i);
        let er = self.ex.eret.clone();
        self.cbr(&k1, &er, &nb);
    }

    /// Same but the helper returns void — always continues.
    fn helper_op_v(&mut self, i: usize, off: usize, next: usize, h: H, args: &[A]) {
        self.flush_seq();
        self.mark_op(off, next);
        self.hcall(h, args);
        let nb = self.next_blk(i);
        self.br(&nb);
    }

    /// After a helper wrote `regs[r]` behind the shadow's back (a container
    /// read's slow path), re-derive the `DynCheck` `(sv, ok)` pair from the
    /// authoritative slot.
    fn refresh_shadow(&mut self, r: u32) {
        if !self.int.contains_key(&r) && !self.float.contains_key(&r) {
            return;
        }
        let regs = self.regs();
        let a = self.vaddr(&regs, r);
        let t = self.ld_tag(&a);
        let ty = self.tag_ty();
        if let Some((sv, ok)) = self.int.get(&r).cloned() {
            let want = self.tconst(self.lyt.t_int);
            let k = self.icmp(ty, "EQ", &t, &want);
            let pv = self.ld(I64, &a, self.lyt.val_pay as u32, 8);
            self.st(&sv, &pv);
            let k8 = self.i1_to_i8(&k);
            self.st(&ok, &k8);
        }
        if let Some((sv, ok)) = self.float.get(&r).cloned() {
            let want = self.tconst(self.lyt.t_float);
            let k = self.icmp(ty, "EQ", &t, &want);
            let pv = self.ld(F64, &a, self.lyt.val_pay as u32, 8);
            self.st(&sv, &pv);
            let k8 = self.i1_to_i8(&k);
            self.st(&ok, &k8);
        }
    }

    /// `helper_op` + a post-call [`Em::refresh_shadow`] on `dst`.
    fn helper_op_read(&mut self, i: usize, off: usize, next: usize, h: H, args: &[A], dst: Reg) {
        self.flush_seq();
        self.mark_op(off, next);
        let k = self.hcall(h, args).unwrap();
        let k1 = self.icmp_ne0(I8, &k);
        let post = self.create_block();
        let er = self.ex.eret.clone();
        self.cbr(&k1, &er, &post);
        self.start_block(&post);
        self.refresh_shadow(dst.index() as u32);
        let nb = self.next_blk(i);
        self.br(&nb);
    }

    /// A `u32` register-index list (call args / field regs / captures) in a
    /// deduplicated entry alloca; returns its address for the helper.
    fn reg_list_slot(&mut self, regs_idx: &[Reg]) -> String {
        let key: Vec<u32> = regs_idx.iter().map(|r| r.index() as u32).collect();
        if let Some(a) = self.rl.get(&key) {
            return a.clone();
        }
        let n = key.len().max(1);
        let id = self.rl.len();
        self.prelude.push(format!(
            "rl{id}_n = llvm.constant <builtin.integer <{n}: i32>> : {I32}"
        ));
        let v = format!("a_rl{id}");
        self.prelude
            .push(format!("{v} = llvm.alloca [{I32} x rl{id}_n] : {PTR}"));
        for (j, r) in key.iter().enumerate() {
            self.prelude.push(format!(
                "rl{id}_c{j} = llvm.constant <builtin.integer <{r}: i32>> : {I32}"
            ));
            self.prelude.push(format!(
                "{v}_s{j} = llvm.gep <{I32}> ({v})[Constant({j})] : {PTR}"
            ));
            self.prelude
                .push(format!("llvm.store *{v}_s{j} <- rl{id}_c{j} [align : 4]"));
        }
        self.rl.insert(key, v.clone());
        v
    }

    /// `if frames.len() == 1 { flush }` before producing `Flow::Return` —
    /// the host reads the root frame's regs after `run()`.
    fn root_flush(&mut self, cont: &str) {
        let flen = self.frames_len();
        let one = self.iconst(1);
        let is_root = self.icmp(I64, "EQ", &flen, &one);
        let ft = self.create_block();
        self.cbr(&is_root, &ft, cont);
        self.start_block(&ft);
        self.flush_jump(cont);
    }

    /// `step_at`-backed whole-op fallback: `estep` itself writes
    /// `code.ip`/`*op_ip` from `cur_ip` and runs `step`.
    fn estep(&mut self) {
        let e = self.ex.estep.clone();
        self.br(&e);
    }

    // ---- batched quota (`bcn`) plumbing — bcgen's settle!/rearm ----

    /// `spent = bcn0 - bcn; bcn0 = bcn; *fuel -= spent; ops_left -= spent`.
    /// Idempotent; returns the post-settle `(fuel, ops_left)` values.
    fn settle_seq(&mut self) -> (String, String) {
        let b0 = self.ld(I64, &self.v_bcn0.clone(), 0, 8);
        let b = self.bcn();
        let spent = self.sub(&b0, &b);
        let a = self.v_bcn0.clone();
        self.st(&a, &b);
        let f = self.ld_word(&self.env.fuel_p.clone(), 0);
        let f2 = self.sub(&f, &spent);
        let p = self.env.fuel_p.clone();
        self.st_word(&p, &f2, 0);
        let ol = self.ld(I64, &self.env.opsleft_p.clone(), 0, 8);
        let ol2 = self.sub(&ol, &spent);
        let p = self.env.opsleft_p.clone();
        self.st(&p, &ol2);
        (f2, ol2)
    }

    /// `bcn = min(*fuel, ops_left); bcn0 = bcn` — armed at entry and re-armed
    /// wherever quota was spent outside our count (a megashim callee draws
    /// from the same counters through its own `bcn`).
    fn rearm_seq(&mut self) {
        let f = self.ld_word(&self.env.fuel_p.clone(), 0);
        let ol = self.ld(I64, &self.env.opsleft_p.clone(), 0, 8);
        let lt = self.icmp(I64, "ULT", &f, &ol);
        let m = self.sel(I64, &lt, &f, &ol);
        let a = self.v_bcn.clone();
        self.st(&a, &m);
        let a = self.v_bcn0.clone();
        self.st(&a, &m);
    }

    /// One op's quota gate — see `jit/src/emit.rs::gate`. Returns the
    /// continuation block the op body emits into.
    fn gate(&mut self, i: usize, off: usize) -> (String, String) {
        let o = self.goff(off);
        let a = self.v_curip.clone();
        self.st(&a, &o);
        let cont = self.create_block();
        let tramp = self.create_block();
        if i > 0 && may_pause(&self.ops[i - 1].1) {
            let p = self.ld(I8, &self.env.paused_p.clone(), 0, 1);
            let p1 = self.icmp_ne0(I8, &p);
            let g = self.create_block();
            self.cbr(&p1, &tramp, &g);
            self.start_block(&g);
        }
        let bcnv = self.bcn();
        let z = self.iconst(0);
        let bz = self.icmp(I64, "EQ", &bcnv, &z);
        self.cbr(&bz, &tramp, &cont);
        self.start_block(&cont);
        let cur = self.bcn();
        let one = self.iconst(1);
        let b1 = self.sub(&cur, &one);
        let a = self.v_bcn.clone();
        self.st(&a, &b1);
        (cont, tramp)
    }

    /// Fill a gate trampoline: settle, then `paused || fuel == 0` → `enext`,
    /// `ops_left == 0` → `eoof`, else re-arm and resume at `cont`.
    fn fill_gate_tramp(&mut self, tramp: &str, cont: &str) {
        self.start_block(tramp);
        let (f, ol) = self.settle_seq();
        let p = self.ld(I8, &self.env.paused_p.clone(), 0, 1);
        let p1 = self.icmp_ne0(I8, &p);
        let z = self.iconst(0);
        let fz = self.icmp(I64, "EQ", &f, &z);
        let nx = self.ev(format!("llvm.or {p1}, {fz} : {I1}"));
        let t2 = self.create_block();
        let en = self.ex.enext.clone();
        self.cbr(&nx, &en, &t2);
        self.start_block(&t2);
        let oz = self.icmp(I64, "EQ", &ol, &z);
        let t3 = self.create_block();
        let eo = self.ex.eoof.clone();
        self.cbr(&oz, &eo, &t3);
        self.start_block(&t3);
        self.rearm_seq();
        self.br(cont);
    }

    // ---- op semantics ----

    /// One op block: the driver's per-op quota gate, then semantics.
    fn emit_op(&mut self, i: usize, off: usize, next: usize, op: &Op) {
        let name = self.blocks[i].clone();
        self.start_block(&name);
        let (_cont, tramp) = self.gate(i, off);
        self.semantics(i, off, next, op);
        self.fill_gate_tramp(&tramp, &_cont);
    }

    fn semantics(&mut self, i: usize, off: usize, next: usize, op: &Op) {
        match op {
            Op::Move { dst, src } => {
                self.emit_move(*dst, *src);
                let nb = self.next_blk(i);
                self.br(&nb);
            }
            Op::Jump { target } => {
                let t = self.tgt_blk(target);
                self.br(&t);
            }
            Op::JumpIf {
                cond,
                target,
                is_true,
            } => {
                // `regs[cond] == Val::Bool(is_true)` — tag + payload byte.
                let regs = self.regs();
                let a = self.vaddr(&regs, cond.index() as u32);
                let t = self.ld_tag(&a);
                let want = self.tconst(self.lyt.t_bool);
                let ty = self.tag_ty();
                let tb2 = self.icmp(ty, "EQ", &t, &want);
                let pb = self.ld(I8, &a, self.lyt.bool_pay as u32, 1);
                let bv = self.iconst8(*is_true as i64);
                let peq = self.icmp(I8, "EQ", &pb, &bv);
                let k = self.ev(format!("llvm.and {tb2}, {peq} : {I1}"));
                let hit = self.mask_by_shadow(*cond, &k);
                let (tb, fb) = (self.tgt_blk(target), self.next_blk(i));
                self.cbr(&hit, &tb, &fb);
            }
            Op::ForNext { idx, bound, target } => {
                let iv = self.int_opnd(idx.index() as u32);
                let bv = self.int_opnd(bound.index() as u32);
                let one = self.iconst(1);
                let i2 = self.add(&iv, &one);
                self.wr_int_dst(idx.index() as u32, &i2);
                let hit = self.icmp(I64, "SLT", &i2, &bv);
                let (tb, fb) = (self.tgt_blk(target), self.next_blk(i));
                self.cbr(&hit, &tb, &fb);
            }
            Op::Switch {
                scrut,
                base,
                default,
                table,
            } => self.emit_switch(*scrut, *base, default, table),
            Op::Format { parts, .. } => {
                // selective materialization instead of `flush_seq` — only the
                // value parts' regs are read by `mj_format`.
                let mut seen: Vec<usize> = Vec::new();
                for part in parts {
                    if let OpFormatPart::Value(r) = part
                        && !seen.contains(&r.index())
                    {
                        seen.push(r.index());
                        self.val_ptr(r.index() as u32);
                    }
                }
                self.mark_op(off, next);
                let regs = self.regs();
                let o = self.goff(off);
                let k = self
                    .hcall(
                        H::Format,
                        &[
                            A::P(regs),
                            A::P(self.env.code.clone()),
                            A::V(o),
                            A::P(self.env.ctx0.clone()),
                            A::P(self.env.ctx1.clone()),
                            A::P(self.env.strs.clone()),
                            A::P(self.env.out.clone()),
                        ],
                    )
                    .unwrap();
                let k1 = self.icmp_ne0(I8, &k);
                let nb = self.next_blk(i);
                let er = self.ex.eret.clone();
                self.cbr(&k1, &er, &nb);
            }
            Op::Return { val } => {
                let vp = self.val_ptr(val.index() as u32);
                let do_ret = self.create_block();
                self.root_flush(&do_ret);
                self.start_block(&do_ret);
                self.settle_seq();
                self.hcall(H::OutReturn, &[A::P(self.env.out.clone()), A::P(vp)]);
                self.ret();
            }
            Op::Panic {} => {
                let k = self.iconst8(ERR_PANIC);
                let a = self.v_ekind.clone();
                self.st(&a, &k);
                let e = self.ex.eerr.clone();
                self.br(&e);
            }
            // jit bodies bypass call tracing like bcgen ones; no debugger
            // reads a jit frame: the shadow copy is dropped.
            Op::TraceEdge { .. } | Op::Shadow { .. } => {
                let nb = self.next_blk(i);
                self.br(&nb);
            }
            Op::Raise { val } => {
                self.flush_seq();
                self.mark_op(off, next);
                let regs = self.regs();
                let s = self.iconst(val.index() as i64);
                self.hcall(
                    H::Raise,
                    &[
                        A::P(regs),
                        A::V(s),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                        A::P(self.env.out.clone()),
                    ],
                );
                let e = self.ex.eret.clone();
                self.root_flush(&e);
            }
            Op::LoadConst { dst, constant } => match constant {
                Constant::Int(v) => {
                    let v = self.iconst(*v);
                    self.wr_int_dst(dst.index() as u32, &v);
                    let nb = self.next_blk(i);
                    self.br(&nb);
                }
                Constant::Float(v) => {
                    let v = self.f64const(*v);
                    self.wr_float_dst(dst.index() as u32, &v);
                    let nb = self.next_blk(i);
                    self.br(&nb);
                }
                Constant::Bool(b) => {
                    let v = self.iconst8(*b as i64);
                    let regs = self.regs();
                    let a = self.vaddr(&regs, dst.index() as u32);
                    self.st_bool(&a, &v);
                    let nb = self.next_blk(i);
                    self.br(&nb);
                }
                Constant::Null => {
                    let regs = self.regs();
                    let a = self.vaddr(&regs, dst.index() as u32);
                    self.st_null(&a);
                    let nb = self.next_blk(i);
                    self.br(&nb);
                }
                Constant::Str(id) => {
                    let regs = self.regs();
                    let d = self.iconst(dst.index() as i64);
                    let id = self.iconst32(id.index() as i64);
                    self.helper_op_v(
                        i,
                        off,
                        next,
                        H::LoadConstStr,
                        &[
                            A::P(regs),
                            A::V(d),
                            A::I32(id),
                            A::P(self.env.ctx0.clone()),
                            A::P(self.env.ctx1.clone()),
                            A::P(self.env.strs.clone()),
                        ],
                    );
                }
                Constant::Array(_) | Constant::Dec(_) => self.estep(),
            },
            Op::LoadBody { dst, body } => {
                let regs = self.regs();
                let a = self.vaddr(&regs, dst.index() as u32);
                let bw = self.body_idx(body.index());
                let b = self.trunc_to(I32, &bw);
                self.st_fn(&a, &b);
                let nb = self.next_blk(i);
                self.br(&nb);
            }
            Op::LoadEntry { dst, slot } => {
                // entry-frame absolute slot: tb = regs - base*VS; v = tb[slot]
                let regs = self.regs();
                let vs = self.iconst(self.lyt.val_size as i64);
                let base = self.env.base.clone();
                let boff = self.mul(&base, &vs);
                let ri = self.p2i(&regs);
                let t = self.sub(&ri, &boff);
                let tb = self.i2p(&t);
                let vp = self.gep(&tb, slot.index() as u32 * self.lyt.val_size as u32);
                let d = self.vaddr(&regs, dst.index() as u32);
                self.cpy_val(&d, &vp);
                let nb = self.next_blk(i);
                self.br(&nb);
            }
            Op::StoreEntry { slot, src } => {
                let vp = self.val_ptr(src.index() as u32);
                let regs = self.regs();
                let vs = self.iconst(self.lyt.val_size as i64);
                let base = self.env.base.clone();
                let boff = self.mul(&base, &vs);
                let ri = self.p2i(&regs);
                let t = self.sub(&ri, &boff);
                let tb = self.i2p(&t);
                let d = self.gep(&tb, slot.index() as u32 * self.lyt.val_size as u32);
                self.cpy_val(&d, &vp);
                let nb = self.next_blk(i);
                self.br(&nb);
            }
            Op::BoolEq { dst, left, right } => self.emit_bool(i, *dst, *left, *right, true),
            Op::BoolNe { dst, left, right } => self.emit_bool(i, *dst, *left, *right, false),
            Op::AddInt { dst, left, right } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.int_opnd(right.index() as u32);
                self.emit_ovf(i, *dst, "llvm.sadd.with.overflow", &a, &b);
            }
            Op::SubInt { dst, left, right } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.int_opnd(right.index() as u32);
                self.emit_ovf(i, *dst, "llvm.ssub.with.overflow", &a, &b);
            }
            Op::MultInt { dst, left, right } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.int_opnd(right.index() as u32);
                self.emit_ovf(i, *dst, "llvm.smul.with.overflow", &a, &b);
            }
            Op::ModInt { dst, left, right } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.int_opnd(right.index() as u32);
                self.emit_mod_int(i, *dst, &a, &b);
            }
            Op::IntLt { dst, left, right } => self.emit_eval_i(i, *dst, *left, *right, "SLT"),
            Op::IntLe { dst, left, right } => self.emit_eval_i(i, *dst, *left, *right, "SLE"),
            Op::IntGt { dst, left, right } => self.emit_eval_i(i, *dst, *left, *right, "SGT"),
            Op::IntGe { dst, left, right } => self.emit_eval_i(i, *dst, *left, *right, "SGE"),
            Op::IntEq { dst, left, right } => self.emit_eval_i(i, *dst, *left, *right, "EQ"),
            Op::IntNe { dst, left, right } => self.emit_eval_i(i, *dst, *left, *right, "NE"),
            Op::AddIntImm { dst, left, val } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.iconst(*val);
                self.emit_ovf(i, *dst, "llvm.sadd.with.overflow", &a, &b);
            }
            Op::SubIntImm { dst, left, val } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.iconst(*val);
                self.emit_ovf(i, *dst, "llvm.ssub.with.overflow", &a, &b);
            }
            Op::MultIntImm { dst, left, val } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.iconst(*val);
                self.emit_ovf(i, *dst, "llvm.smul.with.overflow", &a, &b);
            }
            Op::ModIntImm { dst, left, val } => {
                let a = self.int_opnd(left.index() as u32);
                let b = self.iconst(*val);
                self.emit_mod_int(i, *dst, &a, &b);
            }
            Op::IntLtImm { dst, left, val } => self.emit_eval_imm_i(i, *dst, *left, *val, "SLT"),
            Op::IntLeImm { dst, left, val } => self.emit_eval_imm_i(i, *dst, *left, *val, "SLE"),
            Op::IntGtImm { dst, left, val } => self.emit_eval_imm_i(i, *dst, *left, *val, "SGT"),
            Op::IntGeImm { dst, left, val } => self.emit_eval_imm_i(i, *dst, *left, *val, "SGE"),
            Op::IntEqImm { dst, left, val } => self.emit_eval_imm_i(i, *dst, *left, *val, "EQ"),
            Op::IntNeImm { dst, left, val } => self.emit_eval_imm_i(i, *dst, *left, *val, "NE"),
            Op::AddFloat { dst, left, right } => self.emit_farit(i, *dst, *left, *right, "fadd"),
            Op::SubFloat { dst, left, right } => self.emit_farit(i, *dst, *left, *right, "fsub"),
            Op::MultFloat { dst, left, right } => self.emit_farit(i, *dst, *left, *right, "fmul"),
            Op::DivFloat { dst, left, right } => self.emit_farit(i, *dst, *left, *right, "fdiv"),
            Op::FloatLt { dst, left, right } => self.emit_eval_f(i, *dst, *left, *right, "OLT"),
            Op::FloatLe { dst, left, right } => self.emit_eval_f(i, *dst, *left, *right, "OLE"),
            Op::FloatGt { dst, left, right } => self.emit_eval_f(i, *dst, *left, *right, "OGT"),
            Op::FloatGe { dst, left, right } => self.emit_eval_f(i, *dst, *left, *right, "OGE"),
            Op::FloatEq { dst, left, right } => self.emit_eval_f(i, *dst, *left, *right, "OEQ"),
            Op::FloatNe { dst, left, right } => self.emit_eval_f(i, *dst, *left, *right, "UNE"),
            Op::AddFloatImm { dst, left, val } => self.emit_farit_imm(i, *dst, *left, *val, "fadd"),
            Op::SubFloatImm { dst, left, val } => self.emit_farit_imm(i, *dst, *left, *val, "fsub"),
            Op::MultFloatImm { dst, left, val } => {
                self.emit_farit_imm(i, *dst, *left, *val, "fmul")
            }
            Op::ModFloatImm { dst, left, val } => self.emit_farit_imm(i, *dst, *left, *val, "frem"),
            Op::FloatLtImm { dst, left, val } => self.emit_eval_fimm(i, *dst, *left, *val, "OLT"),
            Op::FloatLeImm { dst, left, val } => self.emit_eval_fimm(i, *dst, *left, *val, "OLE"),
            Op::FloatGtImm { dst, left, val } => self.emit_eval_fimm(i, *dst, *left, *val, "OGT"),
            Op::FloatGeImm { dst, left, val } => self.emit_eval_fimm(i, *dst, *left, *val, "OGE"),
            Op::FloatEqImm { dst, left, val } => self.emit_eval_fimm(i, *dst, *left, *val, "OEQ"),
            Op::FloatNeImm { dst, left, val } => self.emit_eval_fimm(i, *dst, *left, *val, "UNE"),
            Op::BIntLt {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(i, target, *left, Some(*right), None, *is_true, "SLT"),
            Op::BIntLe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(i, target, *left, Some(*right), None, *is_true, "SLE"),
            Op::BIntGt {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(i, target, *left, Some(*right), None, *is_true, "SGT"),
            Op::BIntGe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(i, target, *left, Some(*right), None, *is_true, "SGE"),
            Op::BIntEq {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(i, target, *left, Some(*right), None, *is_true, "EQ"),
            Op::BIntNe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brr(i, target, *left, Some(*right), None, *is_true, "NE"),
            Op::BIntLtImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, "SLT"),
            Op::BIntLeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, "SLE"),
            Op::BIntGtImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, "SGT"),
            Op::BIntGeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, "SGE"),
            Op::BIntEqImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, "EQ"),
            Op::BIntNeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brr(i, target, *left, None, Some(*val), *is_true, "NE"),
            Op::BFloatLt {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(i, target, *left, Some(*right), None, *is_true, "OLT"),
            Op::BFloatLe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(i, target, *left, Some(*right), None, *is_true, "OLE"),
            Op::BFloatGt {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(i, target, *left, Some(*right), None, *is_true, "OGT"),
            Op::BFloatGe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(i, target, *left, Some(*right), None, *is_true, "OGE"),
            Op::BFloatEq {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(i, target, *left, Some(*right), None, *is_true, "OEQ"),
            Op::BFloatNe {
                target,
                left,
                right,
                is_true,
            } => self.emit_brf(i, target, *left, Some(*right), None, *is_true, "UNE"),
            Op::BFloatLtImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, "OLT"),
            Op::BFloatLeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, "OLE"),
            Op::BFloatGtImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, "OGT"),
            Op::BFloatGeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, "OGE"),
            Op::BFloatEqImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, "OEQ"),
            Op::BFloatNeImm {
                target,
                left,
                val,
                is_true,
            } => self.emit_brf(i, target, *left, None, Some(*val), *is_true, "UNE"),
            Op::ToFloat { dst, src } => {
                let iv = self.int_opnd(src.index() as u32);
                let f = self.ev(format!("llvm.sitofp {iv} to {F64}"));
                self.wr_float_dst(dst.index() as u32, &f);
                let nb = self.next_blk(i);
                self.br(&nb);
            }
            Op::Sqrt { dst, src } => {
                let f = self.float_opnd(src.index() as u32);
                let r = self.ev(format!(
                    "llvm.call_intrinsic @\"llvm.sqrt\" ({f}) : llvm.func <{F64} ({F64}) variadic = false>"
                ));
                self.wr_float_dst(dst.index() as u32, &r);
                let nb = self.next_blk(i);
                self.br(&nb);
            }
            // ---- helper-backed ops ----
            Op::NewArray { dst } => {
                let regs = self.regs();
                let d = self.iconst(dst.index() as i64);
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::NewArray,
                    &[
                        A::P(regs),
                        A::V(d),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                    ],
                );
            }
            Op::NewDict { dst } => {
                let regs = self.regs();
                let d = self.iconst(dst.index() as i64);
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::NewDict,
                    &[
                        A::P(regs),
                        A::V(d),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                    ],
                );
            }
            Op::NewInstance { dst, adt, fields } => {
                let fp = self.reg_list_slot(fields);
                let regs = self.regs();
                let d = self.iconst(dst.index() as i64);
                let a = self.iconst32(adt.index() as i64);
                let n = self.iconst(fields.len() as i64);
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::NewInstance,
                    &[
                        A::P(regs),
                        A::V(d),
                        A::I32(a),
                        A::P(fp),
                        A::V(n),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                    ],
                );
            }
            Op::NewClosure {
                dst,
                body,
                captures,
            } => {
                let cp = self.reg_list_slot(captures);
                let regs = self.regs();
                let d = self.iconst(dst.index() as i64);
                let bw = self.body_idx(body.index());
                let b = self.trunc_to(I32, &bw);
                let n = self.iconst(captures.len() as i64);
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::NewClosure,
                    &[
                        A::P(regs),
                        A::V(d),
                        A::I32(b),
                        A::P(cp),
                        A::V(n),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                    ],
                );
            }
            Op::Push { array, value } => {
                let regs = self.regs();
                let a = self.iconst(array.index() as i64);
                let v = self.iconst(value.index() as i64);
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::Push,
                    &[
                        A::P(regs),
                        A::V(a),
                        A::V(v),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                    ],
                );
            }
            Op::Insert { dict, key, value } => {
                let regs = self.regs();
                let d = self.iconst(dict.index() as i64);
                let k = self.iconst32(key.index() as i64);
                let v = self.iconst(value.index() as i64);
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::Insert,
                    &[
                        A::P(regs),
                        A::V(d),
                        A::I32(k),
                        A::V(v),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                        A::P(self.env.strs.clone()),
                    ],
                );
            }
            Op::SetIndex { set, index, value } => {
                let regs = self.regs();
                let s = self.iconst(set.index() as i64);
                let ii = self.iconst(index.index() as i64);
                let v = self.iconst(value.index() as i64);
                self.helper_op(
                    i,
                    off,
                    next,
                    H::SetIndex,
                    &[
                        A::P(regs),
                        A::V(s),
                        A::V(ii),
                        A::V(v),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                        A::P(self.env.out.clone()),
                    ],
                );
            }
            Op::GetIndex {
                dst,
                set,
                index,
                kind,
            } => self.emit_get_index(i, off, next, *dst, *set, *index, *kind),
            Op::GetField {
                dst,
                src,
                slot,
                kind,
            } => self.emit_get_field(i, off, next, *dst, *src, *slot, *kind),
            Op::SetField {
                receiver,
                slot,
                value,
            } => {
                let regs = self.regs();
                let r = self.iconst(receiver.index() as i64);
                let sl = self.iconst(*slot as i64);
                let v = self.iconst(value.index() as i64);
                self.helper_op(
                    i,
                    off,
                    next,
                    H::SetField,
                    &[
                        A::P(regs),
                        A::V(r),
                        A::V(sl),
                        A::V(v),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                        A::P(self.env.out.clone()),
                    ],
                );
            }
            Op::In {
                dst,
                needle,
                haystack,
                condition,
            } => {
                let regs = self.regs();
                let d = self.iconst(dst.index() as i64);
                let n = self.iconst(needle.index() as i64);
                let h = self.iconst(haystack.index() as i64);
                let c = self.iconst8(*condition as i64);
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::ContainsOp,
                    &[A::P(regs), A::V(d), A::V(n), A::V(h), A::I8(c)],
                );
            }
            Op::IsInstance { dst, src, adt } => {
                let regs = self.regs();
                let d = self.iconst(dst.index() as i64);
                let s = self.iconst(src.index() as i64);
                let a = self.iconst32(adt.index() as i64);
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::IsInstance,
                    &[A::P(regs), A::V(d), A::V(s), A::I32(a)],
                );
            }
            Op::IsRaised { dst, src } => {
                let regs = self.regs();
                let d = self.iconst(dst.index() as i64);
                let s = self.iconst(src.index() as i64);
                self.helper_op_v(i, off, next, H::IsRaised, &[A::P(regs), A::V(d), A::V(s)]);
            }
            Op::UnwrapRaised { dst, src } => {
                let regs = self.regs();
                let d = self.iconst(dst.index() as i64);
                let s = self.iconst(src.index() as i64);
                self.helper_op_v(
                    i,
                    off,
                    next,
                    H::UnwrapRaised,
                    &[A::P(regs), A::V(d), A::V(s)],
                );
            }
            Op::Unwrap { dst, src } => {
                let regs = self.regs();
                let d = self.iconst(dst.index() as i64);
                let s = self.iconst(src.index() as i64);
                self.helper_op(
                    i,
                    off,
                    next,
                    H::Unwrap,
                    &[
                        A::P(regs),
                        A::V(d),
                        A::V(s),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                        A::P(self.env.out.clone()),
                    ],
                );
            }
            Op::UnwrapUnit { dst, src } => {
                let regs = self.regs();
                let d = self.iconst(dst.index() as i64);
                let s = self.iconst(src.index() as i64);
                self.helper_op(
                    i,
                    off,
                    next,
                    H::UnwrapUnit,
                    &[
                        A::P(regs),
                        A::V(d),
                        A::V(s),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                        A::P(self.env.out.clone()),
                    ],
                );
            }
            Op::Len { dst, src } => {
                self.flush_seq();
                self.mark_op(off, next);
                let regs = self.regs();
                let s = self.iconst(src.index() as i64);
                let slot = self.slot_i.clone();
                let k = self
                    .hcall(
                        H::Len,
                        &[
                            A::P(regs),
                            A::V(s),
                            A::P(slot.clone()),
                            A::P(self.env.out.clone()),
                        ],
                    )
                    .unwrap();
                let k1 = self.icmp_ne0(I8, &k);
                let w = self.create_block();
                let er = self.ex.eret.clone();
                self.cbr(&k1, &er, &w);
                self.start_block(&w);
                let v = self.ld(I64, &slot, 0, 8);
                self.wr_int_dst(dst.index() as u32, &v);
                let nb = self.next_blk(i);
                self.br(&nb);
            }
            Op::Bin {
                dst,
                left,
                op,
                right,
            } => {
                let regs = self.regs();
                let d = self.iconst(dst.index() as i64);
                let l = self.iconst(left.index() as i64);
                let o = self.iconst8(*op as i64);
                let r = self.iconst(right.index() as i64);
                self.helper_op(
                    i,
                    off,
                    next,
                    H::Bin,
                    &[
                        A::P(regs),
                        A::V(d),
                        A::V(l),
                        A::I8(o),
                        A::V(r),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                        A::P(self.env.out.clone()),
                    ],
                );
            }
            Op::Unary { dst, op, src } => {
                let regs = self.regs();
                let d = self.iconst(dst.index() as i64);
                let o = self.iconst8(*op as i64);
                let s = self.iconst(src.index() as i64);
                self.helper_op(
                    i,
                    off,
                    next,
                    H::Unary,
                    &[
                        A::P(regs),
                        A::V(d),
                        A::I8(o),
                        A::V(s),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                        A::P(self.env.out.clone()),
                    ],
                );
            }
            Op::StrEq { dst, left, right } => {
                self.emit_str_eval(i, off, next, *dst, *left, *right, true)
            }
            Op::StrNe { dst, left, right } => {
                self.emit_str_eval(i, off, next, *dst, *left, *right, false)
            }
            Op::CallNative { dst, id, args } => {
                let ap = self.reg_list_slot(args);
                let regs = self.regs();
                let d = self.iconst(dst.index() as i64);
                let nid = self.iconst32(id.index() as i64);
                let n = self.iconst(args.len() as i64);
                self.helper_op(
                    i,
                    off,
                    next,
                    H::CallNative,
                    &[
                        A::P(self.env.thread.clone()),
                        A::P(regs),
                        A::V(d),
                        A::I32(nid),
                        A::P(ap),
                        A::V(n),
                        A::P(self.env.code.clone()),
                        A::P(self.env.ctx0.clone()),
                        A::P(self.env.ctx1.clone()),
                        A::P(self.env.out.clone()),
                    ],
                );
            }
            Op::CallDirect { dst, body, args } => {
                self.emit_call_direct(i, off, next, *dst, *body, args);
            }
            Op::Call { dst, callee, args } => {
                self.emit_call(i, off, next, *dst, *callee, args);
            }
        }
    }

    /// `Op::GetIndex` read fast path: `regs[set]` a `Val::Array` and
    /// `regs[index]` a `Val::Int` reads the `Vec<Val>` element through the
    /// `RefLock` inline (the borrow flag is only read, never held — a
    /// nonzero flag means a borrow survived an op boundary, impossible —
    /// and `estep` preserves the panic the interpreter's `borrow()` would
    /// raise). A tag miss, an `IntArray`/`FloatArray`/`Dict`/`Str`
    /// receiver, or an out-of-range index falls to `mj_get_index`, whose
    /// `pos()` bounds error needs no inline copy.
    fn emit_get_index(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        dst: Reg,
        set: Reg,
        index: Reg,
        kind: AccessKind,
    ) {
        let sp = self.val_ptr(set.index() as u32);
        let ip = self.val_ptr(index.index() as u32);
        let ty = self.tag_ty();
        let es = self.ex.estep.clone();

        // `Option` access on Null short-circuits before any tag dispatch.
        if kind == AccessKind::Option {
            let (b_null, b_nn) = (self.create_block(), self.create_block());
            let tn = self.tconst(self.lyt.t_null);
            let ts0 = self.ld_tag(&sp);
            let isn = self.icmp(ty, "EQ", &ts0, &tn);
            self.cbr(&isn, &b_null, &b_nn);
            self.start_block(&b_null);
            let regs = self.regs();
            let d = self.vaddr(&regs, dst.index() as u32);
            self.st_null(&d);
            self.refresh_shadow(dst.index() as u32);
            let nb = self.next_blk(i);
            self.br(&nb);
            self.start_block(&b_nn);
        }

        let ts = self.ld_tag(&sp);
        let ta = self.tconst(self.lyt.t_array);
        let isa = self.icmp(ty, "EQ", &ts, &ta);
        let ti = self.ld_tag(&ip);
        let tint = self.tconst(self.lyt.t_int);
        let isi = self.icmp(ty, "EQ", &ti, &tint);
        let both = self.ev(format!("llvm.and {isa}, {isi} : {I1}"));
        let (b_fast, b_slow) = (self.create_block(), self.create_block());
        self.cbr(&both, &b_fast, &b_slow);

        // the helper covers every other receiver/index shape, and the
        // `pos()` out-of-bounds `RtErr` (an error value, not a panic).
        self.start_block(&b_slow);
        let regs = self.regs();
        let d = self.iconst(dst.index() as i64);
        let s = self.iconst(set.index() as i64);
        let ii = self.iconst(index.index() as i64);
        let kk = self.iconst8(kind as i64);
        self.helper_op_read(
            i,
            off,
            next,
            H::GetIndex,
            &[
                A::P(regs),
                A::V(d),
                A::V(s),
                A::V(ii),
                A::I8(kk),
                A::P(self.env.ctx0.clone()),
                A::P(self.env.ctx1.clone()),
                A::P(self.env.out.clone()),
            ],
            dst,
        );

        self.start_block(&b_fast);
        let gc = self.ld(PTR, &sp, self.lyt.arr_pay as u32, 8);
        let fl = self.ld_flag(&gc, self.lyt.rl_flag as u32);
        let z = self.iconst(0);
        let neg = self.icmp(I64, "SLT", &fl, &z);
        let b_ok = self.create_block();
        self.cbr(&neg, &es, &b_ok);
        self.start_block(&b_ok);
        let vec = self.gep(&gc, self.lyt.rl_vec as u32);
        let len = self.ld_word(&vec, self.lyt.vec_len as u32);
        let iv = self.ld(I64, &ip, self.lyt.val_pay as u32, 8);
        // `usize::try_from(i)` folded into one compare: a negative index is
        // unsigned-huge, so ULT fails exactly when `pos()` would error.
        let inb = self.icmp(I64, "ULT", &iv, &len);
        let b_hit = self.create_block();
        self.cbr(&inb, &b_hit, &b_slow);
        self.start_block(&b_hit);
        let base = self.ld(PTR, &vec, self.lyt.vec_ptr as u32, 8);
        let vs = self.iconst(self.lyt.val_size as i64);
        let eoff = self.mul(&iv, &vs);
        let elem = self.gep_dyn(&base, &eoff);
        self.copy_to_dst(i, dst, &elem);
    }

    /// `Op::GetField` read fast paths — `Instance` fields (Inline `data`
    /// and `Spilled` vec) and `Array` vec indexing read through the
    /// `RefLock` inline under the same borrow-flag contract as
    /// `emit_get_index`. Typed arrays (`at()` boxes the element), `Dict`,
    /// a `Direct`-kind `Null`, and any out-of-bounds slot take
    /// `mj_get_field`; a held write lock takes `estep` for panic parity,
    /// and `Option` on `Null` answers `Null` inline.
    fn emit_get_field(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        dst: Reg,
        src: Reg,
        slot: u32,
        kind: AccessKind,
    ) {
        let sp = self.val_ptr(src.index() as u32);
        let t = self.ld_tag(&sp);
        let ty = self.tag_ty();
        let es = self.ex.estep.clone();
        let sc = self.iconst(slot as i64);

        if kind == AccessKind::Option {
            let (b_null, b_nn) = (self.create_block(), self.create_block());
            let tn = self.tconst(self.lyt.t_null);
            let isn = self.icmp(ty, "EQ", &t, &tn);
            self.cbr(&isn, &b_null, &b_nn);
            self.start_block(&b_null);
            let regs = self.regs();
            let d = self.vaddr(&regs, dst.index() as u32);
            self.st_null(&d);
            self.refresh_shadow(dst.index() as u32);
            let nb = self.next_blk(i);
            self.br(&nb);
            self.start_block(&b_nn);
        }

        let (b_inst, b_arrq, b_slow) =
            (self.create_block(), self.create_block(), self.create_block());
        let ti = self.tconst(self.lyt.t_instance);
        let isi = self.icmp(ty, "EQ", &t, &ti);
        self.cbr(&isi, &b_inst, &b_arrq);

        self.start_block(&b_arrq);
        let ta = self.tconst(self.lyt.t_array);
        let isa = self.icmp(ty, "EQ", &t, &ta);
        let b_arr = self.create_block();
        self.cbr(&isa, &b_arr, &b_slow);

        self.start_block(&b_slow);
        let regs = self.regs();
        let d = self.iconst(dst.index() as i64);
        let s = self.iconst(src.index() as i64);
        let sl = self.iconst(slot as i64);
        let kk = self.iconst8(kind as i64);
        self.helper_op_read(
            i,
            off,
            next,
            H::GetField,
            &[
                A::P(regs),
                A::V(d),
                A::V(s),
                A::V(sl),
                A::I8(kk),
                A::P(self.env.out.clone()),
            ],
            dst,
        );

        // Instance: Gc -> RefLock flag -> InstanceData.fields
        self.start_block(&b_inst);
        let gc = self.ld(PTR, &sp, self.lyt.inst_pay as u32, 8);
        let fl = self.ld_flag(&gc, self.lyt.rl_flag_i as u32);
        let z = self.iconst(0);
        let neg = self.icmp(I64, "SLT", &fl, &z);
        let b_fd = self.create_block();
        self.cbr(&neg, &es, &b_fd);
        self.start_block(&b_fd);
        let fp = self.gep(&gc, (self.lyt.rl_inst + self.lyt.id_fields) as u32);
        let fty = self.wty(self.lyt.fld_tsz);
        let ft = self.ld(fty, &fp, self.lyt.fld_tag as u32, self.lyt.fld_tsz as u32);
        let fin = self.iconst_ty(fty, self.lyt.fld_inline as i64);
        let isinl = self.icmp(fty, "EQ", &ft, &fin);
        let (b_inl, b_spl) = (self.create_block(), self.create_block());
        self.cbr(&isinl, &b_inl, &b_spl);
        // Fields::Inline { len: u8, data: [Val; INLINE_FIELDS] }
        self.start_block(&b_inl);
        let len = self.ld(I8, &fp, self.lyt.fld_len as u32, 1);
        let lenz = self.zext64(&len);
        let inb = self.icmp(I64, "ULT", &sc, &lenz);
        let b_ih = self.create_block();
        self.cbr(&inb, &b_ih, &es);
        self.start_block(&b_ih);
        let data = self.gep(&fp, self.lyt.fld_data as u32);
        let el = self.gep(&data, slot * self.lyt.val_size as u32);
        self.copy_to_dst(i, dst, &el);
        // Fields::Spilled(Vec<Val>)
        self.start_block(&b_spl);
        let vl = self.ld_word(&fp, (self.lyt.fld_spilled + self.lyt.vec_len) as u32);
        let inb2 = self.icmp(I64, "ULT", &sc, &vl);
        let b_sh = self.create_block();
        self.cbr(&inb2, &b_sh, &es);
        self.start_block(&b_sh);
        let vb = self.ld(
            PTR,
            &fp,
            (self.lyt.fld_spilled + self.lyt.vec_ptr) as u32,
            8,
        );
        let el2 = self.gep(&vb, slot * self.lyt.val_size as u32);
        self.copy_to_dst(i, dst, &el2);

        // Array receiver: `a.0.borrow()[slot]` — plain Vec<Val> index.
        self.start_block(&b_arr);
        let gca = self.ld(PTR, &sp, self.lyt.arr_pay as u32, 8);
        let fla = self.ld_flag(&gca, self.lyt.rl_flag as u32);
        let nega = self.icmp(I64, "SLT", &fla, &z);
        let b_ao = self.create_block();
        self.cbr(&nega, &es, &b_ao);
        self.start_block(&b_ao);
        let vec = self.gep(&gca, self.lyt.rl_vec as u32);
        let al = self.ld_word(&vec, self.lyt.vec_len as u32);
        let ainb = self.icmp(I64, "ULT", &sc, &al);
        let b_ah = self.create_block();
        self.cbr(&ainb, &b_ah, &es);
        self.start_block(&b_ah);
        let vb2 = self.ld(PTR, &vec, self.lyt.vec_ptr as u32, 8);
        let el3 = self.gep(&vb2, slot * self.lyt.val_size as u32);
        self.copy_to_dst(i, dst, &el3);
    }

    /// `Op::Switch` — the interpreter's `code.ip` jump table as a real
    /// `llvm.switch` per scrutinee arm: `Val::Int` subtracts `base` in i64
    /// (the interpreter's `0 <= idx < len` is exactly "names a case"),
    /// `Val::Instance` subtracts it from `struct_id` `zext`ed (a `sid <
    /// base` negative is no case — the same default the u32 wrap landing
    /// past the table end takes). Any other tag is `default` outright.
    fn emit_switch(&mut self, scrut: Reg, base: u32, default: &BlockTarget, table: &[BlockTarget]) {
        let sp = self.val_ptr(scrut.index() as u32);
        let t = self.ld_tag(&sp);
        let ty = self.tag_ty();
        let es = self.ex.estep.clone();
        let dblk = self.tgt_blk(default);
        let b64 = self.iconst(base as i64);

        let (b_int, b_chk) = (self.create_block(), self.create_block());
        let ti = self.tconst(self.lyt.t_int);
        let isi = self.icmp(ty, "EQ", &t, &ti);
        self.cbr(&isi, &b_int, &b_chk);
        self.start_block(&b_chk);
        let tins = self.tconst(self.lyt.t_instance);
        let isn = self.icmp(ty, "EQ", &t, &tins);
        let b_inst = self.create_block();
        self.cbr(&isn, &b_inst, &dblk);

        self.start_block(&b_int);
        let v = self.ld(I64, &sp, self.lyt.val_pay as u32, 8);
        let idx = self.sub(&v, &b64);
        self.emit_switch_tbl(&idx, &dblk, table);

        self.start_block(&b_inst);
        let gc = self.ld(PTR, &sp, self.lyt.inst_pay as u32, 8);
        let fl = self.ld_flag(&gc, self.lyt.rl_flag_i as u32);
        let z = self.iconst(0);
        let neg = self.icmp(I64, "SLT", &fl, &z);
        let b_ok = self.create_block();
        self.cbr(&neg, &es, &b_ok);
        self.start_block(&b_ok);
        let sidp = self.gep(&gc, (self.lyt.rl_inst + self.lyt.id_sid) as u32);
        let sid32 = self.ld(I32, &sidp, 0, 4);
        let sid = self.zext64(&sid32);
        let idx2 = self.sub(&sid, &b64);
        self.emit_switch_tbl(&idx2, &dblk, table);
    }

    /// A masked `is_bool` answer: a cond reg with a live scalar shadow holds
    /// an `Int`/`Float` — definitely not `Val::Bool(is_true)` — compare-false.
    fn mask_by_shadow(&mut self, r: Reg, k: &str) -> String {
        let idx = r.index() as u32;
        let ok_a = self
            .int
            .get(&idx)
            .map(|(_, o)| o.clone())
            .or_else(|| self.float.get(&idx).map(|(_, o)| o.clone()));
        match ok_a {
            None => k.to_string(),
            Some(ok_a) => {
                let okv = self.ld(I8, &ok_a, 0, 1);
                let z8 = self.iconst8(0);
                let not = self.icmp(I8, "EQ", &okv, &z8);
                self.ev(format!("llvm.and {k}, {not} : {I1}"))
            }
        }
    }

    /// `let Some(v) = a.checked_add/sub/mul(b)` — on overflow,
    /// `RtErr::IntegerOverflow` with `*op_ip = off` (the `eerr` block stores
    /// both from `cur_ip`).
    fn emit_ovf(&mut self, i: usize, dst: Reg, name: &str, a: &str, b: &str) {
        let s = self.ev(format!(
            "llvm.call_intrinsic @\"{name}\" ({a}, {b}) : llvm.func <llvm.struct <{{ {I64}, {I1} }} : Unpacked> ({I64}, {I64}) variadic = false>"
        ));
        let v = self.ev(format!("llvm.extract_value {s} [0] : {I64}"));
        let of = self.ev(format!("llvm.extract_value {s} [1] : {I1}"));
        let okb = self.create_block();
        let t = self.err_tramp();
        self.cbr(&of, &t, &okb);
        self.start_block(&okb);
        self.wr_int_dst(dst.index() as u32, &v);
        let nb = self.next_blk(i);
        self.br(&nb);
        self.fill_err(&t, ERR_OVFW);
    }

    fn emit_bool(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, eq: bool) {
        // `let Val::Bool(l) = ...` twice — a miss is the interpreter's
        // `unreachable!`; `estep` reproduces it (panic inside `step`).
        let l = self.bool_opnd(left);
        let r = self.bool_opnd(right);
        let cc = if eq { "EQ" } else { "NE" };
        let c = self.icmp(I8, cc, &l, &r);
        let c8 = self.i1_to_i8(&c);
        self.wr_bool_dst(dst.index() as u32, &c8);
        let nb = self.next_blk(i);
        self.br(&nb);
    }

    fn bool_opnd(&mut self, r: Reg) -> String {
        let idx = r.index() as u32;
        if let Some(ok_a) = self
            .int
            .get(&idx)
            .map(|(_, o)| o.clone())
            .or_else(|| self.float.get(&idx).map(|(_, o)| o.clone()))
        {
            // a live scalar shadow means the reg is NOT a Bool → estep
            let okv = self.ld(I8, &ok_a, 0, 1);
            let z8 = self.iconst8(0);
            let notok = self.icmp(I8, "EQ", &okv, &z8);
            let nb = self.create_block();
            let es = self.ex.estep.clone();
            self.cbr(&notok, &nb, &es);
            self.start_block(&nb);
        }
        let good = self.create_block();
        let regs = self.regs();
        let a = self.vaddr(&regs, idx);
        let t = self.ld_tag(&a);
        let want = self.tconst(self.lyt.t_bool);
        let ty = self.tag_ty();
        let ok = self.icmp(ty, "EQ", &t, &want);
        let es = self.ex.estep.clone();
        self.cbr(&ok, &good, &es);
        self.start_block(&good);
        self.ld(I8, &a, self.lyt.bool_pay as u32, 1)
    }

    fn emit_mod_int(&mut self, i: usize, dst: Reg, a: &str, b: &str) {
        // b == 0 → Err(ModByZero); i64::MIN % -1 panics in the interpreter —
        // route that pair to `estep` so `step` hits the identical panic.
        let z = self.iconst(0);
        let bz = self.icmp(I64, "EQ", b, &z);
        let run = self.create_block();
        let tramp = self.err_tramp();
        self.cbr(&bz, &tramp, &run);
        self.start_block(&run);
        let c_min = self.iconst(i64::MIN);
        let c_m1 = self.iconst(-1);
        let amin = self.icmp(I64, "EQ", a, &c_min);
        let bm1 = self.icmp(I64, "EQ", b, &c_m1);
        let bad = self.ev(format!("llvm.and {amin}, {bm1} : {I1}"));
        let run2 = self.create_block();
        let es = self.ex.estep.clone();
        self.cbr(&bad, &es, &run2);
        self.start_block(&run2);
        let v = self.ev(format!("llvm.srem {a}, {b} : {I64}"));
        self.wr_int_dst(dst.index() as u32, &v);
        let nb = self.next_blk(i);
        self.br(&nb);
        self.fill_err(&tramp, ERR_MOD0);
    }

    fn emit_eval_i(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, cc: &str) {
        let a = self.int_opnd(left.index() as u32);
        let b = self.int_opnd(right.index() as u32);
        let c = self.icmp(I64, cc, &a, &b);
        let c8 = self.i1_to_i8(&c);
        self.wr_bool_dst(dst.index() as u32, &c8);
        let nb = self.next_blk(i);
        self.br(&nb);
    }

    fn emit_eval_imm_i(&mut self, i: usize, dst: Reg, left: Reg, val: i64, cc: &str) {
        let a = self.int_opnd(left.index() as u32);
        let b = self.iconst(val);
        let c = self.icmp(I64, cc, &a, &b);
        let c8 = self.i1_to_i8(&c);
        self.wr_bool_dst(dst.index() as u32, &c8);
        let nb = self.next_blk(i);
        self.br(&nb);
    }

    fn emit_eval_f(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, cc: &str) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.float_opnd(right.index() as u32);
        let c = self.fcmp(cc, &a, &b);
        let c8 = self.i1_to_i8(&c);
        self.wr_bool_dst(dst.index() as u32, &c8);
        let nb = self.next_blk(i);
        self.br(&nb);
    }

    fn emit_eval_fimm(&mut self, i: usize, dst: Reg, left: Reg, val: i64, cc: &str) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.f64const(f64::from_bits(val as u64));
        let c = self.fcmp(cc, &a, &b);
        let c8 = self.i1_to_i8(&c);
        self.wr_bool_dst(dst.index() as u32, &c8);
        let nb = self.next_blk(i);
        self.br(&nb);
    }

    fn emit_farit(&mut self, i: usize, dst: Reg, left: Reg, right: Reg, o: &str) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.float_opnd(right.index() as u32);
        let v = self.ev(format!("llvm.{o} {a}, {b} : {F64}"));
        self.wr_float_dst(dst.index() as u32, &v);
        let nb = self.next_blk(i);
        self.br(&nb);
    }

    fn emit_farit_imm(&mut self, i: usize, dst: Reg, left: Reg, val: i64, o: &str) {
        let a = self.float_opnd(left.index() as u32);
        let b = self.f64const(f64::from_bits(val as u64));
        let v = self.ev(format!("llvm.{o} {a}, {b} : {F64}"));
        self.wr_float_dst(dst.index() as u32, &v);
        let nb = self.next_blk(i);
        self.br(&nb);
    }

    /// `StrEq`/`StrNe`: the (Str, Str) fast path via `bin_str`; anything
    /// else runs the op through `step` — the interpreter's own behavior.
    #[allow(clippy::too_many_arguments)]
    fn emit_str_eval(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        dst: Reg,
        left: Reg,
        right: Reg,
        eq: bool,
    ) {
        self.flush_seq();
        self.mark_op(off, next);
        let regs = self.regs();
        let d = self.iconst(dst.index() as i64);
        let l = self.iconst(left.index() as i64);
        let r = self.iconst(right.index() as i64);
        let e = self.iconst8(eq as i64);
        // 0 → wrote regs[dst]; 1 → both-Str fast path missed → `step`
        let k = self
            .hcall(
                H::BinStr,
                &[A::P(regs), A::V(d), A::V(l), A::V(r), A::I8(e)],
            )
            .unwrap();
        let k1 = self.icmp_ne0(I8, &k);
        let nb = self.next_blk(i);
        let es = self.ex.estep.clone();
        self.cbr(&k1, &es, &nb);
    }

    /// `B*` register/imm branch ops: `hit = a cc b`, then
    /// `code.ip = if hit == is_true { t } else { next }`.
    #[allow(clippy::too_many_arguments)]
    fn emit_brr(
        &mut self,
        i: usize,
        target: &BlockTarget,
        left: Reg,
        right: Option<Reg>,
        imm: Option<i64>,
        is_true: bool,
        cc: &str,
    ) {
        let a = self.int_opnd(left.index() as u32);
        let b = match right {
            Some(r) => self.int_opnd(r.index() as u32),
            None => self.iconst(imm.unwrap()),
        };
        let hit = self.icmp(I64, cc, &a, &b);
        let (tb, nb) = (self.tgt_blk(target), self.next_blk(i));
        let (t, f) = if is_true { (tb, nb) } else { (nb, tb) };
        self.cbr(&hit, &t, &f);
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_brf(
        &mut self,
        i: usize,
        target: &BlockTarget,
        left: Reg,
        right: Option<Reg>,
        imm: Option<i64>,
        is_true: bool,
        cc: &str,
    ) {
        let a = self.float_opnd(left.index() as u32);
        let b = match right {
            Some(r) => self.float_opnd(r.index() as u32),
            None => self.f64const(f64::from_bits(imm.unwrap() as u64)),
        };
        let hit = self.fcmp(cc, &a, &b);
        let (tb, nb) = (self.tgt_blk(target), self.next_blk(i));
        let (t, f) = if is_true { (tb, nb) } else { (nb, tb) };
        self.cbr(&hit, &t, &f);
    }

    /// `Move` — bcgen's flag-propagating copy for same-kind shadow pairs,
    /// the `wr_val` shape for mixed, a plain copy otherwise.
    fn emit_move(&mut self, dst: Reg, src: Reg) {
        let d = dst.index() as u32;
        let s = src.index() as u32;
        let di = self.int.get(&d).cloned();
        let df = self.float.get(&d).cloned();
        let si = self.int.get(&s).cloned();
        let sf = self.float.get(&s).cloned();
        if let (Some((dsv, dok)), Some((ssv, sok))) = (di.clone(), si.clone()) {
            // `d = s; dok = sok; if !dok { wr(d, rd(s)) }`
            let v = self.ld(I64, &ssv, 0, 8);
            let k = self.ld(I8, &sok, 0, 1);
            self.st(&dsv, &v);
            self.st_al(&dok, &k, 1);
            let c = self.create_block();
            let w = self.create_block();
            let dokv = self.ld(I8, &dok, 0, 1);
            let dok1 = self.icmp_ne0(I8, &dokv);
            self.cbr(&dok1, &c, &w);
            self.start_block(&w);
            let regs = self.regs();
            let vp = self.vaddr(&regs, s);
            let dd = self.vaddr(&regs, d);
            self.cpy_val(&dd, &vp);
            self.br(&c);
            self.start_block(&c);
        } else if let (Some((dsv, dok)), Some((ssv, sok))) = (df.clone(), sf.clone()) {
            let v = self.ldf(&ssv);
            let k = self.ld(I8, &sok, 0, 1);
            self.st(&dsv, &v);
            self.st_al(&dok, &k, 1);
            let c = self.create_block();
            let w = self.create_block();
            let dokv = self.ld(I8, &dok, 0, 1);
            let dok1 = self.icmp_ne0(I8, &dokv);
            self.cbr(&dok1, &c, &w);
            self.start_block(&w);
            let regs = self.regs();
            let vp = self.vaddr(&regs, s);
            let dd = self.vaddr(&regs, d);
            self.cpy_val(&dd, &vp);
            self.br(&c);
            self.start_block(&c);
        } else if let Some(dv) = di {
            // dst int-shadowed, src not int: `match v { Int(x) => shadow, v
            // => { ok=0; wr(d,v) } }` — bcgen's `wr_val` arm.
            self.emit_move_wrv(d, s, dv, sf, true);
        } else if let Some(dv) = df {
            self.emit_move_wrv(d, s, dv, si, false);
        } else {
            // unshadowed dst: materialize a live shadow src into regs first
            let vp = self.val_ptr(s);
            let regs = self.regs();
            let dd = self.vaddr(&regs, d);
            self.cpy_val(&dd, &vp);
        }
    }

    /// dst scalar-shadowed / src not same-kind-shadowed: probe the
    /// (possibly opposite-kind-shadowed) src for the right tag; hit →
    /// shadow write, miss → `ok = 0; wr(d, regs[s])`.
    fn emit_move_wrv(
        &mut self,
        d: u32,
        s: u32,
        dv: (String, String),
        osrc: Option<(String, String)>,
        dst_is_int: bool,
    ) {
        let (dsv, dok) = dv;
        let miss = self.create_block();
        let good = self.create_block();
        let done = self.create_block();
        if let Some((ssv, sok)) = osrc {
            // src carries a live opposite-kind shadow → the Val is that
            // scalar → miss arm writes it through directly.
            let okv = self.ld(I8, &sok, 0, 1);
            let ok1 = self.icmp_ne0(I8, &okv);
            let genb = self.create_block();
            let fmiss = self.create_block();
            self.cbr(&ok1, &fmiss, &genb);
            self.start_block(&fmiss);
            let v = if dst_is_int {
                self.ldf(&ssv)
            } else {
                self.ld(I64, &ssv, 0, 8)
            };
            let regs = self.regs();
            let dd = self.vaddr(&regs, d);
            if dst_is_int {
                self.st_float(&dd, &v);
            } else {
                self.st_int(&dd, &v);
            }
            let z = self.iconst8(0);
            self.st_al(&dok, &z, 1);
            self.br(&done);
            self.start_block(&genb);
        }
        // generic: probe the authoritative slot for the wanted tag
        let want = if dst_is_int {
            self.lyt.t_int
        } else {
            self.lyt.t_float
        };
        let regs = self.regs();
        let sa = self.vaddr(&regs, s);
        let t = self.ld_tag(&sa);
        let w8 = self.tconst(want);
        let ty = self.tag_ty();
        let k = self.icmp(ty, "EQ", &t, &w8);
        self.cbr(&k, &good, &miss);
        self.start_block(&good);
        let v = self.ld(
            if dst_is_int { I64 } else { F64 },
            &sa,
            self.lyt.val_pay as u32,
            8,
        );
        self.st(&dsv, &v);
        let one = self.iconst8(1);
        self.st_al(&dok, &one, 1);
        self.br(&done);
        self.start_block(&miss);
        let regs = self.regs();
        let vp = self.vaddr(&regs, s);
        let dd = self.vaddr(&regs, d);
        self.cpy_val(&dd, &vp);
        let z = self.iconst8(0);
        self.st_al(&dok, &z, 1);
        self.br(&done);
        self.start_block(&done);
    }

    /// `CallDirect` — one `mj_call_body` hop: the megashim enters the callee
    /// frame (Flow::Call at the depth cap / arity check), runs the callee
    /// body through the same bodies table, and pops on `Flow::Return`. The
    /// jit tier's inline frame-push fast path is ported below; this arm is
    /// the slow half of its gate.
    fn emit_call_direct(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        dst: Reg,
        body: compile::BodyId,
        args: &[Reg],
    ) {
        let cchunk = &self.prog.chunks[body];
        // Compile-time gates — see jit::emit_call_direct: a static arity
        // miss and a capturing callee keep the megashim; large frames skip
        // the unrolled Null fill.
        if std::env::var("PLGEN_NO_INLINE_CALL").is_err()
            && args.len() == cchunk.args as usize
            && cchunk.captures.is_empty()
            && (cchunk.regs as usize) <= 64
        {
            return self.emit_call_direct_inline(i, off, next, dst, body, args, cchunk);
        }
        let ap = self.reg_list_slot(args);
        let nargs = args.len();
        self.flush_seq();
        // `code.ip = next; *op_ip = off` before the call (the shim's frame
        // push saves `code.ip` as the caller resume slot; errors locate via
        // op_ip).
        self.mark_op(off, next);
        // the callee draws fuel/ops_left through its own `bcn` — charge our
        // spent ops first so its reads are exact
        self.settle_seq();
        let b = self.body_idx(body.index());
        let d = self.iconst(dst.index() as i64);
        let n = self.iconst(nargs as i64);
        let rp = self
            .hcall(
                H::CallBody,
                &[
                    A::P(self.env.env.clone()),
                    A::P(self.env.bodies_tbl.clone()),
                    A::V(b),
                    A::V(d),
                    A::P(ap),
                    A::V(n),
                ],
            )
            .unwrap();
        self.post_call(i, dst, &rp);
    }

    /// The inline `CallDirect` fast path — `jit/src/emit.rs`
    /// `emit_call_direct_inline` ported op for op: under the runtime gates
    /// (depth cap, `regs`/`frames` capacity) the callee frame is pushed
    /// inline over the probed layout, `mimas_body_N` runs as a direct
    /// `llvm.call`, and the `Flow::Return` tail is popped inline.
    /// Anything slower — capacity growth, the depth cap (which must
    /// produce `Flow::Call`) — routes to the `mj_call_body` megashim.
    #[allow(clippy::too_many_arguments)]
    fn emit_call_direct_inline(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        dst: Reg,
        body: compile::BodyId,
        args: &[Reg],
        cchunk: &compile::Chunk,
    ) {
        let vs = self.lyt.val_size as u32;
        let fsz = self.lyt.frame_size as u32;
        self.flush_seq();
        // `code.ip = next; *op_ip = off` — on the fast path `code.ip` gets
        // overwritten with the callee offset below, but `*op_ip` locates
        // this op for any propagated error, and the slow path needs both.
        self.mark_op(off, next);
        self.settle_seq();

        // Runtime gates, all checked in the pre-branch block:
        //   frames.len() < INLINE_CALL_DEPTH    (else Flow::Call to driver)
        //   regs.cap - regs.len >= callee.regs  (else Vec grow → shim)
        //   frames.len() < frames.cap           (ditto for the push)
        let thread = self.env.thread.clone();
        let flen = self.frames_len();
        let dcap = self.iconst(INLINE_CALL_DEPTH as i64);
        let depth_ok = self.icmp(I64, "ULT", &flen, &dcap);
        let rlen = self.ld_word(&thread, (self.lyt.regs_off + self.lyt.vec_len) as u32);
        let rcap = self.ld_word(&thread, (self.lyt.regs_off + self.lyt.vec_cap) as u32);
        let slack = self.sub(&rcap, &rlen);
        let need = self.iconst(cchunk.regs as i64);
        let regs_ok = self.icmp(I64, "UGE", &slack, &need);
        let fcap = self.ld_word(&thread, (self.lyt.frames_off + self.lyt.vec_cap) as u32);
        let frames_ok = self.icmp(I64, "ULT", &flen, &fcap);
        let ok01 = self.band1(&depth_ok, &regs_ok);
        let ok = self.band1(&ok01, &frames_ok);
        let fast = self.create_block();
        let slow = self.create_block();
        self.cbr(&ok, &fast, &slow);

        // ---- slow: the megashim handles Flow::Call-at-cap, grows, errors ----
        self.start_block(&slow);
        let ap = self.reg_list_slot(args);
        let b = self.body_idx(body.index());
        let d = self.iconst(dst.index() as i64);
        let n = self.iconst(args.len() as i64);
        let rp = self
            .hcall(
                H::CallBody,
                &[
                    A::P(self.env.env.clone()),
                    A::P(self.env.bodies_tbl.clone()),
                    A::V(b),
                    A::V(d),
                    A::P(ap),
                    A::V(n),
                ],
            )
            .unwrap();
        self.post_call(i, dst, &rp);

        // ---- fast: enter_call_regs inline ----
        self.start_block(&fast);
        let rp0 = self.ld(
            PTR,
            &thread,
            (self.lyt.regs_off + self.lyt.vec_ptr) as u32,
            8,
        );
        // new_base = old regs.len (loaded above as `rlen`)
        let vsc = self.iconst(vs as i64);
        let nboff = self.mul(&rlen, &vsc);
        let nwin = self.gep_dyn(&rp0, &nboff); // callee window base
        // resize fill: `Val::Null` — only the tag byte is ever read
        let tnull = self.tconst(self.lyt.t_null);
        for k in 0..cchunk.regs as u32 {
            self.st_off(
                &nwin,
                &tnull,
                k * vs + self.lyt.val_tag as u32,
                self.lyt.tag_size as u32,
            );
        }
        let nregs_c = self.iconst(cchunk.regs as i64);
        let nlen = self.add(&rlen, &nregs_c);
        self.st_word(&thread, &nlen, (self.lyt.regs_off + self.lyt.vec_len) as u32);
        // arg copies: `regs[new_base + param] = regs[caller_base + arg]` —
        // both indices are compile-time constants; the caller window is
        // `v_regs` (unmoved — capacity was checked).
        let caller = self.regs();
        for (i, &pr) in cchunk.params.iter().enumerate() {
            let s = self.vaddr(&caller, args[i].index() as u32);
            let d = self.vaddr(&nwin, pr.index() as u32);
            self.cpy_val(&d, &s);
        }
        // caller frame's saved ip = resume offset (what `code.ip` held)
        let fptr = self.ld(
            PTR,
            &thread,
            (self.lyt.frames_off + self.lyt.vec_ptr) as u32,
            8,
        );
        let one = self.iconst(1);
        let two = self.iconst(2);
        let fm1 = self.sub(&flen, &one);
        let fszc = self.iconst(fsz as i64);
        let cfo = self.mul(&fm1, &fszc);
        let cf = self.gep_dyn(&fptr, &cfo);
        let nxt = self.goff(next);
        self.st_word(&cf, &nxt, self.lyt.frame_ip as u32);
        // push the callee frame
        let nfo = self.mul(&flen, &fszc);
        let nf = self.gep_dyn(&fptr, &nfo);
        let biw = self.body_idx(body.index());
        let bi = self.trunc_to(I32, &biw);
        self.st_off(&nf, &bi, self.lyt.frame_chunk as u32, 4);
        let cip = self.chunk_off(body.index());
        self.st_word(&nf, &cip, self.lyt.frame_ip as u32);
        let rr = self.iconst32(dst.index() as i64);
        self.st_off(&nf, &rr, self.lyt.frame_ret as u32, 4);
        self.st_word(&nf, &rlen, self.lyt.frame_base as u32);
        let nfl = self.add(&flen, &one);
        self.st_word(&thread, &nfl, (self.lyt.frames_off + self.lyt.vec_len) as u32);
        // code.ip = callee chunk offset
        self.store_ip(&cip);
        // the call itself — a direct native call, same `env` bundle (the
        // callee reads its own fields out of it; `env.out` is shared)
        let rv = self.v();
        self.ln(format!(
            "{rv} = llvm.call @{} ({}) : llvm.func <{VOID} ({PTR}) variadic = false>",
            body_name(body.index()),
            self.env.env
        ));
        // driver's `Flow::Return` pop, inlined over the probed
        // `RtResult<Flow>`/Frame layout: `*out == Ok(Flow::Return(v))` → pop
        // the callee frame, truncate `regs`, restore the caller's saved ip,
        // write `v` into the `return_reg` slot; anything else propagates
        // `out` verbatim through `eret` (what `mj_pop_return`'s non-2 tags
        // did).
        let oty = match self.lyt.out_tsz {
            1 => I8,
            2 => I16,
            4 => I32,
            8 => I64,
            d => unreachable!("bad out tag width {d}"),
        };
        let out = self.env.out.clone();
        let otag = self.ld(oty, &out, self.lyt.out_tag as u32, self.lyt.out_tsz as u32);
        let orwant = self.iconst_ty(oty, self.lyt.out_ret as i64);
        let isret = self.icmp(oty, "EQ", &otag, &orwant);
        let resumed = self.create_block();
        let er = self.ex.eret.clone();
        self.cbr(&isret, &resumed, &er);
        self.start_block(&resumed);
        // popped = frames[flen-1]; frames.len -= 1; regs.len = popped.base
        let flen2 = self.frames_len();
        let fptr2 = self.ld(
            PTR,
            &thread,
            (self.lyt.frames_off + self.lyt.vec_ptr) as u32,
            8,
        );
        let fm2 = self.sub(&flen2, &one);
        let pfo = self.mul(&fm2, &fszc);
        let pf = self.gep_dyn(&fptr2, &pfo);
        let pbase = self.ld_word(&pf, self.lyt.frame_base as u32);
        let pret32 = self.ld(I32, &pf, self.lyt.frame_ret as u32, 4);
        let pret = self.zext64(&pret32);
        self.st_word(&thread, &fm2, (self.lyt.frames_off + self.lyt.vec_len) as u32);
        self.st_word(&thread, &pbase, (self.lyt.regs_off + self.lyt.vec_len) as u32);
        // caller = frames[flen-2]; code.ip = caller.ip
        let fm3 = self.sub(&flen2, &two);
        let cfo2 = self.mul(&fm3, &fszc);
        let cf2 = self.gep_dyn(&fptr2, &cfo2);
        let cip2 = self.ld_word(&cf2, self.lyt.frame_ip as u32);
        self.store_ip(&cip2);
        let cbase = self.ld_word(&cf2, self.lyt.frame_base as u32);
        // `thread.regs` may have moved under the callee — rebuild the window
        // and write the return value into `caller_base + return_reg`
        let rp2 = self.ld(
            PTR,
            &thread,
            (self.lyt.regs_off + self.lyt.vec_ptr) as u32,
            8,
        );
        let widx = self.add(&cbase, &pret);
        let woff = self.mul(&widx, &vsc);
        let daddr = self.gep_dyn(&rp2, &woff);
        let retp = self.gep(&out, self.lyt.out_ret_pay as u32);
        self.cpy_val(&daddr, &retp);
        let boff2 = self.mul(&cbase, &vsc);
        let regs2 = self.gep_dyn(&rp2, &boff2);
        let a = self.v_regs.clone();
        self.st(&a, &regs2);
        self.rearm_seq();
        self.dst_refresh(i, dst, &regs2);
    }

    /// `Call` — one `mj_call_dyn` hop: callee resolution (incl. the
    /// `CallTarget::Value` signature check), depth cap, enter/run/pop.
    fn emit_call(
        &mut self,
        i: usize,
        off: usize,
        next: usize,
        dst: Reg,
        callee: Reg,
        args: &[Reg],
    ) {
        // Flush first — the shim reads the callee Val and arg slots out of
        // the window. `code.ip = next; *op_ip = off` before the call: a
        // `not_callable` error locates via `op_ip` exactly like the
        // interpreter's decoder-advanced arm.
        self.flush_seq();
        self.mark_op(off, next);
        self.settle_seq();
        let ap = self.reg_list_slot(args);
        let nargs = args.len();
        let regs = self.regs();
        let c = self.iconst(callee.index() as i64);
        let d = self.iconst(dst.index() as i64);
        let n = self.iconst(nargs as i64);
        let rp = self
            .hcall(
                H::CallDyn,
                &[
                    A::P(self.env.env.clone()),
                    A::P(self.env.bodies_tbl.clone()),
                    A::P(regs),
                    A::V(c),
                    A::V(d),
                    A::P(ap),
                    A::V(n),
                ],
            )
            .unwrap();
        self.post_call(i, dst, &rp);
    }

    /// After a megashim call: null means propagate `out` verbatim (`eret`);
    /// otherwise the returned pointer is the caller's rebuilt window —
    /// re-pin it, re-arm the quota (the callee consumed fuel/ops_left
    /// through its own `bcn`), refresh dst's shadow, and resume.
    fn post_call(&mut self, i: usize, dst: Reg, rp: &str) {
        let np = {
            let z = self.iconst(0);
            self.i2p(&z)
        };
        let isz = self.icmp(PTR, "EQ", rp, &np);
        let resumed = self.create_block();
        let er = self.ex.eret.clone();
        self.cbr(&isz, &er, &resumed);
        self.start_block(&resumed);
        self.rearm_seq();
        let a = self.v_regs.clone();
        self.st(&a, rp);
        self.dst_refresh(i, dst, rp);
    }

    /// Post-call tail: `regs[dst]` now holds the callee's return value —
    /// refresh its scalar shadow (if any) and resume at the next op.
    fn dst_refresh(&mut self, i: usize, dst: Reg, regs2: &str) {
        let d = dst.index() as u32;
        let ty = self.tag_ty();
        if let Some((sv, ok)) = self.int.get(&d).cloned() {
            let a = self.vaddr(regs2, d);
            let t = self.ld_tag(&a);
            let want = self.tconst(self.lyt.t_int);
            let k = self.icmp(ty, "EQ", &t, &want);
            let v = self.ld(I64, &a, self.lyt.val_pay as u32, 8);
            let z = self.iconst(0);
            let sv0 = self.sel(I64, &k, &v, &z);
            self.st(&sv, &sv0);
            let k8 = self.i1_to_i8(&k);
            self.st_al(&ok, &k8, 1);
        } else if let Some((sv, ok)) = self.float.get(&d).cloned() {
            let a = self.vaddr(regs2, d);
            let t = self.ld_tag(&a);
            let want = self.tconst(self.lyt.t_float);
            let k = self.icmp(ty, "EQ", &t, &want);
            let v = self.ld(F64, &a, self.lyt.val_pay as u32, 8);
            let z = self.f64const(0.0);
            let sv0 = self.sel(F64, &k, &v, &z);
            self.st(&sv, &sv0);
            let k8 = self.i1_to_i8(&k);
            self.st_al(&ok, &k8, 1);
        }
        let nb = self.next_blk(i);
        self.br(&nb);
    }
}

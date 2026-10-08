//! mimas-wasmgen — emit a freestanding wasm module from a compiled mimas
//! [`compile::Ir`] (the SSA-ish block IR) via the Bytecode Alliance `waffle`
//! backend: [`wfull::emit_waffle_ir`].
//!
//! Third specialization lane next to bcgen (Rust source, AOT) and jit
//! (Cranelift, runtime/native): pure byte emission, so the emitter itself runs
//! anywhere — including inside wasm, i.e. runtime codegen on the browser host
//! where neither rustc nor Cranelift can produce executable code.
//!
//! A body that can't emit is skipped, not fatal — the caller keeps it on the
//! interpreter. Every body of every corpus game currently emits.
//!
//! Layout: `wrt` holds the shared substrate (class analysis, heap tags,
//! statics, hand-coded helpers, dynamic-call trampolines); `wfull` lowers
//! bodies to waffle `FunctionBody`s and assembles the module.
//!
//! Divergences from the interpreter, all deliberate at this stage:
//! - checked int arith traps (`unreachable`) instead of raising
//!   `RtErr::IntegerOverflow` — same halt, different payload.
//! - `x % 0` / `x / 0` trap in wasm where the interpreter raises
//!   `ModByZero`/`DivByZero` — same control transfer, no `RtErr`.
//! - `i64::MIN % -1` returns 0 in wasm; the interpreter raises overflow.

pub mod wfull;
pub mod wrt;

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
    /// Bodies skipped (unsupported inst or operand class).
    pub skipped: Vec<Skip>,
}

pub(crate) type Bail = String;

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
#[derive(Clone, Debug)]
pub(crate) struct Sig {
    pub params: Vec<K>,
    pub ret: Option<K>,
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
/// Passthrough-native classification for [`wfull::emit_waffle_ir`]'s
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

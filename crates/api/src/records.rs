use shared::{AdtId, Literal, Ty};

/// A per-native literal checker, submitted via `vm::api::NativeValidator` and joined by Rust
/// path (same mechanism as `NativeDoc`). The solver calls it when *every* call argument is a
/// literal: `Ok(())` proves the call can't raise, so a `T!` return narrows to `T`; `Err(msg)`
/// becomes a compile error at the call site. Non-literal calls are never validated and keep
/// the honest `T!`.
pub type LitValidator = fn(&[Literal]) -> Result<(), String>;

pub struct ApiFunction<C> {
    pub name: String,
    pub module: Vec<String>,
    pub parameters: Vec<Option<Ty>>,
    pub return_ty: Option<Ty>,
    pub doc: String,
    pub validate: Option<LitValidator>,
    pub call: C,
}

pub struct ApiMethod<C> {
    pub recv_ty: Ty,
    pub name: String,
    pub parameters: Vec<Option<Ty>>,
    pub return_ty: Option<Ty>,
    pub takes_self: bool,
    /// The receiver is mutated by the call -- a `&mut` first param on the Rust side. Set by
    /// matching the registered fn's path against `vm::api::NativeMutates` submissions, the
    /// same way `doc` is joined. The solver reads this to reject mutating a collection while
    /// a `for` loop is iterating it.
    pub mutates_recv: bool,
    pub doc: String,
    pub validate: Option<LitValidator>,
    pub call: C,
}

pub struct ApiConstant {
    pub name: String,
    pub module: Vec<String>,
    /// `Some` makes this an associated constant on that receiver instead of a free one.
    pub recv_ty: Option<Ty>,
    pub ty: Ty,
    pub value: Literal,
    pub doc: String,
}

pub enum ApiEntry<C> {
    Function(ApiFunction<C>),
    Method(ApiMethod<C>),
    Constant(ApiConstant),
}

pub struct ApiAdt {
    pub name: String,
    pub module: Vec<String>,
    pub kind: ApiAdtKind,
    pub adt_id: AdtId,
    pub doc: String,
    pub variants: Vec<ApiVariant>,
    /// `Api::add_bin_op`/`add_unary_op` set this — the solver accepts infix
    /// and unary ops on the type and defers the operand shapes to runtime.
    pub op_overloads: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiAdtKind {
    Enum,
    Struct,
}

pub struct ApiVariant {
    pub name: String,
    pub layout_id: AdtId,
    pub doc: String,
    pub fields: ApiVariantFields,
}

pub enum ApiVariantFields {
    Unit,
    Tuple(Vec<Ty>),
    Named(Vec<(String, Ty)>),
}

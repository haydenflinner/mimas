use shared::{AdtId, Literal, Ty, units::Dim};

/// A per-native literal checker, submitted via `vm::api::NativeValidator` and joined by Rust
/// path (same mechanism as `NativeDoc`). The solver calls it when *every* call argument is a
/// literal: `Ok(())` proves the call can't raise, so a `T!` return narrows to `T`; `Err(msg)`
/// becomes a compile error at the call site. Non-literal calls are never validated and keep
/// the honest `T!`.
pub type LitValidator = fn(&[Literal]) -> Result<(), String>;

/// `(file!(), CARGO_MANIFEST_DIR, line!())` captured at a native's definition site, joined by
/// Rust path the same way `NativeDoc` is (see `vm::api::NativeSrc`). `file` is relative to the
/// workspace root the defining crate compiled under; `manifest` pins that workspace down so a
/// host can reconstruct the absolute path and link the built-in's symbol menu to its source
/// instead of shrugging "no source here".
pub type NativeSrc = (&'static str, &'static str, u32);

pub struct ApiFunction<C> {
    pub name: String,
    pub module: Vec<String>,
    pub parameters: Vec<Option<Ty>>,
    /// Per-slot dimension the checker enforces at call sites, parallel to `parameters`
    /// (`None` = unchecked). Populated from unit-carrying `MimasType`s like `Secs`/`Hz`.
    pub param_dims: Vec<Option<Dim>>,
    /// The identifiers the Rust signature gave each parameter, parallel to `parameters`.
    /// Synthesized `arg{i}` names fill in for natives whose signature isn't on record
    /// (`add_described`, or a pruned `NativeMeta` submission -- see `vm::api::NativeMeta`).
    pub param_names: Vec<String>,
    pub return_ty: Option<Ty>,
    /// Dimension of the return value, when it measures something.
    pub return_dim: Option<Dim>,
    pub doc: String,
    pub validate: Option<LitValidator>,
    pub src: Option<NativeSrc>,
    pub call: C,
}

pub struct ApiMethod<C> {
    pub recv_ty: Ty,
    pub name: String,
    pub parameters: Vec<Option<Ty>>,
    /// Per-slot dimension the checker enforces at call sites, parallel to `parameters`.
    pub param_dims: Vec<Option<Dim>>,
    /// Same deal as [`ApiFunction::param_names`].
    pub param_names: Vec<String>,
    pub return_ty: Option<Ty>,
    /// Dimension of the return value, when it measures something.
    pub return_dim: Option<Dim>,
    pub takes_self: bool,
    /// The receiver is mutated by the call -- a `&mut` first param on the Rust side. Set by
    /// matching the registered fn's path against `vm::api::NativeMutates` submissions, the
    /// same way `doc` is joined. The solver reads this to reject mutating a collection while
    /// a `for` loop is iterating it.
    pub mutates_recv: bool,
    pub doc: String,
    pub validate: Option<LitValidator>,
    pub src: Option<NativeSrc>,
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
    /// `(name, ty, dim)` — `dim` carries the field's dimension for the
    /// units pass (`None` = unchecked), populated from unit-carrying
    /// `MimasType`s like `Secs`/`Px` the same way `param_dims` is.
    Named(Vec<(String, Ty, Option<Dim>)>),
}

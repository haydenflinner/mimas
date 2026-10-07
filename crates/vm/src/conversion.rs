use std::collections::HashMap;

use api::Registry;
use shared::{
    FnHeader, FnParam, Ty,
    units::{Dim, DimTree},
};

use crate::{Array, Ctx, Dict, DictMap, Instance, RtErr, RtResult, Str, Val};
use gc_arena::Gc;

#[derive(Debug, Clone)]
pub struct TypeError {
    pub expected: String,
    pub got: String,
}

impl From<TypeError> for RtErr {
    fn from(e: TypeError) -> Self {
        RtErr::InvalidArgument(format!("expected {}, received {}", e.expected, e.got))
    }
}

pub trait MimasType<'gc>: Sized {
    fn mimas_ty(reg: &Registry) -> Option<Ty>;
    /// The dimension the checker should track for this slot, when the type stands for a
    /// measured quantity (`Secs` -> `s`). `None` means unchecked, the same as before.
    fn mimas_dim(_: &Registry) -> Option<Dim> {
        None
    }
    /// The dimension claim for this slot when it may nest inside a container —
    /// `Vec<Px>` claims `[px]`, `(PxX, PxY)` claims `(px↔, px↕)` element-wise.
    /// Defaults to the scalar [`Self::mimas_dim`]; containers override it.
    fn mimas_dim_tree(reg: &Registry) -> Option<DimTree> {
        Self::mimas_dim(reg).map(DimTree::Q)
    }
    fn from_value(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError>;
    fn into_value(self, ctx: Ctx<'gc>) -> Val<'gc>;
}

/// Lets native fns return either a plain `T: MimasType` (Ok-wrapped automatically) or a
/// `Result<T, E: Into<RtError>>` for runtime-error surfacing. The `IntoFn` macro goes through
/// this trait so users don't need to pick a different registration path just to be able to fail.
pub trait IntoNativeResult<'gc> {
    fn return_ty(reg: &Registry) -> Option<Ty>;
    /// The dimension claim the return value carries, when it stands for measured
    /// quantities — a [`DimTree`] so a `(PxX, PxY)` return claims `(px↔, px↕)`.
    fn return_dim(reg: &Registry) -> Option<DimTree> {
        let _ = reg;
        None
    }
    fn into_native_result(self, ctx: Ctx<'gc>) -> RtResult<Val<'gc>>;
}

impl<'gc, T> IntoNativeResult<'gc> for T
where
    T: MimasType<'gc>,
{
    fn return_ty(reg: &Registry) -> Option<Ty> {
        <T as MimasType<'gc>>::mimas_ty(reg)
    }
    fn return_dim(reg: &Registry) -> Option<DimTree> {
        <T as MimasType<'gc>>::mimas_dim_tree(reg)
    }
    fn into_native_result(self, ctx: Ctx<'gc>) -> RtResult<Val<'gc>> {
        Ok(self.into_value(ctx))
    }
}

impl<'gc, T, E> IntoNativeResult<'gc> for Result<T, E>
where
    T: MimasType<'gc>,
    E: Into<RtErr>,
{
    fn return_ty(reg: &Registry) -> Option<Ty> {
        <T as MimasType<'gc>>::mimas_ty(reg)
    }
    fn return_dim(reg: &Registry) -> Option<DimTree> {
        <T as MimasType<'gc>>::mimas_dim_tree(reg)
    }
    fn into_native_result(self, ctx: Ctx<'gc>) -> RtResult<Val<'gc>> {
        self.map(|v| v.into_value(ctx)).map_err(Into::into)
    }
}

pub enum NeverReturn {}

impl<'gc> MimasType<'gc> for NeverReturn {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        Some(Ty::Never)
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        Err(ty_error("!", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        match self {}
    }
}

impl<'gc> MimasType<'gc> for () {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        Some(Ty::Unit)
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        if v == Val::Null {
            Ok(())
        } else {
            Err(ty_error("unit", v))
        }
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Null
    }
}

impl<'gc> MimasType<'gc> for bool {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        Some(Ty::Bool)
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_bool().ok_or_else(|| ty_error("bool", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Bool(self)
    }
}

impl<'gc> MimasType<'gc> for String {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        Some(Ty::Str)
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        Ok(v.as_str()
            .ok_or_else(|| ty_error("str", v))?
            .as_str()
            .to_string())
    }
    fn into_value(self, ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Str(ctx.intern(&self))
    }
}

macro_rules! impl_int {
    ($($t:ty),+) => { $(
        impl<'gc> MimasType<'gc> for $t {
            fn mimas_ty(_: &Registry) -> Option<Ty> { Some(Ty::Int) }
            fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
                v.as_int()
                    .and_then(|n| <$t>::try_from(n).ok())
                    .ok_or_else(|| ty_error(stringify!($t), v))
            }
            fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> { Val::Int(self as i64) }
        }
    )+ }
}
impl_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

macro_rules! impl_float {
    ($($t:ty),+) => { $(
        impl<'gc> MimasType<'gc> for $t {
            fn mimas_ty(_: &Registry) -> Option<Ty> { Some(Ty::Float) }
            fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
                v.as_float().map(|f| f as $t).ok_or_else(|| ty_error(stringify!($t), v))
            }
            fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> { Val::Float(self as f64) }
        }
    )+ }
}
impl_float!(f32, f64);

impl<'gc> MimasType<'gc> for Val<'gc> {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        None
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        Ok(v)
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        self
    }
}

impl<'gc> MimasType<'gc> for Str<'gc> {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        Some(Ty::Str)
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_str().ok_or_else(|| ty_error("str", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Str(self)
    }
}

impl<'gc> MimasType<'gc> for &'gc str {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        Some(Ty::Str)
    }

    fn from_value(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        let s = v.as_str().ok_or_else(|| ty_error("str", v))?;
        // `&'gc str` must point into arena memory — an inline `Str`
        // carries its bytes in the value word, so materialize it through
        // the table.
        match s.as_gc_str() {
            Some(g) => Ok(g),
            None => Ok(ctx.intern_heap(s.as_str()).as_gc_str().unwrap()),
        }
    }

    fn into_value(self, ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Str(ctx.intern(self))
    }
}

impl<'gc> MimasType<'gc> for Array<'gc> {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        Some(Ty::array(Ty::Anon(0)))
    }
    fn from_value(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        // `as_untyped_array` demotes a typed array in place and hands back its
        // inner `Array`, so a mutating native writes through to the same
        // elements the script's `IntArray`/`FloatArray` still points at.
        v.as_untyped_array(ctx).ok_or_else(|| ty_error("array", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Array(self)
    }
}

impl<'gc> MimasType<'gc> for Dict<'gc> {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        Some(Ty::Dict(Box::new(Ty::Anon(0))))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_dict().ok_or_else(|| ty_error("dict", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Dict(self)
    }
}

// ----- DataFrame / PlExpr -----------------------------------------------------------------------
//
// Unlike a `#[mimas] struct`, these are real `Val` variants (see `val::DataFrame`/`val::PlExpr`),
// not `Val::Instance`s decomposed field-by-field -- there's no way to fit an opaque, `Arc`-backed
// `polars::frame::DataFrame` through `Val`'s closed set of field types otherwise. But the type
// checker still wants a `Ty::Adt(AdtId)` to hang parameter/return types off of, so these two
// lifetime-free marker types exist purely to be `add_adt`'d (by `library::std_lib::dataframe`) and
// hand out an `AdtId` -- they carry no data of their own and are never constructed.
#[cfg(feature = "dataframe")]
pub struct DataFrameTy;
#[cfg(feature = "dataframe")]
impl crate::adt::MimasAdt for DataFrameTy {
    fn descriptor(_reg: &Registry) -> crate::adt::ApiAdtDescriptor {
        crate::adt::ApiAdtDescriptor {
            name: "DataFrame",
            module: &["std", "polars"],
            kind: api::ApiAdtKind::Struct,
            doc: "A polars DataFrame -- a column-oriented, `Arc`-backed table.",
            variants: vec![crate::adt::ApiVariantShape {
                name: "DataFrame".into(),
                doc: "",
                fields: api::ApiVariantFields::Unit,
            }],
        }
    }
}

#[cfg(feature = "dataframe")]
pub struct PlExprTy;
#[cfg(feature = "dataframe")]
impl crate::adt::MimasAdt for PlExprTy {
    fn descriptor(_reg: &Registry) -> crate::adt::ApiAdtDescriptor {
        crate::adt::ApiAdtDescriptor {
            name: "PlExpr",
            module: &["std", "polars"],
            kind: api::ApiAdtKind::Struct,
            doc: "A polars expression tree, e.g. `col(\"age\") > 30`.",
            variants: vec![crate::adt::ApiVariantShape {
                name: "PlExpr".into(),
                doc: "",
                fields: api::ApiVariantFields::Unit,
            }],
        }
    }
}

#[cfg(feature = "dataframe")]
impl<'gc> MimasType<'gc> for crate::val::DataFrame<'gc> {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        Some(Ty::adt(reg.get::<DataFrameTy>()?.adt_id))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_dataframe().ok_or_else(|| ty_error("DataFrame", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::DataFrame(self)
    }
}

#[cfg(feature = "dataframe")]
impl<'gc> MimasType<'gc> for crate::val::PlExpr<'gc> {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        Some(Ty::adt(reg.get::<PlExprTy>()?.adt_id))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_plexpr().ok_or_else(|| ty_error("PlExpr", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::PlExpr(self)
    }
}

#[cfg(feature = "dataframe")]
pub struct GroupByTy;
#[cfg(feature = "dataframe")]
impl crate::adt::MimasAdt for GroupByTy {
    fn descriptor(_reg: &Registry) -> crate::adt::ApiAdtDescriptor {
        crate::adt::ApiAdtDescriptor {
            name: "GroupBy",
            module: &["std", "polars"],
            kind: api::ApiAdtKind::Struct,
            doc: "The result of `DataFrame::group_by`, before `.agg([..])`.",
            variants: vec![crate::adt::ApiVariantShape {
                name: "GroupBy".into(),
                doc: "",
                fields: api::ApiVariantFields::Unit,
            }],
        }
    }
}

#[cfg(feature = "dataframe")]
impl<'gc> MimasType<'gc> for crate::val::GroupBy<'gc> {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        Some(Ty::adt(reg.get::<GroupByTy>()?.adt_id))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_group_by().ok_or_else(|| ty_error("GroupBy", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::GroupBy(self)
    }
}

// ----- dec ---------------------------------------------------------------------------------
//
// An exact decimal (`shared::dec`, docs/decimal.md). The payload is the unboxed `Val::Dec`;
// this marker is `add_adt`'d so `dec` works in annotations, beside `int` and `float`.
// r[impl mimas.dec]
pub struct DecTy;
impl crate::adt::MimasAdt for DecTy {
    fn descriptor(_reg: &Registry) -> crate::adt::ApiAdtDescriptor {
        crate::adt::ApiAdtDescriptor {
            name: "dec",
            module: &[],
            kind: api::ApiAdtKind::Struct,
            doc: "An exact decimal number: about 16 digits, kept as written (`3.50`).",
            variants: vec![crate::adt::ApiVariantShape {
                name: "dec".into(),
                doc: "",
                fields: api::ApiVariantFields::Unit,
            }],
        }
    }
}

impl<'gc> MimasType<'gc> for shared::dec::Dec {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        Some(Ty::adt(reg.get::<DecTy>()?.adt_id))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        match v {
            Val::Dec(d) => Ok(d),
            Val::Int(n) => shared::dec::Dec::from_int(n).map_err(|_| ty_error("dec", v)),
            _ => Err(ty_error("dec", v)),
        }
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Dec(self)
    }
}

// ----- Tensor ------------------------------------------------------------------------------
//
// Same shape as `DataFrameTy`: the payload lives in a real `Val` variant (`val::Tensor`), and
// this marker exists purely to be `add_adt`'d so `Tensor` works in mimas annotations.
#[cfg(feature = "tensor")]
pub struct TensorTy;
#[cfg(feature = "tensor")]
impl crate::adt::MimasAdt for TensorTy {
    fn descriptor(_reg: &Registry) -> crate::adt::ApiAdtDescriptor {
        crate::adt::ApiAdtDescriptor {
            name: "Tensor",
            module: &["std", "tensor"],
            kind: api::ApiAdtKind::Struct,
            doc: "An N-D `f32` tensor on the Burn Flex backend (std::tensor).",
            variants: vec![crate::adt::ApiVariantShape {
                name: "Tensor".into(),
                doc: "",
                fields: api::ApiVariantFields::Unit,
            }],
        }
    }
}

#[cfg(feature = "tensor")]
impl<'gc> MimasType<'gc> for crate::val::Tensor<'gc> {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        Some(Ty::adt(reg.get::<TensorTy>()?.adt_id))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_tensor().ok_or_else(|| ty_error("Tensor", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Tensor(self)
    }
}

// ----- DarklyImage -------------------------------------------------------------------------
//
// Same reasoning as DataFrame/PlExpr above: a decoded pixel buffer doesn't fit Val's closed set
// of field types, so it's a real Val variant with a marker adt to hang a Ty off of, not a
// `#[mimas] struct`.
#[cfg(feature = "darkly")]
pub struct DarklyImageTy;
#[cfg(feature = "darkly")]
impl crate::adt::MimasAdt for DarklyImageTy {
    fn descriptor(_reg: &Registry) -> crate::adt::ApiAdtDescriptor {
        crate::adt::ApiAdtDescriptor {
            name: "DarklyImage",
            module: &["std", "darkly"],
            kind: api::ApiAdtKind::Struct,
            doc: "Decoded pixel bytes from a `.darkly` raster/mask layer.",
            variants: vec![crate::adt::ApiVariantShape {
                name: "DarklyImage".into(),
                doc: "",
                fields: api::ApiVariantFields::Unit,
            }],
        }
    }
}

#[cfg(feature = "darkly")]
impl<'gc> MimasType<'gc> for crate::val::DarklyImage<'gc> {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        Some(Ty::adt(reg.get::<DarklyImageTy>()?.adt_id))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_darkly_image()
            .ok_or_else(|| ty_error("DarklyImage", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::DarklyImage(self)
    }
}

impl<'gc> MimasType<'gc> for Instance<'gc> {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        None
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_instance().ok_or_else(|| ty_error("instance", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Instance(self)
    }
}

impl<'gc, T: MimasType<'gc>> MimasType<'gc> for Option<T> {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        T::mimas_ty(reg).map(|i| Ty::Option(Box::new(i)))
    }
    fn mimas_dim(reg: &Registry) -> Option<Dim> {
        T::mimas_dim(reg)
    }
    fn mimas_dim_tree(reg: &Registry) -> Option<DimTree> {
        T::mimas_dim_tree(reg)
    }
    fn from_value(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        if v == Val::Null {
            Ok(None)
        } else {
            Ok(Some(T::from_value(ctx, v)?))
        }
    }
    fn into_value(self, ctx: Ctx<'gc>) -> Val<'gc> {
        self.map(|v| v.into_value(ctx)).unwrap_or(Val::Null)
    }
}

/// Rust-side mirror of mimas's `T!`. Native authors return `Raisable<T>` to expose a mimas
/// result; `Raisable::Raised(msg)` becomes `Val::Raised`, which mimas code can `absolve` or
/// unwrap with `!`. Use `Result<T, E: Display>::into()` to convert in one step.
pub enum Raisable<T> {
    Ok(T),
    Raised(String),
}

impl<T, E: std::fmt::Display> From<std::result::Result<T, E>> for Raisable<T> {
    fn from(r: std::result::Result<T, E>) -> Self {
        match r {
            Ok(v) => Raisable::Ok(v),
            Err(e) => Raisable::Raised(e.to_string()),
        }
    }
}

impl<'gc, T: MimasType<'gc>> MimasType<'gc> for Raisable<T> {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        T::mimas_ty(reg).map(|i| Ty::Result(Box::new(i)))
    }
    fn mimas_dim(reg: &Registry) -> Option<Dim> {
        T::mimas_dim(reg)
    }
    fn mimas_dim_tree(reg: &Registry) -> Option<DimTree> {
        T::mimas_dim_tree(reg)
    }
    fn from_value(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        match v {
            Val::Raised(s) => Ok(Raisable::Raised(
                // the str-message API stays str — stringify is cold here,
                // and an unprintable payload still reads as a note
                ctx.to_string(*s)
                    .unwrap_or_else(|_| "<unprintable raise payload>".into()),
            )),
            other => Ok(Raisable::Ok(T::from_value(ctx, other)?)),
        }
    }
    fn into_value(self, ctx: Ctx<'gc>) -> Val<'gc> {
        match self {
            Raisable::Ok(v) => v.into_value(ctx),
            Raisable::Raised(s) => Val::Raised(Gc::new(ctx.mutation(), Val::Str(ctx.intern(&s)))),
        }
    }
}

// Mutating natives should take `Array<'gc>` / `Dict<'gc>` directly so they can use the heap
// newtype's `borrow_mut(&ctx)`. These collection impls just copy for ergonomic non-mutating
// reads.

impl<'gc, T: MimasType<'gc>> MimasType<'gc> for Vec<T> {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        T::mimas_ty(reg).map(|i| Ty::array(i))
    }
    fn mimas_dim_tree(reg: &Registry) -> Option<DimTree> {
        T::mimas_dim_tree(reg).map(|t| DimTree::Arr(Box::new(t)))
    }
    fn from_value(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        let arr = v
            .as_untyped_array(ctx)
            .ok_or_else(|| ty_error("array", v))?;
        let items: Vec<Val<'gc>> = arr.0.borrow().iter().copied().collect();
        items.into_iter().map(|v| T::from_value(ctx, v)).collect()
    }
    fn into_value(self, ctx: Ctx<'gc>) -> Val<'gc> {
        let out: Vec<Val<'gc>> = self.into_iter().map(|v| v.into_value(ctx)).collect();
        ctx.array_val(out)
    }
}

impl<'gc, T: MimasType<'gc>> MimasType<'gc> for HashMap<String, T> {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        T::mimas_ty(reg).map(|i| Ty::Dict(Box::new(i)))
    }
    fn from_value(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        let dict = v.as_dict().ok_or_else(|| ty_error("dict", v))?;
        let items: Vec<(Str<'gc>, Val<'gc>)> =
            dict.0.borrow().iter().map(|(&k, &v)| (k, v)).collect();
        items
            .into_iter()
            .map(|(k, v)| Ok((k.as_str().to_string(), T::from_value(ctx, v)?)))
            .collect()
    }
    fn into_value(self, ctx: Ctx<'gc>) -> Val<'gc> {
        let mut out = DictMap::new();
        for (k, v) in self {
            let k = ctx.intern(&k);
            out.insert(k, v.into_value(ctx));
        }
        Val::Dict(ctx.new_dict(out))
    }
}

// ----- Anon<'gc, N> ----------------------------------------------------------------------------

impl<'gc, const N: u32> MimasType<'gc> for crate::anon::Anon<'gc, N> {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        Some(Ty::Anon(N))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        Ok(crate::anon::Anon(v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        self.0
    }
}

impl<'gc, const N: u32> MimasType<'gc> for crate::anon::ArrayOf<'gc, N> {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        Some(Ty::array(Ty::Anon(N)))
    }
    fn from_value(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_untyped_array(ctx)
            .map(crate::anon::ArrayOf)
            .ok_or_else(|| ty_error("array", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Array(self.0)
    }
}

impl<'gc, const N: u32> MimasType<'gc> for crate::anon::SeqOf<'gc, N> {
    fn mimas_ty(_: &Registry) -> Option<Ty> { Some(Ty::array(Ty::Anon(N))) }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        if v.is_seq() { Ok(crate::anon::SeqOf(v)) } else { Err(ty_error("array", v)) }
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> { self.0 }
}

impl<'gc, const N: u32> MimasType<'gc> for crate::anon::DictOf<'gc, N> {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        Some(Ty::Dict(Box::new(Ty::Anon(N))))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_dict()
            .map(crate::anon::DictOf)
            .ok_or_else(|| ty_error("dict", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Dict(self.0)
    }
}

impl<'gc, A: MimasType<'gc>, R: MimasType<'gc>> MimasType<'gc> for crate::anon::Fn1<'gc, A, R> {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        Some(Ty::Fn(FnHeader::new(
            vec![FnParam::new(None, A::mimas_ty(reg)?, false)],
            R::mimas_ty(reg)?,
            false,
        )))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        Ok(crate::anon::Fn1(v, std::marker::PhantomData))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        self.0
    }
}

impl<'gc, A, B, R> MimasType<'gc> for crate::anon::Fn2<'gc, A, B, R>
where
    A: MimasType<'gc>,
    B: MimasType<'gc>,
    R: MimasType<'gc>,
{
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        Some(Ty::Fn(FnHeader::new(
            vec![
                FnParam::new(None, A::mimas_ty(reg)?, false),
                FnParam::new(None, B::mimas_ty(reg)?, false),
            ],
            R::mimas_ty(reg)?,
            false,
        )))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        Ok(crate::anon::Fn2(v, std::marker::PhantomData))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        self.0
    }
}

impl<'gc, A, B, C, R> MimasType<'gc> for crate::anon::Fn3<'gc, A, B, C, R>
where
    A: MimasType<'gc>,
    B: MimasType<'gc>,
    C: MimasType<'gc>,
    R: MimasType<'gc>,
{
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        Some(Ty::Fn(FnHeader::new(
            vec![
                FnParam::new(None, A::mimas_ty(reg)?, false),
                FnParam::new(None, B::mimas_ty(reg)?, false),
                FnParam::new(None, C::mimas_ty(reg)?, false),
            ],
            R::mimas_ty(reg)?,
            false,
        )))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        Ok(crate::anon::Fn3(v, std::marker::PhantomData))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        self.0
    }
}

macro_rules! impl_tuple {
    () => {};
    ($head:ident $(, $tail:ident)*) => {
        impl<'gc, $head, $($tail),*> MimasType<'gc> for ($head, $($tail),*)
        where
            $head: MimasType<'gc>,
            $($tail: MimasType<'gc>,)*
        {
            fn mimas_ty(reg: &Registry) -> Option<Ty> {
                Some(Ty::Tuple(vec![
                    <$head as MimasType<'gc>>::mimas_ty(reg)?,
                    $(<$tail as MimasType<'gc>>::mimas_ty(reg)?,)*
                ]))
            }
            /// `(PxX, PxY)` claims `(px↔, px↕)` — only when every element has a
            /// dim to check; a partly-measured tuple stays unchecked rather
            /// than claim "plain number" for its unchecked slots.
            fn mimas_dim_tree(reg: &Registry) -> Option<DimTree> {
                Some(DimTree::Tup(vec![
                    <$head as MimasType<'gc>>::mimas_dim_tree(reg)?,
                    $(<$tail as MimasType<'gc>>::mimas_dim_tree(reg)?,)*
                ]))
            }
            #[allow(non_snake_case)]
            fn from_value(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
                let arr = v
                    .as_untyped_array(ctx)
                    .ok_or_else(|| ty_error("tuple", v))?;
                let items: Vec<Val<'gc>> = arr.0.borrow().iter().copied().collect();
                let mut it = items.into_iter();
                let $head = <$head as MimasType<'gc>>::from_value(
                    ctx,
                    it.next().ok_or_else(|| ty_error("tuple element", Val::Null))?,
                )?;
                $(
                    let $tail = <$tail as MimasType<'gc>>::from_value(
                        ctx,
                        it.next().ok_or_else(|| ty_error("tuple element", Val::Null))?,
                    )?;
                )*
                Ok(($head, $($tail),*))
            }
            #[allow(non_snake_case)]
            fn into_value(self, ctx: Ctx<'gc>) -> Val<'gc> {
                let ($head, $($tail),*) = self;
                let items = vec![
                    <$head as MimasType<'gc>>::into_value($head, ctx),
                    $(<$tail as MimasType<'gc>>::into_value($tail, ctx),)*
                ];
                ctx.array_val(items)
            }
        }
        impl_tuple!($($tail),*);
    };
}

impl_tuple!(A, B, C, D, E, F, G, H);

pub(crate) fn ty_error<'gc>(expected: &str, got: Val<'gc>) -> TypeError {
    TypeError {
        expected: expected.into(),
        got: format!("{got:?}"),
    }
}

/// One argument of a host call: any [`MimasType`], plus `&str`, which is only a `MimasType` for
/// the arena's own lifetime.
pub trait Arg {
    fn arg_ty(reg: &Registry) -> Option<Ty>;
    fn into_arg<'gc>(self, ctx: Ctx<'gc>) -> Val<'gc>;
}

impl<T: for<'gc> MimasType<'gc>> Arg for T {
    fn arg_ty(reg: &Registry) -> Option<Ty> {
        T::mimas_ty(reg)
    }

    fn into_arg<'gc>(self, ctx: Ctx<'gc>) -> Val<'gc> {
        self.into_value(ctx)
    }
}

impl Arg for &str {
    fn arg_ty(_: &Registry) -> Option<Ty> {
        Some(Ty::Str)
    }

    fn into_arg<'gc>(self, ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Str(ctx.intern(self))
    }
}

/// Passing back a value the host was handed earlier, which is the only way to reach one whose type
/// only exists in the script -- a struct it declared, say.
impl Arg for &crate::Stashed {
    fn arg_ty(_: &Registry) -> Option<Ty> {
        None
    }

    fn into_arg<'gc>(self, ctx: Ctx<'gc>) -> Val<'gc> {
        ctx.fetch(self)
    }
}

/// The arguments of a host call into a script, as a tuple of [`Arg`]s: `()`, `(a,)`, `(a, b)`,
/// up to eight.
pub trait Args {
    fn tys(&self, reg: &Registry) -> Vec<Option<Ty>>;
    fn into_values<'gc>(self, ctx: Ctx<'gc>) -> Vec<Val<'gc>>;
}

macro_rules! impl_args {
    ($($t:ident),*) => {
        impl<$($t: Arg),*> Args for ($($t,)*) {
            fn tys(&self, _reg: &Registry) -> Vec<Option<Ty>> {
                vec![$($t::arg_ty(_reg)),*]
            }

            #[allow(non_snake_case)]
            fn into_values<'gc>(self, _ctx: Ctx<'gc>) -> Vec<Val<'gc>> {
                let ($($t,)*) = self;
                vec![$($t.into_arg(_ctx)),*]
            }
        }
    };
}

impl_args!();
impl_args!(A);
impl_args!(A, B);
impl_args!(A, B, C);
impl_args!(A, B, C, D);
impl_args!(A, B, C, D, E);
impl_args!(A, B, C, D, E, F);
impl_args!(A, B, C, D, E, F, G);
impl_args!(A, B, C, D, E, F, G, H);

impl<'gc, R: MimasType<'gc>> MimasType<'gc> for crate::anon::Fn0<'gc, R> {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        Some(Ty::Fn(FnHeader::new(vec![], R::mimas_ty(reg)?, false)))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        Ok(crate::anon::Fn0(v, std::marker::PhantomData))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        self.0
    }
}

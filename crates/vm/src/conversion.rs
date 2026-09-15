use std::collections::HashMap;

use api::Registry;
use shared::Ty;

use crate::{Array, Ctx, Dict, DictMap, Instance, RtErr, RtResult, Str, Val};

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
    fn from_value(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError>;
    fn into_value(self, ctx: Ctx<'gc>) -> Val<'gc>;
}

/// Lets native fns return either a plain `T: MimasType` (Ok-wrapped automatically) or a
/// `Result<T, E: Into<RtError>>` for runtime-error surfacing. The `IntoFn` macro goes through
/// this trait so users don't need to pick a different registration path just to be able to fail.
pub trait IntoNativeResult<'gc> {
    fn return_ty(reg: &Registry) -> Option<Ty>;
    fn into_native_result(self, ctx: Ctx<'gc>) -> RtResult<Val<'gc>>;
}

impl<'gc, T> IntoNativeResult<'gc> for T
where
    T: MimasType<'gc>,
{
    fn return_ty(reg: &Registry) -> Option<Ty> {
        <T as MimasType<'gc>>::mimas_ty(reg)
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

    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        Ok(v.as_str().ok_or_else(|| ty_error("str", v))?.as_str())
    }

    fn into_value(self, ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Str(ctx.intern(self))
    }
}

impl<'gc> MimasType<'gc> for Array<'gc> {
    fn mimas_ty(_: &Registry) -> Option<Ty> {
        Some(Ty::Array(Box::new(Ty::Anon(0))))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_array().ok_or_else(|| ty_error("array", v))
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
        Some(Ty::Adt(reg.get::<DataFrameTy>()?.adt_id))
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
        Some(Ty::Adt(reg.get::<PlExprTy>()?.adt_id))
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
        Some(Ty::Adt(reg.get::<GroupByTy>()?.adt_id))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_group_by().ok_or_else(|| ty_error("GroupBy", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::GroupBy(self)
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
        Some(Ty::Adt(reg.get::<DarklyImageTy>()?.adt_id))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_darkly_image().ok_or_else(|| ty_error("DarklyImage", v))
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
    fn from_value(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        match v {
            Val::Raised(s) => Ok(Raisable::Raised(s.as_str().to_string())),
            other => Ok(Raisable::Ok(T::from_value(ctx, other)?)),
        }
    }
    fn into_value(self, ctx: Ctx<'gc>) -> Val<'gc> {
        match self {
            Raisable::Ok(v) => v.into_value(ctx),
            Raisable::Raised(s) => Val::Raised(ctx.intern(&s)),
        }
    }
}

// Mutating natives should take `Array<'gc>` / `Dict<'gc>` directly so they can use the heap
// newtype's `borrow_mut(&ctx)`. These collection impls just copy for ergonomic non-mutating
// reads.

impl<'gc, T: MimasType<'gc>> MimasType<'gc> for Vec<T> {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        T::mimas_ty(reg).map(|i| Ty::Array(Box::new(i)))
    }
    fn from_value(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        let arr = v.as_array().ok_or_else(|| ty_error("array", v))?;
        let items: Vec<Val<'gc>> = arr.0.borrow().iter().copied().collect();
        items.into_iter().map(|v| T::from_value(ctx, v)).collect()
    }
    fn into_value(self, ctx: Ctx<'gc>) -> Val<'gc> {
        let out: Vec<Val<'gc>> = self.into_iter().map(|v| v.into_value(ctx)).collect();
        Val::Array(ctx.new_array(out))
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
        Some(Ty::Array(Box::new(Ty::Anon(N))))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        v.as_array()
            .map(crate::anon::ArrayOf)
            .ok_or_else(|| ty_error("array", v))
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        Val::Array(self.0)
    }
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
            #[allow(non_snake_case)]
            fn from_value(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
                let arr = v.as_array().ok_or_else(|| ty_error("tuple", v))?;
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
                Val::Array(ctx.new_array(items))
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

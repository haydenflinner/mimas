//! Invertible transforms — Mimas's slice of Uiua's `un`/`under` machinery.
//!
//! An `Iso` is an opaque value holding a pair of callables, built two ways:
//!
//!   * `iso(to, from)` — an isomorphism: `to` maps a value into a new space and
//!     `from` maps it back. `un(iso)` hands back `from`, and `under(iso, f, x)`
//!     runs `from(f(to(x)))`.
//!   * `lens(do, undo)` — a focus/update pair: `do` splits a source into
//!     `(focus, ctx)` and `undo(ctx, focus)` rebuilds a source. `under` runs `f`
//!     on the focus and recombines (`under(at(2), f, xs)` edits `xs[2]`).
//!
//! Signatures carry the transform's types — `iso(to, from): Iso<S, F>` reads
//! "S ⇄ F" — so `under`'s `f` checks against the focus type rather than a fresh
//! inference var. The adt itself is monomorphic; the type args ride `Ty::Adt`'s
//! argument list, which unifies pairwise. There is no `Iso<S, F>` annotation
//! syntax: the args only ever come from these signatures.
//!
//! A native can't call back into the VM, so `under` and `at` are `#[native]`s
//! only for their signatures — codegen lowers them (`Intrinsic::Under`/`At`).
//! `un` is a real native: it hands the stored inverse back as a value and the
//! caller's own call invokes it.
//!
//! Instance layout — fields are positional and solver-invisible (the adt
//! declares `Unit` shape, same trick as `DataFrameTy`):
//!   [0] kind: 0 = iso | 1 = lens
//!   [1] a:    to      | do
//!   [2] b:    from    | undo

use std::any::TypeId;
use std::marker::PhantomData;

use api::{ApiAdtKind, ApiVariantFields, Intrinsic, Registry};
use macros::native;
use shared::Ty;
use vm::{
    Ctx, Fields, Instance, RtErr, RtResult, Val,
    adt::{ApiAdtDescriptor, ApiVariantShape, InstanceOf, MimasAdt},
    anon,
    api::Api,
    conversion::{MimasType, TypeError},
};

/// Marker for the `Iso` adt — like `DataFrameTy`, it exists so `Ty::Adt` has
/// something to hang off; the runtime value is an `Instance` laid out as above.
pub(crate) struct IsoTy;

impl MimasAdt for IsoTy {
    fn descriptor(_reg: &Registry) -> ApiAdtDescriptor {
        ApiAdtDescriptor {
            name: "Iso",
            module: &[],
            kind: ApiAdtKind::Struct,
            doc: "An invertible transform: a forward callable and the inverse/update that maps a result back into source space. Build with `iso(to, from)` or `lens(do, undo)`; run it backwards with `un` or edit through it with `under`.",
            variants: vec![ApiVariantShape {
                name: "Iso".into(),
                doc: "",
                fields: ApiVariantFields::Unit,
            }],
        }
    }
}

impl<'gc> MimasType<'gc> for IsoTy {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        Some(Ty::adt(reg.get::<IsoTy>()?.adt_id))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        match v {
            Val::Instance(_) => Ok(IsoTy),
            other => Err(TypeError {
                expected: "Iso".into(),
                got: format!("{other:?}"),
            }),
        }
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        unreachable!("IsoTy is a marker -- `iso`/`lens`/`at` build the Instance")
    }
}

/// Signature-facing `Iso<S, F>` — phantom over the source/focus types so an
/// `iso(to, from)` call's result type remembers `S ⇄ F` and `under` can check
/// `f` against `F`. Resolves the family's single adt through `IsoTy`'s TypeId.
pub(crate) struct Iso<S, F>(PhantomData<fn(S) -> F>);

impl<'gc, S: MimasType<'gc>, F: MimasType<'gc>> MimasType<'gc> for Iso<S, F> {
    fn mimas_ty(reg: &Registry) -> Option<Ty> {
        Some(Ty::Adt(
            reg.get::<IsoTy>()?.adt_id,
            vec![S::mimas_ty(reg)?, F::mimas_ty(reg)?],
        ))
    }
    fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
        match v {
            Val::Instance(_) => Ok(Iso(PhantomData)),
            other => Err(TypeError {
                expected: "Iso".into(),
                got: format!("{other:?}"),
            }),
        }
    }
    fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
        unreachable!("Iso is a sig marker -- `InstanceOf` carries the handle")
    }
}

/// `Iso`'s runtime `struct_id`, resolved through the registration binding.
fn iso_struct_id<'gc>(ctx: Ctx<'gc>) -> Option<u32> {
    ctx.binding(TypeId::of::<IsoTy>())
        .map(|b| u32::from(b.adt_id))
}

fn make_iso<'gc>(ctx: Ctx<'gc>, kind: i64, a: Val<'gc>, b: Val<'gc>) -> RtResult<Instance<'gc>> {
    let Some(sid) = iso_struct_id(ctx) else {
        return Err(RtErr::Custom(
            "Iso isn't registered -- install library::std first".into(),
        ));
    };
    Ok(ctx.new_instance(sid, Fields::new(vec![Val::Int(kind), a, b])))
}

/// `iso(to, from)` — declare `to` and `from` as each other's inverse.
#[native]
fn iso<'gc>(
    ctx: Ctx<'gc>,
    to: anon::Fn1<'gc, anon::T<'gc>, anon::U<'gc>>,
    from: anon::Fn1<'gc, anon::U<'gc>, anon::T<'gc>>,
) -> RtResult<InstanceOf<'gc, Iso<anon::T<'gc>, anon::U<'gc>>>> {
    make_iso(ctx, 0, to.0, from.0).map(InstanceOf::new)
}

/// `lens(do, undo)` — declare a focus/update pair for `under`. `do` takes the
/// source and returns `(focus, ctx)`; `undo(ctx, focus)` rebuilds the source
/// from the original context and an edited focus. The `at` intrinsic's lowered
/// code `call_native`s this with its synthesized closures.
#[native]
fn lens<'gc>(
    ctx: Ctx<'gc>,
    r#do: anon::Fn1<'gc, anon::T<'gc>, (anon::U<'gc>, anon::V<'gc>)>,
    undo: anon::Fn2<'gc, anon::V<'gc>, anon::U<'gc>, anon::T<'gc>>,
) -> RtResult<InstanceOf<'gc, Iso<anon::T<'gc>, anon::U<'gc>>>> {
    make_iso(ctx, 1, r#do.0, undo.0).map(InstanceOf::new)
}

/// `un(iso)` — the inverse direction of an `iso(to, from)`, as a callable.
/// `un(iso)(x)` is `from(x)`; a lens has no inverse and errors instead.
#[native]
fn un<'gc>(
    ctx: Ctx<'gc>,
    v: InstanceOf<'gc, Iso<anon::T<'gc>, anon::U<'gc>>>,
) -> RtResult<anon::Fn1<'gc, anon::U<'gc>, anon::T<'gc>>> {
    let no_inverse = || {
        RtErr::InvalidArgument(
            "un: this value has no declared inverse -- build one with `iso(to, from)`".into(),
        )
    };
    let Some(sid) = iso_struct_id(ctx) else {
        return Err(no_inverse());
    };
    let inst = v.0.0.borrow();
    if inst.struct_id != sid || inst.fields.len() != 3 {
        return Err(no_inverse());
    }
    match inst.fields[0] {
        Val::Int(0) => Ok(anon::Fn1(inst.fields[2], PhantomData)),
        _ => Err(RtErr::InvalidArgument(
            "un: a lens has no inverse -- edit through it with `under` instead".into(),
        )),
    }
}

/// `deep_clone(v)` — a structural copy sharing no mutable state with `v`. Used
/// by `at`'s undo so `under` never mutates the source it was handed; exposed
/// because "copy this collection" is useful on its own.
#[native]
fn deep_clone<'gc>(ctx: Ctx<'gc>, v: anon::T<'gc>) -> anon::T<'gc> {
    anon::Anon(ctx.deep_clone(v.0))
}

/// `under(iso, f, x)` / `under(iso, f)` — Uiua's `⍜`. Intrinsic: codegen emits
/// the transform+edit+rebuild sequence itself (and a `|x| ...` closure for the
/// curried form).
#[native]
fn under<'gc>(
    _iso: InstanceOf<'gc, Iso<anon::T<'gc>, anon::U<'gc>>>,
    _f: anon::Fn1<'gc, anon::U<'gc>, anon::U<'gc>>,
    _x: Option<anon::T<'gc>>,
) -> Val<'gc> {
    unreachable!("intrinsics cannot be reached")
}

/// `at(i)` — the index lens for `under`: `Iso<[V], V>` over an array. Intrinsic:
/// codegen synthesizes the do/undo closures capturing `i` (do splits `src` into
/// `(src[i], src)`; undo rebuilds a cloned source with `i` set to the edited
/// focus).
#[native]
fn at<'gc>(_i: i64) -> InstanceOf<'gc, Iso<anon::ArrayOf<'gc, 5>, anon::Anon<'gc, 5>>> {
    unreachable!("intrinsics cannot be reached")
}

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    // adt first: `iso`/`lens`/`under` sigs mention `Iso`'s `Ty`.
    api.add_adt::<IsoTy>();
    api.add(iso);
    api.add(lens);
    api.add(un);
    api.add(deep_clone);
    let id = api.add(under);
    api.mark_intrinsic(id, Intrinsic::Under);
    let id = api.add(at);
    api.mark_intrinsic(id, Intrinsic::At);
}

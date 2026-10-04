use std::{any::TypeId, rc::Rc};

use api::{
    AdtBinding, ApiAdt, ApiAdtKind, ApiConstant, ApiFunction, ApiMethod, ApiVariant, Intrinsic,
    Library, NativeId, Registry,
};
use compile::{BinOp, UnaryOp};
use shared::{Fx, Literal, Ty};

use crate::{
    RtErr, RtResult, Val,
    adt::{ApiAdtDescriptor, MimasAdt},
    conversion::{IntoNativeResult, MimasType},
    heap::Ctx,
    native::{NativeRef, make_native},
    val::InstanceOps,
};

/// A single native-registration request, emitted by the `#[mimas]` attribute macro and
/// gathered at install time through the [`inventory`](crate::inventory) crate.
///
/// This is what allows the macro to both convert _and_ import your item. The key issue is that
/// there's obviously no way to find and mutate your VM at this stage, so instead we use the
/// inventory crate to collect a series of functions we generate per-impl in the macro that can be
/// called later to do the installation.
///
/// inventory makes **no guarantee** about the order in which submissions are visited. That
/// collides with the load-order rule in [`Api::add_adt`]: a Rust ADT must be registered before
/// any fn whose mimas signature mentions it. `phase` lets `library::std` run a deterministic
/// two-pass sweep -- ADTs ([`Self::PHASE_ADT`]) first, everything else ([`Self::PHASE_FN`])
/// second -- so that rule holds regardless of inventory's iteration order.
pub struct MimasReg {
    pub phase: u8,
    pub register: NativeFnReg,
}

impl MimasReg {
    pub const PHASE_ADT: u8 = 0;
    pub const PHASE_FN: u8 = 1;
}

inventory::collect!(MimasReg);

/// Metadata harvested from a `#[native]` / `#[mimas]` item -- its doc comment and the identifiers
/// of its declared parameters -- submitted via [`inventory`] and keyed by the item's full Rust
/// path (`concat!(module_path!(), "::", <ident>)`). The install path joins these onto the
/// [`ApiFunction`]/[`ApiMethod`] it builds by matching the path against
/// [`std::any::type_name_of_val`] of the registered fn, so the receiver/module -- known only at the
/// `api.add_*` call site, not the macro -- never has to travel with the doc.
///
/// `parameters` holds every declared parameter after `ctx`, including a method's receiver --
/// consumers skip the leading slots their `parameters`/`param_dims` vecs don't cover.
///
/// Like every inventory registry this is subject to the link-pruning footgun (a submission in an
/// unreferenced object file can be dropped under `codegen-units > 1`). That only affects the
/// doc-generation build, which we own and can pin to `codegen-units = 1`; missing metadata
/// degrades to `arg{i}` names and an empty doc, never a wrong signature.
pub struct NativeMeta {
    pub path: &'static str,
    pub parameters: &'static [&'static str],
    pub doc: &'static str,
    /// Indices into `parameters` (receiver counts) named by `#[consumes(...)]` --
    /// the grades pass kills the binding passed to one of these slots.
    pub consumes: &'static [usize],
    /// `#[must_use]` on the item -- discarding the return warns.
    pub must_use: bool,
}

inventory::collect!(NativeMeta);

/// Marks a parameter of a `#[native]` / `#[mimas]` fn as `&mut`, submitted via [`inventory`]
/// and keyed by the item's full Rust path exactly like [`NativeMeta`]. `index` counts from the
/// receiver: 0 is `self` (or the first param of a free fn), 1.. are the declared params.
///
/// Only `index == 0` is consumed today, joined onto [`ApiMethod::mutates_recv`] at
/// `install_method` time so the solver can reject mutating a collection mid-`for`. Mutating
/// *non-receiver* params (a `&mut` collection passed to a free fn) are recorded for the same
/// check to grow into later.
///
/// Subject to the same link-pruning footgun as [`NativeMeta`]: a pruned submission degrades to
/// `mutates_recv: false` -- it can only ever weaken the lint, never corrupt a signature.
pub struct NativeMutates {
    pub path: &'static str,
    pub index: usize,
}

inventory::collect!(NativeMutates);

/// A literal-call validator for a `#[native]` / `#[mimas]` fn, submitted via [`inventory`] and
/// keyed by the item's full Rust path exactly like [`NativeMeta`]. When every call argument is a
/// literal the solver runs `validate` on them: `Ok` proves the call can't raise so a `T!`
/// return narrows to `T`; `Err(msg)` becomes a compile error. See [`api::LitValidator`].
///
/// Subject to the same link-pruning footgun as [`NativeMeta`]: a pruned submission degrades to
/// `validate: None` -- the call just keeps its honest `T!`, never a wrong one.
pub struct NativeValidator {
    pub path: &'static str,
    pub validate: api::LitValidator,
}

inventory::collect!(NativeValidator);

/// A native's definition site (`file!()`/`line!()` at the item), submitted via [`inventory`]
/// and keyed by the item's full Rust path exactly like [`NativeMeta`]. The install path joins
/// it onto [`ApiFunction::src`]/[`ApiMethod::src`] so a host can link a built-in's symbol menu
/// straight to its source.
///
/// Subject to the same link-pruning footgun as [`NativeMeta`]: a pruned submission degrades to
/// `src: None` -- the menu just keeps its old "no source here" line.
pub struct NativeSrc {
    pub path: &'static str,
    /// `file!()` at the def site -- relative to the workspace root the crate compiled under.
    pub file: &'static str,
    /// `env!("CARGO_MANIFEST_DIR")` of the defining crate -- locates which workspace `file` is
    /// relative to.
    pub manifest: &'static str,
    pub line: u32,
}

inventory::collect!(NativeSrc);

/// A native's declared side-effect footprint, submitted via [`inventory`] and keyed by the
/// item's full Rust path exactly like [`NativeMutates`]. `#[effects(net, io)]` on a
/// `#[native]`/`#[mimas]` fn emits this; the install path joins it onto
/// [`ApiFunction::effects`]/[`ApiMethod::effects`]. `effects` packs [`shared::Fx`] bits.
///
/// Subject to the same link-pruning footgun as [`NativeMeta`]: a pruned submission degrades
/// to `effects: None`, which inference reads as `Fx::unknown()` -- fails closed, never
/// wrong.
pub struct NativeEffects {
    pub path: &'static str,
    pub effects: u8,
}

inventory::collect!(NativeEffects);

pub type NativeFnReg = for<'a, 'gc> fn(&mut Api<'a, 'gc>);

pub struct Api<'a, 'gc> {
    pub ctx: Ctx<'gc>,
    pub library: &'a mut Library<()>,
}

/// The inventory registries are static once the binary links, but each
/// install asked them four linear questions per registered fn -- an
/// O(n²) scan over every `#[mimas]` submission, repeated on every Vm
/// spawn (a fuzz sweep spawns thousands). Index each registry once per
/// process instead.
static META_BY_PATH: std::sync::LazyLock<
    std::collections::HashMap<&'static str, &'static NativeMeta>,
> = std::sync::LazyLock::new(|| {
    inventory::iter::<NativeMeta>
        .into_iter()
        .map(|m| (m.path, m))
        .collect()
});
static SRC_BY_PATH: std::sync::LazyLock<
    std::collections::HashMap<&'static str, &'static NativeSrc>,
> = std::sync::LazyLock::new(|| {
    inventory::iter::<NativeSrc>
        .into_iter()
        .map(|s| (s.path, s))
        .collect()
});
static MUTATES_RECV: std::sync::LazyLock<std::collections::HashSet<&'static str>> =
    std::sync::LazyLock::new(|| {
        inventory::iter::<NativeMutates>
            .into_iter()
            .filter(|m| m.index == 0)
            .map(|m| m.path)
            .collect()
    });
static VALIDATOR_BY_PATH: std::sync::LazyLock<
    std::collections::HashMap<&'static str, api::LitValidator>,
> = std::sync::LazyLock::new(|| {
    inventory::iter::<NativeValidator>
        .into_iter()
        .map(|v| (v.path, v.validate))
        .collect()
});
static EFFECTS_BY_PATH: std::sync::LazyLock<std::collections::HashMap<&'static str, Fx>> =
    std::sync::LazyLock::new(|| {
        inventory::iter::<NativeEffects>
            .into_iter()
            .map(|e| (e.path, Fx::from_bits_truncate(e.effects)))
            .collect()
    });

/// Look up what `#[native]`/`#[mimas]` submitted for the fn at `path` (`type_name_of_val(&f)`)
/// and pair `arity` slots with their declared names, dropping the `skip` leading ones the arity
/// doesn't cover (a method's receiver). Missing or mismatched submissions degrade to `arg{i}`
/// names and an empty doc (see [`NativeMeta`]).
/// `#[native]` submits `module::name` from inside the fn body, but a method on
/// `impl T` registers under `type_name`'s `module::T::name` -- on a miss, retry
/// with the penultimate (impl-name) segment out.
fn meta_entry(path: &str) -> Option<&'static NativeMeta> {
    META_BY_PATH.get(path).copied().or_else(|| {
        let (rest, last) = path.rsplit_once("::")?;
        let (head, _) = rest.rsplit_once("::")?;
        let stripped = format!("{head}::{last}");
        META_BY_PATH.get(stripped.as_str()).copied()
    })
}

fn meta_for(path: &str, skip: usize, arity: usize) -> (String, Vec<String>) {
    let meta = meta_entry(path);
    let names = meta
        .and_then(|m| m.parameters.get(skip..))
        .filter(|names| names.len() == arity);
    let param_names = (0..arity)
        .map(|i| names.map_or_else(|| format!("arg{i}"), |names| names[i].to_string()))
        .collect();
    let doc = meta.map(|m| m.doc.to_string()).unwrap_or_default();
    (doc, param_names)
}

/// The def-site `(file, manifest, line)` submitted for `path` -- see [`NativeSrc`].
fn src_for(path: &str) -> Option<api::NativeSrc> {
    SRC_BY_PATH.get(path).map(|s| (s.file, s.manifest, s.line))
}

/// Whether `path` was submitted as mutating its receiver -- see [`NativeMutates`].
fn mutates_recv(path: &str) -> bool {
    MUTATES_RECV.contains(path)
}

/// The literal-call validator for `path` if one was submitted -- see [`NativeValidator`].
fn validator_for(path: &str) -> Option<api::LitValidator> {
    VALIDATOR_BY_PATH.get(path).copied()
}

/// The declared effect set for `path` if one was submitted -- see [`NativeEffects`]. `None`
/// means unannotated: effect inference reads the call as `Fx::unknown()`.
fn effects_for(path: &str) -> Option<Fx> {
    EFFECTS_BY_PATH.get(path).copied()
}

/// `(consumes, consumes_recv, must_use)` for `path`: the `consumes` indices of
/// [`NativeMeta`] re-based by `skip` onto the entry's `parameters` (indices
/// under `skip` -- a method's receiver at 0 -- come back as `consumes_recv`).
/// Missing meta degrades to all-false -- never a consume check that was never
/// declared.
fn grades_for(path: &str, skip: usize, arity: usize) -> (Vec<bool>, bool, bool) {
    let Some(meta) = meta_entry(path) else {
        return (vec![false; arity], false, false);
    };
    let mut consumes = vec![false; arity];
    let mut consumes_recv = false;
    for &i in meta.consumes {
        match i.checked_sub(skip) {
            Some(j) => {
                if let Some(slot) = consumes.get_mut(j) {
                    *slot = true;
                }
            }
            None => consumes_recv = true,
        }
    }
    (consumes, consumes_recv, meta.must_use)
}

impl<'a, 'gc> Api<'a, 'gc> {
    pub fn add<F, Marker>(&mut self, f: F) -> NativeId
    where
        F: IntoFn<'gc, Marker>,
    {
        let name = short_name_of_val(&f);
        f.install(self, name, Vec::new())
    }

    pub fn add_named<F, Marker>(&mut self, name: impl Into<String>, f: F)
    where
        F: IntoFn<'gc, Marker>,
    {
        f.install(self, name.into(), Vec::new());
    }

    pub fn add_method<F, Marker>(&mut self, f: F) -> NativeId
    where
        F: IntoMethod<'gc, Marker>,
    {
        let name = short_name_of_val(&f);
        f.install_method(self, name)
    }

    pub fn add_method_named<F, Marker>(&mut self, name: impl Into<String>, f: F)
    where
        F: IntoMethod<'gc, Marker>,
    {
        f.install_method(self, name.into());
    }

    /// Associated functions don't carry a `self` arg the macro can introspect, so the receiver
    /// `Ty` (the builtin's namespace -- `Ty::Int`, `Ty::Bool`, etc.) is passed explicitly.
    pub fn add_assoc<F, Marker>(&mut self, recv_ty: Ty, f: F)
    where
        F: IntoFn<'gc, Marker>,
    {
        let name = short_name_of_val(&f);
        f.install_assoc(self, recv_ty, name);
    }

    /// The mimas `Ty` of a registered Rust type -- the receiver arg for [`Self::add_assoc`].
    /// Panics if `add_adt::<T>()` hasn't run yet.
    pub fn ty_of<T: MimasType<'gc>>(&self) -> Ty {
        T::mimas_ty(self.library.registry()).expect("ty_of requires the type be registered first")
    }

    /// [`Self::add_assoc`] with the receiver resolved from a registered adt -- what `#[mimas]
    /// impl` uses for `self`-less fns, whose generated shim name isn't the mimas-facing one.
    pub fn add_assoc_of<T, F, Marker>(&mut self, name: impl Into<String>, f: F)
    where
        T: MimasType<'gc>,
        F: IntoFn<'gc, Marker>,
    {
        let recv_ty =
            T::mimas_ty(self.library.registry()).expect("assoc receiver must be a registered adt");
        f.install_assoc(self, recv_ty, name.into());
    }

    /// `a <op> b` for `T` instances — `f` receives both operands as `Val`s
    /// in source order plus the `op`, and declines shapes/ops it doesn't
    /// handle via `RtErr::invalid_bin`. Dispatch consults the LHS's type
    /// first, then the RHS's, so one impl can cover `v * s` and `s * v`.
    /// `==`/`!=` can't be overridden — instance equality is structural and
    /// is handled before this dispatch.
    ///
    /// Panics if `add_adt::<T>()` hasn't run yet.
    pub fn add_bin_op<T, F>(&mut self, f: F)
    where
        T: MimasType<'gc> + 'static,
        F: for<'g> Fn(Ctx<'g>, Val<'g>, Val<'g>, BinOp) -> RtResult<Val<'g>> + 'static,
    {
        let f = Rc::new(f);
        let ops = &self.ctx.fixture::<InstanceOps>().bin;
        for sid in self.layout_ids::<T>("add_bin_op") {
            ops.borrow_mut().insert(sid, f.clone());
        }
        self.mark_op_overloads::<T>();
    }

    /// `-a`/`+a`/`!a`/`~a` for `T` instances — see [`Self::add_bin_op`];
    /// decline via `RtErr::InvalidUnaryOperand`.
    ///
    /// Panics if `add_adt::<T>()` hasn't run yet.
    pub fn add_unary_op<T, F>(&mut self, f: F)
    where
        T: MimasType<'gc> + 'static,
        F: for<'g> Fn(Ctx<'g>, Val<'g>, UnaryOp) -> RtResult<Val<'g>> + 'static,
    {
        let f = Rc::new(f);
        let ops = &self.ctx.fixture::<InstanceOps>().unary;
        for sid in self.layout_ids::<T>("add_unary_op") {
            ops.borrow_mut().insert(sid, f.clone());
        }
        self.mark_op_overloads::<T>();
    }

    /// Flag the adt so the solver accepts infix/unary ops on it — the
    /// operand shapes themselves stay a runtime concern of the impl.
    fn mark_op_overloads<T: MimasType<'gc> + 'static>(&mut self) {
        let adt = self.library.registry().get::<T>().expect("adt").adt_id;
        if let Some(a) = self.library.adt_mut(adt) {
            a.op_overloads = true;
        }
    }

    /// The `struct_id`s an instance of `T` can carry — one for structs, one
    /// per variant for enums.
    fn layout_ids<T: MimasType<'gc> + 'static>(&self, api: &str) -> Vec<u32> {
        self.library
            .registry()
            .get::<T>()
            .unwrap_or_else(|| panic!("{api} requires add_adt::<T>() first"))
            .variant_layout_ids
            .iter()
            .map(|id| u32::from(*id))
            .collect()
    }

    pub fn mark_intrinsic(&mut self, nid: NativeId, i: Intrinsic) {
        self.library.mark_instrinsic(nid, i);
    }

    // must run before any `add(fn)` whose signature mentions `T`, or any add_adt::<U>
    // whose fields reference `T` -- field types resolve eagerly through the registry.
    //
    // todo, make this order independent
    pub fn add_adt<T: MimasAdt>(&mut self) {
        self.add_adt_in(TypeId::of::<T>(), None, T::descriptor);
    }

    /// Registers a type whose shape is only known at runtime, such as one described through
    /// reflection. `type_id` keys it the way [`Self::add_adt`] keys a Rust type.
    pub fn add_adt_described(
        &mut self,
        type_id: TypeId,
        describe: impl FnOnce(&Registry) -> ApiAdtDescriptor,
    ) -> AdtBinding {
        self.add_adt_in(type_id, None, describe)
    }

    fn add_adt_in(
        &mut self,
        type_id: TypeId,
        module: Option<Vec<String>>,
        describe: impl FnOnce(&Registry) -> ApiAdtDescriptor,
    ) -> AdtBinding {
        let adt_id = self.library.registry_mut().alloc();
        // pre-bind so the type can be self-referential
        self.library.registry_mut().bind_id(
            type_id,
            AdtBinding {
                adt_id,
                variant_layout_ids: Vec::new(),
            },
        );
        let desc = describe(self.library.registry());
        let variant_layout_ids: Vec<_> = match desc.kind {
            ApiAdtKind::Enum => desc
                .variants
                .iter()
                .map(|_| self.library.registry_mut().alloc())
                .collect(),
            ApiAdtKind::Struct => vec![adt_id],
        };
        let binding = AdtBinding {
            adt_id,
            variant_layout_ids: variant_layout_ids.clone(),
        };
        self.library
            .registry_mut()
            .bind_id(type_id, binding.clone());

        let variants = desc
            .variants
            .into_iter()
            .zip(variant_layout_ids.iter().copied())
            .map(|(shape, layout_id)| ApiVariant {
                name: shape.name,
                layout_id,
                doc: shape.doc.to_string(),
                fields: shape.fields,
            })
            .collect();

        let module =
            module.unwrap_or_else(|| desc.module.iter().map(|s| (*s).to_string()).collect());
        self.library.push_adt(ApiAdt {
            name: desc.name.to_string(),
            module,
            kind: desc.kind,
            adt_id,
            doc: desc.doc.to_string(),
            variants,
            op_overloads: false,
        });

        let mut bindings = self.ctx.state().mimas_bindings.borrow_mut(&self.ctx);
        bindings.0.insert(type_id, binding.clone());
        binding
    }

    /// An associated fn whose signature is only known at runtime. `call` receives the arguments
    /// in order, and it has to be `'static`, so it can't hold on to anything the Vm collects.
    pub fn add_assoc_described(
        &mut self,
        recv_ty: Ty,
        name: impl Into<String>,
        parameters: Vec<Ty>,
        return_ty: Ty,
        call: impl for<'g> Fn(Ctx<'g>, &[Val<'g>]) -> RtResult<Val<'g>> + 'static,
    ) {
        let native = make_native(&self.ctx, move |ctx, args| call(ctx, args));
        let id = self.library.method(ApiMethod {
            recv_ty,
            name: name.into(),
            param_dims: vec![None; parameters.len()],
            param_names: (0..parameters.len()).map(|i| format!("arg{i}")).collect(),
            parameters: parameters.into_iter().map(Some).collect(),
            return_ty: Some(return_ty),
            return_dim: None,
            takes_self: false,
            mutates_recv: false,
            // described fns have no Rust path to join effects metadata on -- unaudited
            effects: None,
            consumes: Vec::new(),
            consumes_recv: false,
            must_use: false,
            doc: String::new(),
            validate: None,
            src: None,
            call: (),
        });
        self.store_native(id, native);
    }

    /// Prelude-level constant (no module). The module-scoped counterpart is
    /// [`ModuleApi::constant`].
    pub fn constant(
        &mut self,
        name: impl Into<String>,
        ty: Ty,
        value: Literal,
        doc: impl Into<String>,
    ) {
        self.library.constant(ApiConstant {
            name: name.into(),
            module: Vec::new(),
            recv_ty: None,
            ty,
            value,
            doc: doc.into(),
        });
    }

    /// An associated constant (`Player::MAX_HEALTH`) -- what `#[mimas] impl` uses for consts
    /// in the block.
    pub fn assoc_constant(
        &mut self,
        recv_ty: Ty,
        name: impl Into<String>,
        ty: Ty,
        value: Literal,
        doc: impl Into<String>,
    ) {
        self.library.constant(ApiConstant {
            name: name.into(),
            module: Vec::new(),
            recv_ty: Some(recv_ty),
            ty,
            value,
            doc: doc.into(),
        });
    }

    pub fn module<'b>(&'b mut self, path: impl Into<String>) -> ModuleApi<'b, 'a, 'gc> {
        let path: Vec<String> = path.into().split("::").map(String::from).collect();
        ModuleApi { parent: self, path }
    }

    fn store_native(&self, id: NativeId, native: NativeRef<'gc>) {
        let mut natives = self.ctx.state().natives.borrow_mut(&self.ctx);
        if natives.len() <= id.index() {
            natives.resize(id.index() + 1, None);
        }
        natives[id.index()] = Some(native);
    }
}

pub struct ModuleApi<'b, 'a, 'gc> {
    parent: &'b mut Api<'a, 'gc>,
    path: Vec<String>,
}

impl<'b, 'a, 'gc> ModuleApi<'b, 'a, 'gc> {
    pub fn add<F, Marker>(&mut self, f: F) -> NativeId
    where
        F: IntoFn<'gc, Marker>,
    {
        let name = short_name_of_val(&f);
        f.install(self.parent, name, self.path.clone())
    }

    pub fn add_named<F, Marker>(&mut self, name: impl Into<String>, f: F)
    where
        F: IntoFn<'gc, Marker>,
    {
        f.install(self.parent, name.into(), self.path.clone());
    }

    pub fn add_adt<T: MimasAdt>(&mut self) {
        self.parent
            .add_adt_in(TypeId::of::<T>(), Some(self.path.clone()), T::descriptor);
    }

    /// A module fn whose signature is only known at runtime, like [`Api::add_assoc_described`].
    /// A `None` parameter takes any value, the way `print` does.
    /// Returns the [`NativeId`] so the caller can `mark_intrinsic` it.
    pub fn add_described(
        &mut self,
        name: impl Into<String>,
        parameters: Vec<Option<Ty>>,
        return_ty: Ty,
        call: impl for<'g> Fn(Ctx<'g>, &[Val<'g>]) -> RtResult<Val<'g>> + 'static,
    ) -> NativeId {
        let native = make_native(&self.parent.ctx, move |ctx, args| call(ctx, args));
        let id = self.parent.library.function(ApiFunction {
            name: name.into(),
            module: self.path.clone(),
            param_dims: vec![None; parameters.len()],
            param_names: (0..parameters.len()).map(|i| format!("arg{i}")).collect(),
            parameters,
            return_ty: Some(return_ty),
            return_dim: None,
            // described fns have no Rust path to join effects metadata on -- unaudited
            effects: None,
            consumes: Vec::new(),
            must_use: false,
            doc: String::new(),
            validate: None,
            src: None,
            call: (),
        });
        self.parent.store_native(id, native);
        id
    }

    pub fn constant(
        &mut self,
        name: impl Into<String>,
        ty: Ty,
        value: Literal,
        doc: impl Into<String>,
    ) {
        self.parent.library.constant(ApiConstant {
            name: name.into(),
            module: self.path.clone(),
            recv_ty: None,
            ty,
            value,
            doc: doc.into(),
        });
    }
}

// `Marker` is a phantom tuple (e.g. `(A, B)`) that disambiguates which arity-impl matches
// a given fn item. Rust picks the impl whose Fn-signature aligns with F's; without the
// marker type param the per-arity impls would all collide on `impl IntoFn<'gc> for F`.

pub trait IntoFn<'gc, Marker>: Copy + 'static {
    fn install(self, api: &mut Api<'_, 'gc>, name: String, module: Vec<String>) -> NativeId;

    fn install_assoc(self, api: &mut Api<'_, 'gc>, recv_ty: Ty, name: String);
}

pub trait IntoMethod<'gc, Marker>: Copy + 'static {
    fn install_method(self, api: &mut Api<'_, 'gc>, name: String) -> NativeId;
}

macro_rules! impl_into_fn {
    ($($arg:ident),*) => {
        impl<'gc, F, R $(, $arg)*> IntoFn<'gc, ($($arg,)*)> for F
        where
            F: Fn(Ctx<'gc> $(, $arg)*) -> R + Copy + 'static,
            $($arg: MimasType<'gc>,)*
            R: IntoNativeResult<'gc>,
        {
            #[allow(non_snake_case, unused_variables, unused_mut)]
            fn install(self, api: &mut Api<'_, 'gc>, name: String, module: Vec<String>) -> NativeId {
                let reg = api.library.registry();
                let parameters = vec![$(<$arg as MimasType<'gc>>::mimas_ty(reg),)*];
                let param_dims = vec![$(<$arg as MimasType<'gc>>::mimas_dim(reg),)*];
                let return_ty = <R as IntoNativeResult<'gc>>::return_ty(reg);
                let return_dim = <R as IntoNativeResult<'gc>>::return_dim(reg);
                let (doc, param_names) = meta_for(
                    std::any::type_name_of_val(&self),
                    0,
                    parameters.len(),
                );
                let src = src_for(std::any::type_name_of_val(&self));
                let (consumes, _, must_use) = grades_for(
                    std::any::type_name_of_val(&self),
                    0,
                    parameters.len(),
                );
                let native = make_native(&api.ctx, move |ctx, args| {
                    let mut it = args.iter().copied();
                    $(let $arg = <$arg as MimasType<'gc>>::from_value(
                        ctx,
                        it.next().expect("native called with too few args"),
                    ).map_err(RtErr::from)?;)*
                    self(ctx $(, $arg)*).into_native_result(ctx)
                });
                let id = api.library.function(ApiFunction {
                    name,
                    module,
                    parameters,
                    param_dims,
                    param_names,
                    return_ty,
                    return_dim,
                    effects: effects_for(std::any::type_name_of_val(&self)),
                    consumes,
                    must_use,
                    doc,
                    validate: validator_for(std::any::type_name_of_val(&self)),
                    src,
                    call: (),
                });
                api.store_native(id, native);
                id
            }

            #[allow(non_snake_case, unused_variables, unused_mut)]
            fn install_assoc(self, api: &mut Api<'_, 'gc>, recv_ty: Ty, name: String) {
                let reg = api.library.registry();
                let parameters = vec![$(<$arg as MimasType<'gc>>::mimas_ty(reg),)*];
                let param_dims = vec![$(<$arg as MimasType<'gc>>::mimas_dim(reg),)*];
                let return_ty = <R as IntoNativeResult<'gc>>::return_ty(reg);
                let return_dim = <R as IntoNativeResult<'gc>>::return_dim(reg);
                let (doc, param_names) = meta_for(
                    std::any::type_name_of_val(&self),
                    0,
                    parameters.len(),
                );
                let src = src_for(std::any::type_name_of_val(&self));
                let (consumes, consumes_recv, must_use) = grades_for(
                    std::any::type_name_of_val(&self),
                    0,
                    parameters.len(),
                );
                let native = make_native(&api.ctx, move |ctx, args| {
                    let mut it = args.iter().copied();
                    $(let $arg = <$arg as MimasType<'gc>>::from_value(
                        ctx,
                        it.next().expect("native called with too few args"),
                    ).map_err(RtErr::from)?;)*
                    self(ctx $(, $arg)*).into_native_result(ctx)
                });
                let id = api.library.method(ApiMethod {
                    recv_ty,
                    name,
                    parameters,
                    param_dims,
                    param_names,
                    return_ty,
                    return_dim,
                    takes_self: false,
                    mutates_recv: false,
                    effects: effects_for(std::any::type_name_of_val(&self)),
                    consumes,
                    consumes_recv,
                    must_use,
                    doc,
                    validate: validator_for(std::any::type_name_of_val(&self)),
                    src,
                    call: (),
                });
                api.store_native(id, native);
            }
        }
    };
}

// something something more tuples
impl_into_fn!();
impl_into_fn!(A);
impl_into_fn!(A, B);
impl_into_fn!(A, B, C);
impl_into_fn!(A, B, C, D);
impl_into_fn!(A, B, C, D, E);
impl_into_fn!(A, B, C, D, E, F1);
impl_into_fn!(A, B, C, D, E, F1, G);
impl_into_fn!(A, B, C, D, E, F1, G, H);

macro_rules! impl_into_method {
    ($($arg:ident),*) => {
        impl<'gc, F, Recv, R $(, $arg)*> IntoMethod<'gc, (Recv, $($arg,)*)> for F
        where
            F: Fn(Ctx<'gc>, Recv $(, $arg)*) -> R + Copy + 'static,
            Recv: MimasType<'gc>,
            $($arg: MimasType<'gc>,)*
            R: IntoNativeResult<'gc>,
        {
            #[allow(non_snake_case, unused_variables, unused_mut)]
            fn install_method(self, api: &mut Api<'_, 'gc>, name: String) -> NativeId {
                let reg = api.library.registry();
                let recv_ty = <Recv as MimasType<'gc>>::mimas_ty(reg)
                    .expect("native method receiver must have a concrete Ty");
                let parameters = vec![$(<$arg as MimasType<'gc>>::mimas_ty(reg),)*];
                let param_dims = vec![$(<$arg as MimasType<'gc>>::mimas_dim(reg),)*];
                let return_ty = <R as IntoNativeResult<'gc>>::return_ty(reg);
                let return_dim = <R as IntoNativeResult<'gc>>::return_dim(reg);
                let (doc, param_names) = meta_for(
                    std::any::type_name_of_val(&self),
                    1,
                    parameters.len(),
                );
                let (consumes, consumes_recv, must_use) = grades_for(
                    std::any::type_name_of_val(&self),
                    1,
                    parameters.len(),
                );
                let native = make_native(&api.ctx, move |ctx, args| {
                    let mut it = args.iter().copied();
                    let recv = <Recv as MimasType<'gc>>::from_value(
                        ctx,
                        it.next().expect("native called without receiver"),
                    ).map_err(RtErr::from)?;
                    $(let $arg = <$arg as MimasType<'gc>>::from_value(
                        ctx,
                        it.next().expect("native called with too few args"),
                    ).map_err(RtErr::from)?;)*
                    self(ctx, recv $(, $arg)*).into_native_result(ctx)
                });
                let id = api.library.method(ApiMethod {
                    recv_ty,
                    name,
                    parameters,
                    param_dims,
                    param_names,
                    return_ty,
                    return_dim,
                    takes_self: true,
                    mutates_recv: mutates_recv(std::any::type_name_of_val(&self)),
                    effects: effects_for(std::any::type_name_of_val(&self)),
                    consumes,
                    consumes_recv,
                    must_use,
                    doc,
                    validate: validator_for(std::any::type_name_of_val(&self)),
                    src: src_for(std::any::type_name_of_val(&self)),
                    call: (),
                });
                api.store_native(id, native);
                id
            }
        }
    };
}

impl_into_method!();
impl_into_method!(A);
impl_into_method!(A, B);
impl_into_method!(A, B, C);
impl_into_method!(A, B, C, D);
impl_into_method!(A, B, C, D, E);
impl_into_method!(A, B, C, D, E, F1);
impl_into_method!(A, B, C, D, E, F1, G);

fn short_name_of_val<T: ?Sized>(value: &T) -> String {
    let full = std::any::type_name_of_val(value);
    full.rsplit("::").next().unwrap_or(full).to_string()
}

pub fn install_into<'gc, F>(ctx: Ctx<'gc>, install_fn: F) -> Library<()>
where
    F: FnOnce(&mut Api<'_, 'gc>),
{
    let mut library = Library::<()>::new();
    let mut api = Api {
        ctx,
        library: &mut library,
    };
    // register every `#[mimas]` item linked into the binary (from any crate), interleaved with the
    // caller's explicit install so dependencies resolve regardless of source:
    // 1. `#[mimas]` adts first -- a fn may name them.
    // 2. the caller's `install_fn` (e.g. library::std), which registers its own adts + fns in a
    //    self-consistent order.
    // 3. `#[mimas]` fns last -- so they can name both `#[mimas]` adts and anything `install_fn`
    //    registered (e.g. a std type).
    // (inventory iteration order is otherwise unspecified; the two phases come from MimasReg.)
    for reg in inventory::iter::<MimasReg> {
        if reg.phase == MimasReg::PHASE_ADT {
            (reg.register)(&mut api);
        }
    }
    install_fn(&mut api);
    for reg in inventory::iter::<MimasReg> {
        if reg.phase != MimasReg::PHASE_ADT {
            (reg.register)(&mut api);
        }
    }
    library
}

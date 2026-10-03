//! Proc-macros for authoring mimas natives from Rust: two attribute macros that split along
//! the conversion-vs-registration line, plus the `MimasEnum` / `MimasStruct` derives.
//!
//! Every native fn the VM calls has to be written in a particular shape (a `Ctx<'gc>` first
//! parameter, gc handles instead of borrowed slices -- see [`convert`]), and *separately* has
//! to be registered into a library so mimas code can find it:
//!
//! - [`macro@native`] does **conversion only**. You register the result yourself with an explicit
//!   `api.add_method(..)` / `api.add_assoc(..)` / `api.add(..)` call. This is what the standard
//!   library's builtin methods (e.g. `array.push`, `int.random`) use, because they attach to a
//!   *receiver `Ty`* the macro can't infer from tokens alone.
//!
//! - [`macro@mimas`] does **registration** (plus, for fns, the same conversion). Accepts a free fn,
//!   a struct / enum (emits the derive impls too -- don't also `#[derive]`), a const of a primitive
//!   type, or an inherent impl block (see [`methods`]). Bare `#[mimas]` targets the prelude,
//!   `#[mimas(foo::bar)]` a module path. Registration is deferred through the [`inventory`] crate
//!   -- see [`register`] and `vm::api::MimasReg`.

use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    DeriveInput, FnArg, Ident, Pat, Token, parse_macro_input, parse_quote, punctuated::Punctuated,
};

mod convert;
mod derive;
mod methods;
mod register;

/// Conversion only -- rewrites a `fn(Ctx<'gc>, ..)` into the shape the VM's `IntoFn` /
/// `IntoMethod` machinery accepts; you register it yourself. For a free fn that should just
/// land in the prelude, use [`macro@mimas`] instead and skip the manual `api.add`.
#[proc_macro_attribute]
pub fn native(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as syn::ItemFn);
    let src = src_submission(&input.sig.ident, input.sig.ident.span());
    let effects = match effects_submission(&input.sig.ident, &mut input.attrs) {
        Ok(effects) => effects,
        Err(e) => return e.to_compile_error().into(),
    };
    match convert::expand_conversion(&mut input) {
        Ok((_, mutating)) => {
            // `submit!` expands to an unnamed const, which an `impl` block rejects -- the meta
            // submission goes inside the fn body instead, so `#[native]` works on impl methods.
            let meta = meta_submission(
                &input.sig.ident,
                &param_names(&input.sig),
                &collect_doc(&input.attrs),
            );
            if let Some(meta) = meta {
                input.block.stmts.insert(0, meta);
            }
            let mutates = mutates_submissions(&input.sig.ident, &mutating);
            TokenStream::from(quote!(#input #src #mutates #effects))
        }
        Err(e) => e.to_compile_error().into(),
    }
}

/// Conversion **and** automatic registration -- no manual `api.add*` call.
///
/// ```ignore
/// #[mimas]                       // -> registered into the prelude
/// fn greet(ctx: Ctx<'_>, who: &str) -> String { format!("hi {who}") }
///
/// #[mimas(std::fs)]              // -> registered into the `std::fs` module
/// fn read(path: &str) -> Raisable<String> { /* .. */ }
///
/// #[mimas]                       // -> methods/assoc fns on a #[mimas] adt
/// impl Player { fn damage(&mut self, amount: i64) { /* .. */ } }
/// ```
///
/// There's no `Api` in scope at the definition site, so the call is deferred: the macro emits a
/// wrapper fn holding it plus an `inventory::submit!` -- see [`register`] and
/// `vm::api::MimasReg` for how wrappers are collected at install time.
#[proc_macro_attribute]
pub fn mimas(attr: TokenStream, item: TokenStream) -> TokenStream {
    match expand_mimas(attr, item) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

#[proc_macro_derive(MimasEnum, attributes(mimas_dim))]
pub fn derive_mimas_enum(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive::expand_derive(&input, true) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

#[proc_macro_derive(MimasStruct, attributes(mimas_dim))]
pub fn derive_mimas_struct(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive::expand_derive(&input, false) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand_mimas(attr: TokenStream, item: TokenStream) -> Result<TokenStream2, syn::Error> {
    // `#[mimas]` registers into the prelude; `#[mimas(foo::bar)]` names a module path
    let module: Option<String> = if attr.is_empty() {
        None
    } else {
        let path: syn::Path = syn::parse(attr)?;
        Some(
            path.segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::"),
        )
    };

    match syn::parse::<syn::Item>(item)? {
        syn::Item::Fn(mut function) => {
            let src = src_submission(&function.sig.ident, function.sig.ident.span());
            let effects = effects_submission(&function.sig.ident, &mut function.attrs)?;
            let (_, mutating) = convert::expand_conversion(&mut function)?;
            let meta = meta_submission(
                &function.sig.ident,
                &param_names(&function.sig),
                &collect_doc(&function.attrs),
            );
            if let Some(meta) = meta {
                function.block.stmts.insert(0, meta);
            }
            let mutates = mutates_submissions(&function.sig.ident, &mutating);
            let registration = register::fn_registration(&function.sig.ident, module.as_deref());
            Ok(quote!(#function #registration #src #mutates #effects))
        }
        // struct / enum: emit the same impls the derives would (so don't *also* `#[derive]`
        // them) plus the `add_adt` submission
        syn::Item::Struct(item) => {
            let di = DeriveInput::from(item);
            let impls = derive::expand_derive(&di, false)?;
            let registration = register::adt_registration(&di.ident, module.as_deref());
            Ok(quote!(#di #impls #registration))
        }
        syn::Item::Enum(item) => {
            let di = DeriveInput::from(item);
            let impls = derive::expand_derive(&di, true)?;
            let registration = register::adt_registration(&di.ident, module.as_deref());
            Ok(quote!(#di #impls #registration))
        }
        syn::Item::Const(item) => {
            let registration = register::const_registration(&item, module.as_deref())?;
            Ok(quote!(#item #registration))
        }
        syn::Item::Impl(block) => {
            if module.is_some() {
                return Err(syn::Error::new_spanned(
                    &block.self_ty,
                    "methods attach to their adt, not a module -- drop the path argument",
                ));
            }
            methods::expand_impl(block)
        }
        other => Err(syn::Error::new_spanned(
            other,
            "`#[mimas]` supports free fns, structs, enums, consts, and inherent impl blocks.",
        )),
    }
}

// proc_macro_crate v3 ignores [lib.name] in integration tests -- it returns the sanitized
// package name (`mimas_vm`) but the only importable name is the lib name (`vm`).
fn vm_path() -> TokenStream2 {
    if let Ok(found) = crate_name("mimas-vm") {
        let name = match found {
            FoundCrate::Itself => "vm",
            FoundCrate::Name(n) if n == "mimas_vm" => "vm",
            FoundCrate::Name(n) => {
                let ident = Ident::new(&n, proc_macro2::Span::call_site());
                return quote!(::#ident);
            }
        };
        let ident = Ident::new(name, proc_macro2::Span::call_site());
        return quote!(::#ident);
    }
    if let Ok(FoundCrate::Name(name)) = crate_name("mimas") {
        let ident = Ident::new(&name, proc_macro2::Span::call_site());
        return quote!(::#ident::vm);
    }
    quote!(::vm)
}

/// Collects an item's leading `///` doc-comment (each line stripped of the one leading space
/// rustdoc adds). `#[doc = "..."]` is how the compiler desugars `///`, so both spellings work.
fn collect_doc(attrs: &[syn::Attribute]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        let syn::Meta::NameValue(nv) = &attr.meta else {
            continue;
        };
        let syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(s),
            ..
        }) = &nv.value
        else {
            continue;
        };
        lines.push(
            s.value()
                .strip_prefix(' ')
                .map(str::to_string)
                .unwrap_or_else(|| s.value()),
        );
    }
    lines.join("\n")
}

/// Ships a fn's parameter names and doc-comment to install time keyed by the item's full Rust
/// path, so `vm::api::meta_for` can join them onto the registered `ApiFunction`/`ApiMethod`
/// (whose receiver/module are known only at the `api.add_*` call site, not here). Callers
/// insert the returned statement into the fn's body: `inventory::submit!` expands to an
/// unnamed `const`, which an `impl` block rejects but a fn body accepts.
fn meta_submission(fn_ident: &Ident, params: &[String], doc: &str) -> Option<syn::Stmt> {
    if doc.is_empty() && params.is_empty() {
        return None;
    }
    let vm = vm_path();
    let name = fn_ident.to_string();
    Some(parse_quote! {
        #vm::inventory::submit! {
            #vm::api::NativeMeta {
                path: ::std::concat!(::std::module_path!(), "::", #name),
                parameters: &[#(#params),*],
                doc: #doc,
            }
        }
    })
}

/// Collects the identifiers used in a function signature's parameters, skipping `ctx`.
fn param_names(sig: &syn::Signature) -> Vec<String> {
    sig.inputs
        .iter()
        .skip(1)
        .map(|arg| match arg {
            FnArg::Typed(t) => match &*t.pat {
                Pat::Ident(i) => i.ident.to_string(),
                _ => "_".to_string(),
            },
            FnArg::Receiver(_) => "self".to_string(),
        })
        .collect()
}

/// Ships the definition site's `file!()`/`line!()` to install time, keyed by `key_ident`'s full
/// Rust path exactly like [`doc_submission`]. `span` locates the recorded line -- a method
/// passes its own ident's span so the link lands on the method, not the generated shim (which
/// `key_ident` names, since install-time lookup happens by `type_name_of_val(&shim)`).
pub(crate) fn src_submission(key_ident: &Ident, span: proc_macro2::Span) -> TokenStream2 {
    let vm = vm_path();
    let name = key_ident.to_string();
    // `submit!` expands to an anonymous `const _` which impl bodies reject -- a named const
    // wrapper keeps it legal there (in-impl `#[native]` methods) without changing behavior.
    let holder = quote::format_ident!("__mimas_src_{name}");
    let file = quote::quote_spanned!(span => ::std::file!());
    let line = quote::quote_spanned!(span => ::std::line!());
    quote! {
        #[allow(non_upper_case_globals)]
        const #holder: () = {
            #vm::inventory::submit! {
                #vm::api::NativeSrc {
                    path: ::std::concat!(::std::module_path!(), "::", #name),
                    file: #file,
                    manifest: ::std::env!("CARGO_MANIFEST_DIR"),
                    line: #line,
                }
            }
        };
    }
}

/// Ships a fn's declared effect set to install time keyed by the item's full Rust path, for
/// `vm::api::effects_for` to join onto the registered `ApiFunction`/`ApiMethod` (see
/// `vm::api::NativeEffects`). The `#[effects(...)]` helper attribute is drained from `attrs`
/// so it never reaches rustc. Effect names mirror `shared::Fx`: `doc`, `net`, `rng`,
/// `yield`, `io`, `time`.
///
/// `#[effects]` with no list declares the pure set -- an explicit audit of "does nothing".
pub(crate) fn effects_submission(
    fn_ident: &Ident,
    attrs: &mut Vec<syn::Attribute>,
) -> Result<Option<TokenStream2>, syn::Error> {
    let Some(pos) = attrs.iter().position(|a| a.path().is_ident("effects")) else {
        return Ok(None);
    };
    let attr = attrs.remove(pos);
    let names = attr.parse_args_with(Punctuated::<Ident, Token![,]>::parse_terminated)?;
    let mut bits: u8 = 0;
    for name in names {
        bits |= match name.to_string().as_str() {
            "doc" => 1 << 0,
            "net" => 1 << 1,
            "rng" => 1 << 2,
            "yield" => 1 << 3,
            "io" => 1 << 4,
            "time" => 1 << 5,
            _ => {
                return Err(syn::Error::new_spanned(
                    name,
                    "unknown effect -- expected one of `doc`, `net`, `rng`, `yield`, `io`, `time`",
                ));
            }
        };
    }
    let vm = vm_path();
    let name = fn_ident.to_string();
    Ok(Some(quote! {
        #vm::inventory::submit! {
            #vm::api::NativeEffects {
                path: ::std::concat!(::std::module_path!(), "::", #name),
                effects: #bits,
            }
        }
    }))
}

/// Ships the indices of `&mut` params to install time keyed by the item's full Rust path, for
/// `vm::api::mutates_recv` to join onto the registered `ApiMethod` (see `vm::api::NativeMutates`).
pub(crate) fn mutates_submissions(fn_ident: &Ident, indices: &[usize]) -> TokenStream2 {
    let vm = vm_path();
    let name = fn_ident.to_string();
    let submissions = indices.iter().map(|index| {
        quote! {
            #vm::inventory::submit! {
                #vm::api::NativeMutates {
                    path: ::std::concat!(::std::module_path!(), "::", #name),
                    index: #index,
                }
            }
        }
    });
    quote!(#(#submissions)*)
}

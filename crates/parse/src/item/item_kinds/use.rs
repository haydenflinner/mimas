use itertools::Itertools;

use shared::Fx;

use crate::{
    expr::Ident,
    item::{IntoItem, ItemKind},
};

#[derive(Debug, PartialEq, Clone)]
pub enum Use {
    Singular(Vec<Ident>, Ident),
    Multi(Vec<Ident>, Vec<Ident>),
    All(Vec<Ident>),
    /// `use "page-name";` -- a request to the host, which provides that script's items before
    /// compiling (splicing its source in). The compiler itself has nothing to resolve.
    Host(HostUse),
}
impl From<Use> for ItemKind {
    fn from(us: Use) -> Self {
        Self::Use(us)
    }
}
impl IntoItem for Use {}

/// A `use "…"` host include: the quoted name — which may carry a
/// `@<64-hex>` pin or `@latest` follow tail, see [`use_target`] — plus
/// an optional `granting {…}` clause, the capability subset the page
/// passes down to the included code. Resolving the name and enforcing
/// the grant stay the host's business (`mimas.use.page`); this is what
/// the line *asks* for.
#[derive(Debug, PartialEq, Clone)]
pub struct HostUse {
    /// The quoted name, verbatim: `kit`, `kit@<64-hex>`, `kit@latest`.
    pub name: String,
    /// The `granting { net, rng }` clause: `Some` when written
    /// (`Some(vec![])` for `granting {}`, a purity assertion), `None`
    /// for a bare `use "…";`.
    pub grants: Option<Vec<Ident>>,
}

impl HostUse {
    /// What the name asks the host for (pin, follow or live page).
    pub fn target(&self) -> UseTarget<'_> {
        use_target(&self.name)
    }

    /// The clause as an effect set; `None` when none was written. The
    /// parser validates the names, so an unknown word contributes
    /// nothing here (it already reported at parse).
    // r[impl mimas.use.page.granting]
    pub fn grants_fx(&self) -> Option<Fx> {
        self.grants.as_ref().map(|grants| {
            grants
                .iter()
                .filter_map(|g| Fx::named(&g.lexeme))
                .fold(Fx::empty(), |acc, fx| acc | fx)
        })
    }
}

/// What a `use "…"` name asks the host for. The `@` tail grammar:
/// `name@<64-hex>` pins the include to the module blob stored under
/// that address, `name@latest` follows the author's head, and anything
/// else — a `@` that doesn't end in either, like `we@rd` — is ordinary
/// page-name text looked up live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseTarget<'a> {
    /// `use "kit"` — resolve the live page at every splice.
    Live(&'a str),
    /// `use "kit@latest"` — resolve like a live page, and keep
    /// following the author's head: publishes to `kit` propagate to
    /// this use without re-pinning.
    Follow(&'a str),
    /// `use "kit@<64-hex>"` — the module blob stored under `hash`.
    /// `page` is a human hint (where the pin was taken); resolution
    /// keys on `hash` alone.
    Pinned { page: &'a str, hash: &'a str },
}

/// Split a `use` name into its target. Only a trailing `@latest` or
/// `@<64-hex>` is a version tail — a `@` anywhere else in the name (or
/// a `@latest` with no name before it) is ordinary text. The grammar
/// `store::use_target`/`pins.ts` apply, extended with `@latest`.
// r[impl mimas.use.page.target]
pub fn use_target(name: &str) -> UseTarget<'_> {
    match name.rsplit_once('@') {
        Some((page, "latest")) if !page.is_empty() => UseTarget::Follow(page),
        Some((page, hash)) if is_pin_hash(hash) => UseTarget::Pinned { page, hash },
        _ => UseTarget::Live(name),
    }
}

/// A module-blob address: 64 lowercase hex digits, what `blake3::to_hex`
/// writes and `pins.ts` matches. Uppercase hex is *not* a pin —
/// `name@ABCD…` stays a live page name.
fn is_pin_hash(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[mutants::skip]
impl std::fmt::Display for Use {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Use::Singular(path, item) => {
                f.pad(&format!("use {}::{};", path.iter().join("::"), item))
            }
            Use::Multi(path, items) => f.pad(&format!(
                "use {}::{{ {} }};",
                path.iter().join("::"),
                items.iter().join(", ")
            )),
            Use::All(path) => f.pad(&format!("use {}::*;", path.iter().join("::"))),
            Use::Host(host) => {
                let grants = match &host.grants {
                    Some(grants) => format!(
                        " granting {{ {} }}",
                        grants.iter().map(|g| g.lexeme.as_str()).join(", ")
                    ),
                    None => String::new(),
                };
                f.pad(&format!("use {:?}{};", host.name, grants))
            }
        }
    }
}

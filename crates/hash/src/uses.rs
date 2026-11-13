//! The `use "…"` ledger: each include site's version mode (live,
//! `@latest` follow, `@<64-hex>` pin) and its `granting {…}` set, plus
//! the pure grant check a host's effect audit feeds.
//!
//! `use_sites` is the parser-backed record — string/comment impostors
//! can't appear in it, and a source that doesn't parse yields no sites
//! (the host's line scan is the lenient path for mid-edit text).
//! `check_grant` reuses the `Fx` inference the const-purity checker
//! already computes (`mimas.const.computed`): the named page's audited
//! set must fit the grant its use line declared.

use parse::{ItemKind, Parser, StmtKind, Use, lex::Lexer};
use shared::{Fx, Located};

pub use parse::{UseTarget, use_target};

/// The version mode a `use "…"` declares — the owned counterpart of
/// [`parse::UseTarget`], for records that outlive the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Version {
    /// No tail: the live page, resolved at every splice.
    Live,
    /// `…@latest`: follows the author's head; publishes propagate.
    Follow,
    /// `…@<64-hex>`: the module blob at this address.
    Pinned(String),
}

/// One `use "…"` site's declaration. `name` is the spelled text (tail
/// included); `page` the name minus the version tail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UseSite {
    pub name: String,
    pub page: String,
    pub version: Version,
    /// The `granting {…}` set; `None` when no clause was written — the
    /// include then runs under the page's ambient caps, unchecked.
    pub grants: Option<Fx>,
    /// Byte span of the whole `use` item.
    pub start: usize,
    pub end: usize,
}

/// A `use "…"` site and the page that declared it — `owner: None` for
/// the assembled page's own source, `Some(name)` for an include (the
/// name the use that pulled it spelled). The version/grant ledger of a
/// whole resolved closure — what a publish records and what a checker
/// reads downstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UseRecord {
    pub owner: Option<String>,
    pub site: UseSite,
}

/// Every top-level `use "…"` site in `source`, in source order.
// r[impl mimas.use.page.target]
pub fn use_sites(source: &str) -> Vec<UseSite> {
    let Ok(ast) = Parser::new_extending(Lexer::new(source, 0, "uses".into())).try_into_ast()
    else {
        return Vec::new();
    };
    ast.stmts()
        .iter()
        .filter_map(|stmt| {
            let StmtKind::Item(item) = stmt.kind() else {
                return None;
            };
            let ItemKind::Use(Use::Host(host)) = item.kind() else {
                return None;
            };
            let (page, version) = match host.target() {
                UseTarget::Live(name) => (name.to_string(), Version::Live),
                UseTarget::Follow(page) => (page.to_string(), Version::Follow),
                UseTarget::Pinned { page, hash } => {
                    (page.to_string(), Version::Pinned(hash.to_string()))
                }
            };
            let span = item.location().span;
            Some(UseSite {
                name: host.name.clone(),
                page,
                version,
                grants: host.grants_fx(),
                start: span.start,
                end: span.end,
            })
        })
        .collect()
}

/// One include edge whose audited effects escape the grant its use
/// site declared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantDenial {
    /// The spelled use name (`kit`, `kit@H`, `kit@latest`).
    pub name: String,
    /// The effect set the named page's own code was audited at.
    pub required: Fx,
    /// What the use site granted.
    pub granted: Fx,
}

impl GrantDenial {
    /// The ungranted footprint: `required - granted` — an `unaudited`
    /// requirement without an `unaudited` grant shows here too.
    pub fn missing(&self) -> Fx {
        self.required.difference(self.granted)
    }
}

impl std::fmt::Display for GrantDenial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "`{}` needs {} beyond the granted {}",
            self.name,
            self.missing(),
            self.granted
        )
    }
}

/// Check one include edge: the audited `required` effects of the page
/// a `use` pulls must fit the grant the site declared. A site with no
/// `granting` clause (`None`) has nothing to check against — the
/// include runs under the page's ambient caps, as it always has.
///
/// `mimas.use.page.grant-narrow`: grants narrow, never widen — the host
/// folds that half by unioning the site's own `grants` into `required`
/// before the call, so a clause asking for more than the includer's
/// authority trips the same denial a code-over-grant breach does. The
/// solver runs both halves directly on spliced source
/// (`grades.rs`), where the includer's authority is its `needs` or
/// ambient `Fx::ANY`.
// r[impl mimas.use.page.grant-check]
pub fn check_grant(name: &str, required: Fx, grants: Option<Fx>) -> Result<(), GrantDenial> {
    match grants {
        Some(granted) if !required.fits(granted) => Err(GrantDenial {
            name: name.to_string(),
            required,
            granted,
        }),
        _ => Ok(()),
    }
}

/// Every site's denial. `required` maps a spelled use name to the
/// audited `Fx` of the page it resolved to — the page mark's effects
/// (`audit_includes`/`IncludeAudit.effects` in the host). An
/// unresolved name skips its site: missing includes report through
/// `include.missing`, not the grant check.
pub fn check_grants(
    sites: &[UseSite],
    mut required: impl FnMut(&str) -> Option<Fx>,
) -> Vec<GrantDenial> {
    sites
        .iter()
        .filter_map(|site| {
            required(&site.name).and_then(|fx| check_grant(&site.name, fx, site.grants).err())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Globals;

    const H: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    // r[verify mimas.use.page.target]
    #[test]
    fn target_grammar() {
        assert_eq!(use_target("kit"), UseTarget::Live("kit"));
        assert_eq!(use_target("we@rd"), UseTarget::Live("we@rd"));
        assert_eq!(use_target("kit@latest"), UseTarget::Follow("kit"));
        assert_eq!(
            use_target(&format!("kit@{H}")),
            UseTarget::Pinned {
                page: "kit",
                hash: H
            }
        );
        // no name before `@`, non-exact words and uppercase hex all
        // stay ordinary page-name text
        assert_eq!(use_target("@latest"), UseTarget::Live("@latest"));
        assert_eq!(use_target("kit@LATest"), UseTarget::Live("kit@LATest"));
        assert_eq!(
            use_target(&format!("kit@{}", H.to_uppercase())),
            UseTarget::Live(&format!("kit@{}", H.to_uppercase()))
        );
    }

    // r[verify mimas.use.page.target]
    // r[verify mimas.use.page.granting]
    #[test]
    fn use_sites_ledger() {
        let src = "use \"kit\";\n\
                   use \"dep@latest\" granting { net, rng };\n\
                   use \"pin@0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\" granting {};\n\
                   let x = 1;";
        let sites = use_sites(src);
        assert_eq!(sites.len(), 3, "{sites:?}");
        assert_eq!(sites[0].page, "kit");
        assert_eq!(sites[0].version, Version::Live);
        assert_eq!(sites[0].grants, None);
        assert_eq!(sites[1].page, "dep");
        assert_eq!(sites[1].version, Version::Follow);
        assert_eq!(sites[1].grants, Some(Fx::NET | Fx::RNG));
        assert_eq!(sites[2].version, Version::Pinned(H.to_string()));
        assert_eq!(sites[2].grants, Some(Fx::empty()));
        // a source mid-edit that doesn't parse yields no sites
        assert!(use_sites("use \"kit\";\nfn broken(").is_empty());
    }

    // r[verify mimas.use.page.grant-check]
    #[test]
    fn grant_check_fits() {
        assert!(check_grant("kit", Fx::RNG, Some(Fx::RNG | Fx::NET)).is_ok());
        let denied = check_grant("kit", Fx::IO | Fx::RNG, Some(Fx::RNG)).unwrap_err();
        assert_eq!(denied.missing(), Fx::IO);
        // `unaudited` requirements need the flag in the grant — ANY
        // alone doesn't buy ungraded calls
        assert!(check_grant("kit", Fx::RNG | Fx::UNAUDITED, Some(Fx::ANY)).is_err());
        assert!(
            check_grant("kit", Fx::RNG | Fx::UNAUDITED, Some(Fx::ANY | Fx::UNAUDITED)).is_ok()
        );
        // no clause -> ambient, nothing to check
        assert!(check_grant("kit", Fx::ANY | Fx::UNAUDITED, None).is_ok());
        // check_grants maps spelled names to audited sets
        let sites = use_sites("use \"kit\" granting { rng };\nuse \"dep\";");
        let denials = check_grants(&sites, |name| match name {
            "kit" => Some(Fx::IO),
            "dep" => Some(Fx::ANY | Fx::UNAUDITED),
            _ => None,
        });
        assert_eq!(denials.len(), 1);
        assert_eq!(denials[0].name, "kit");
    }

    #[test]
    fn globals_records_the_closure_ledger() {
        let globals = Globals::for_page(
            "use \"kit\" granting { rng };\nfn f() { 1 }",
            |name| (name == "kit").then(|| "use \"deep\";\nfn g() { 2 }".to_string()),
        );
        let records = globals.use_records();
        assert_eq!(records.len(), 2, "{records:?}");
        assert_eq!(records[0].owner, None);
        assert_eq!(records[0].site.page, "kit");
        assert_eq!(records[0].site.grants, Some(Fx::RNG));
        assert_eq!(records[1].owner.as_deref(), Some("kit"));
        assert_eq!(records[1].site.page, "deep");
        assert_eq!(records[1].site.grants, None);
    }
}

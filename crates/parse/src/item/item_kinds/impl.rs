use itertools::Itertools;

use crate::{
    Ident,
    item::{IntoItem, Item, ItemKind},
};

/// Representation of an impl block in mimas.
#[derive(Debug, PartialEq, Clone)]
pub struct Impl {
    /// The type this impl block is for.
    pub target: Ident,
    /// The pact being implemented, if any.
    pub pact: Option<Ident>,
    /// A pact this impl forwards members from: `impl Drag for Pow via
    /// SlotDrag` covers each `Drag` member the block omits with `Pow`'s
    /// `SlotDrag` member of the same name (`mimas.impl.via`).
    pub via: Option<Ident>,
    /// The items this impl declares.
    pub items: Vec<Item>,
    /// `impl S<T: Bound>`: bounds on the type's own parameters, each with at
    /// least one pact (`mimas.generic.impl-bound`). Empty for a plain impl.
    pub bounds: Vec<crate::components::TypeParam>,
    /// `type Item = Card;`: the pact's associated types this impl fixes
    /// (`mimas.pact.assoc-type`).
    pub assoc: Vec<(Ident, crate::components::Annotation)>,
    /// `impl Iter for [T]` / `impl Iter for {K: V}`: the built-in shape the
    /// impl is for, with the parameters it names. The target is then the
    /// library's `array` or `dict` (`mimas.impl.builtin-shape`).
    pub shape: Option<(crate::components::Annotation, Vec<crate::components::TypeParam>)>,
}
impl Impl {
    pub(crate) fn new(
        target: Ident,
        pact: Option<Ident>,
        via: Option<Ident>,
        items: Vec<Item>,
    ) -> Self {
        Self {
            target,
            pact,
            via,
            items,
            bounds: vec![],
            assoc: vec![],
            shape: None,
        }
    }
}
impl From<Impl> for ItemKind {
    fn from(imp: Impl) -> Self {
        Self::Impl(imp)
    }
}
impl IntoItem for Impl {}

#[mutants::skip]
impl std::fmt::Display for Impl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let head = match &self.pact {
            Some(pact) => format!("impl {} for {}", pact, self.target),
            None => format!("impl {}", self.target),
        };
        let head = match &self.via {
            Some(via) => format!("{head} via {via}"),
            None => head,
        };
        if self.items.is_empty() {
            f.pad(&format!("{head} {{}}"))
        } else {
            f.pad(&format!("{head} {{ {} }}", self.items.iter().join(" ")))
        }
    }
}

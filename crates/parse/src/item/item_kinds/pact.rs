use crate::{
    Expr,
    components::{Annotation, Binding},
    expr::Ident,
    item::{IntoItem, ItemKind},
};
use itertools::Itertools;
use shared::{Located, Location};

/// Representation of a `pact` declaration in mimas.
#[derive(Debug, PartialEq, Clone)]
pub struct Pact {
    pub name: Ident,
    /// The pacts this one requires: `pact Drag: Slots` declares that every
    /// implementer of `Drag` also implements `Slots`, and that `Slots`'
    /// members are reachable through a `Drag` bound (`mimas.pact.requires`).
    pub requires: Vec<crate::components::Bound>,
    pub items: Vec<PactItem>,
    /// `pact Add<Rhs = Self>`: the pact's own type parameters, each with an
    /// optional default (`mimas.pact.params`). A bound or an impl names the
    /// arguments; a missing one takes its default.
    pub params: Vec<(Ident, Option<Annotation>)>,
    /// The pact's laws, its `check |x| …` lines (`mimas.pact.law`). Each was
    /// lowered to a top-level `fn` whose parameters are the pact's values;
    /// every impl runs it over its type's samples (`expand_laws`).
    pub laws: Vec<Law>,
}
impl Pact {
    pub(crate) fn new(
        name: Ident,
        requires: Vec<crate::components::Bound>,
        items: Vec<PactItem>,
    ) -> Self {
        Self {
            name,
            requires,
            items,
            params: vec![],
            laws: vec![],
        }
    }
}

/// One `check |x, y| …` line in a pact: the law as written, the parameter
/// names, and the synthesized function that evaluates it for given values.
#[derive(Debug, PartialEq, Clone)]
pub struct Law {
    pub text: String,
    /// The closure's body as written: each impl's law check runs a copy of
    /// it typed for that impl, so the law's calls resolve statically.
    pub body: String,
    pub params: Vec<String>,
    pub fn_name: String,
    pub location: Location,
}

impl From<Pact> for ItemKind {
    fn from(p: Pact) -> Self {
        Self::Pact(p)
    }
}
impl IntoItem for Pact {}

#[mutants::skip]
impl std::fmt::Display for Pact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let requires = if self.requires.is_empty() {
            String::new()
        } else {
            format!(": {}", self.requires.iter().join(" + "))
        };
        let params = if self.params.is_empty() {
            String::new()
        } else {
            let ps = self
                .params
                .iter()
                .map(|(n, d)| match d {
                    Some(d) => format!("{n} = {d}"),
                    None => n.to_string(),
                })
                .join(", ");
            format!("<{ps}>")
        };
        let name = format!("{}{params}", self.name);
        if self.items.is_empty() {
            f.pad(&format!("pact {name}{requires} {{}}"))
        } else {
            f.pad(&format!(
                "pact {name}{requires} {{ {} }}",
                self.items.iter().join(" ")
            ))
        }
    }
}

#[derive(Debug, Clone)]
pub enum PactItem {
    /// An associated type: `type Item;`. Each impl fixes it
    /// (`type Item = Card;`), and a member signature may name it
    /// (`mimas.pact.assoc-type`).
    Type {
        name: Ident,
        location: Location,
    },
    /// A constant signature: `const NAME: T;`. Pacts cannot declare default values for constants
    /// (an `impl` block must always supply one).
    Const {
        name: Ident,
        annotation: Annotation,
        location: Location,
    },
    /// A method signature: `fn name(params) -> ret;` or, with a default body, `fn name(params) ->
    /// ret { body }`. Parameters may not have default values in a pact signature.
    Fn {
        name: Ident,
        /// `fn map<U>(self, f: (Item) -> U) -> () -> U?`: the member's own
        /// type parameters, fresh at each call (`mimas.pact.member-generic`).
        generics: Vec<crate::components::TypeParam>,
        parameters: Vec<Binding>,
        return_type: Option<Annotation>,
        /// `Some(body)` if the pact provides a default implementation. Impls may omit this method
        /// when a default exists.
        default: Option<Expr>,
        location: Location,
    },
}

impl Located for PactItem {
    fn location(&self) -> Location {
        match self {
            PactItem::Const { location, .. }
            | PactItem::Fn { location, .. }
            | PactItem::Type { location, .. } => *location,
        }
    }
}

impl PartialEq for PactItem {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PactItem::Type { name, .. }, PactItem::Type { name: other, .. }) => name == other,
            (
                PactItem::Const {
                    name, annotation, ..
                },
                PactItem::Const {
                    name: other_name,
                    annotation: other_annotation,
                    ..
                },
            ) => (name, annotation) == (other_name, other_annotation),
            (
                PactItem::Fn {
                    name,
                    parameters,
                    return_type,
                    default,
                    ..
                },
                PactItem::Fn {
                    name: other_name,
                    parameters: other_parameters,
                    return_type: other_return_type,
                    default: other_default,
                    ..
                },
            ) => {
                (name, parameters, return_type, default)
                    == (
                        other_name,
                        other_parameters,
                        other_return_type,
                        other_default,
                    )
            }
            _ => false,
        }
    }
}

#[mutants::skip]
impl std::fmt::Display for PactItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PactItem::Type { name, .. } => f.pad(&format!("type {name};")),
            PactItem::Const {
                name, annotation, ..
            } => f.pad(&format!("const {}: {};", name, annotation)),
            PactItem::Fn {
                name,
                generics,
                parameters,
                return_type,
                default,
                ..
            } => {
                let param_str = parameters.iter().join(", ");
                let tps = if generics.is_empty() {
                    String::new()
                } else {
                    format!("<{}>", generics.iter().map(|g| g.name.lexeme.as_str()).join(", "))
                };
                let sig = match return_type {
                    Some(ret) => format!("fn {}{tps}({param_str}) -> {}", name, ret),
                    None => format!("fn {}{tps}({param_str})", name),
                };
                match default {
                    Some(body) => f.pad(&format!("{sig} {body}")),
                    None => f.pad(&format!("{sig};")),
                }
            }
        }
    }
}

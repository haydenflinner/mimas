use std::fmt::Display;

use indexmap::IndexMap;
use itertools::Itertools;
use shared::{Located, Location};

use crate::{
    Expr, Ident, Literal, NodeId,
    components::{POISON, Poison},
};

#[derive(Debug, Clone)]
pub struct Pat {
    id: NodeId,
    kind: PatKind,
    location: Location,
}

impl Pat {
    pub fn new(kind: PatKind, location: Location) -> Self {
        Self {
            id: NodeId::new(),
            kind,
            location,
        }
    }

    pub fn kind(&self) -> &PatKind {
        &self.kind
    }

    pub fn as_ident(&self) -> Option<&Ident> {
        let PatKind::Ident(ident) = self.kind() else {
            return None;
        };
        Some(ident)
    }

    /// The ident leaves this pattern binds, in traversal order: `(a, (b, c))` yields `a`, `b`,
    /// `c`. `_` counts as bound.
    pub fn bound_leaves(&self) -> Vec<&Pat> {
        let mut out = Vec::new();
        let mut stack = vec![self];
        while let Some(pat) = stack.pop() {
            match pat.kind() {
                PatKind::Ident(_) => out.push(pat),
                PatKind::Tuple(pats) | PatKind::TupleVariant(_, pats) | PatKind::Or(pats) => {
                    stack.extend(pats.iter().rev());
                }
                PatKind::Struct(_, fields) => stack.extend(fields.values()),
                PatKind::NullBind(inner) => stack.push(inner),
                PatKind::Variant(_) | PatKind::Literal(_) | PatKind::Poison(_) => {}
            }
        }
        out
    }

    pub fn id(&self) -> NodeId {
        self.id
    }
}

impl Located for Pat {
    fn location(&self) -> Location {
        self.location
    }
}

#[mutants::skip]
impl Display for Pat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&self.kind.to_string())
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum PatKind {
    Ident(Ident),
    Tuple(Vec<Pat>),
    Struct(Box<Expr>, IndexMap<String, Pat>),
    TupleVariant(Box<Expr>, Vec<Pat>),
    Variant(Box<Expr>),
    Or(Vec<Pat>),
    Literal(Literal),
    NullBind(Box<Pat>),
    Poison(Poison),
}

#[mutants::skip]
impl Display for PatKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatKind::Tuple(tup) => f.pad(&format!(
                "({})",
                tup.iter().map(ToString::to_string).join(", ")
            )),
            PatKind::Ident(ident) => f.pad(&ident.to_string()),
            PatKind::Struct(path, fields) => f.pad(&format!(
                "{path} {{ {} }}",
                fields.iter().map(|(k, v)| format!("{k}: {v}")).join(", ")
            )),
            PatKind::TupleVariant(path, p) => f.pad(&format!(
                "{path}({})",
                p.iter().map(ToString::to_string).join(", ")
            )),
            PatKind::Variant(path) => f.pad(&path.to_string()),
            PatKind::Or(pats) => f.pad(&pats.iter().map(ToString::to_string).join(" | ")),
            PatKind::Literal(lit) => f.pad(&lit.to_string()),
            PatKind::NullBind(ident) => f.pad(&format!("{ident}?")),
            PatKind::Poison(_) => f.pad(POISON),
        }
    }
}

impl From<Ident> for Pat {
    fn from(value: Ident) -> Self {
        let location = value.location();
        Pat {
            id: NodeId::new(),
            kind: PatKind::Ident(value),
            location,
        }
    }
}

impl PartialEq<Pat> for Pat {
    fn eq(&self, other: &Pat) -> bool {
        self.kind == other.kind
    }
}

impl From<Ident> for PatKind {
    fn from(value: Ident) -> Self {
        Self::Ident(value)
    }
}

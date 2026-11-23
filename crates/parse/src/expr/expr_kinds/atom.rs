use crate::{Expr, ExprKind, Ident, IntoExpr};
use itertools::Itertools;

/// `#EmptyList`: an atom, a name that needs no declaration
/// (r[impl mimas.atom.literal]). The checker types it as the built-in
/// `atom`, or as the variant of the same name where a declared enum is
/// expected (r[impl mimas.atom.enum]).
#[derive(Debug, PartialEq, Clone)]
pub struct Atom {
    /// The name without its `#`, located over the whole `#Name`.
    pub name: Ident,
    pub payload: AtomPayload,
}

/// What an atom carries. One form per atom: positional values or named
/// fields, never both (r[impl mimas.atom.payload]).
#[derive(Debug, PartialEq, Clone, Default)]
pub enum AtomPayload {
    /// `#EmptyList`
    #[default]
    None,
    /// `#EmptyList("the list was empty")`, `#Pair(1, 2)`
    Positional(Vec<Expr>),
    /// `#ParseError(line: 3)`
    Named(Vec<(Ident, Expr)>),
}

impl AtomPayload {
    /// The payload's values in written order.
    pub fn values(&self) -> Vec<&Expr> {
        match self {
            AtomPayload::None => vec![],
            AtomPayload::Positional(values) => values.iter().collect(),
            AtomPayload::Named(fields) => fields.iter().map(|(_, v)| v).collect(),
        }
    }
}

impl Atom {
    pub(crate) fn new(name: Ident, payload: AtomPayload) -> Self {
        Self { name, payload }
    }
}

impl From<Atom> for ExprKind {
    fn from(atom: Atom) -> Self {
        Self::Atom(atom)
    }
}
impl IntoExpr for Atom {}

#[mutants::skip]
impl std::fmt::Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = &self.name.lexeme;
        match &self.payload {
            AtomPayload::None => f.pad(&format!("#{name}")),
            AtomPayload::Positional(values) => f.pad(&format!(
                "#{name}({})",
                values.iter().map(ToString::to_string).join(", ")
            )),
            AtomPayload::Named(fields) => f.pad(&format!(
                "#{name}({})",
                fields
                    .iter()
                    .map(|(k, v)| format!("{}: {v}", k.lexeme))
                    .join(", ")
            )),
        }
    }
}

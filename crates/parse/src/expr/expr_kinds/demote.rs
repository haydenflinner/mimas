use crate::{Expr, ExprKind, IntoExpr};

/// A demote expression (`expr?`) in mimas — a `T!` becomes `T?`, a raised
/// error reads back as `null` instead of propagating.
#[derive(Debug, PartialEq, Clone)]
pub struct Demote {
    /// The expression being demoted.
    pub expr: Expr,
}

impl From<Demote> for ExprKind {
    fn from(demote: Demote) -> Self {
        Self::Demote(demote)
    }
}
impl IntoExpr for Demote {}

#[mutants::skip]
impl std::fmt::Display for Demote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&format!("{}?", self.expr))
    }
}

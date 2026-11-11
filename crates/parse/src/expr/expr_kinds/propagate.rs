use crate::{Expr, ExprKind, IntoExpr};

/// A propagate expression (`expr?`) in mimas: the value of a `T?` or a `T!`,
/// or an early exit from the enclosing function with its null or its raise.
#[derive(Debug, PartialEq, Clone)]
pub struct Propagate {
    /// The option or raising expression.
    pub expr: Expr,
}

impl From<Propagate> for ExprKind {
    fn from(propagate: Propagate) -> Self {
        Self::Propagate(propagate)
    }
}
impl IntoExpr for Propagate {}

#[mutants::skip]
impl std::fmt::Display for Propagate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&format!("{}?", self.expr))
    }
}

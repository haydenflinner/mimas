use crate::{Expr, ExprKind, IntoExpr, op};

/// Representation of an equality expression in mimas.
#[derive(Debug, PartialEq, Clone)]
pub struct Equality {
    /// The left hand side of the equality.
    pub left: Expr,
    /// The operator used in this equality.
    pub op: EqualityOp,
    /// The right hand side of the equality.
    pub right: Expr,
}
impl Equality {
    /// Creates a new equality.
    pub(crate) fn new(left: Expr, op: EqualityOp, right: Expr) -> Self {
        Self { left, op, right }
    }
}
impl From<Equality> for ExprKind {
    fn from(equality: Equality) -> Self {
        Self::Equality(equality)
    }
}
impl IntoExpr for Equality {}

op!(
    /// The various equality operations supported in mimas.
    EqualityOp {
        DoubleEqual => Equal,
        BangEqual => NotEqual,
        Greater => Greater,
        GreaterEqual => GreaterOrEqual,
        Less => Less,
        LessEqual => LessOrEqual,
    }
);

impl EqualityOp {
    /// The operator that makes the original comparison fail -- used by `assert!` rewriting.
    pub(crate) fn invert(self) -> Self {
        match self {
            Self::Equal => Self::NotEqual,
            Self::NotEqual => Self::Equal,
            Self::Greater => Self::LessOrEqual,
            Self::GreaterOrEqual => Self::Less,
            Self::Less => Self::GreaterOrEqual,
            Self::LessOrEqual => Self::Greater,
        }
    }
}

#[mutants::skip]
impl std::fmt::Display for Equality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&format!("{} {} {}", self.left, self.op, self.right))
    }
}

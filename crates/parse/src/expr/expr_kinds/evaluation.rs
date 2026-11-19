use crate::{Expr, ExprKind, IntoExpr, op};

/// A mathematical evaluation.
#[derive(Debug, PartialEq, Clone)]
pub struct Evaluation {
    /// The left hand side of the evaluation.
    pub left: Expr,
    /// The operator used in this evaluation.
    pub op: EvaluationOp,
    /// The right hand side of the evaluation.
    pub right: Expr,
}
impl Evaluation {
    /// Creates a new evaluation.
    pub(crate) fn new(left: Expr, op: EvaluationOp, right: Expr) -> Self {
        Self { left, op, right }
    }
}
impl From<Evaluation> for ExprKind {
    fn from(evaluation: Evaluation) -> Self {
        Self::Evaluation(evaluation)
    }
}
impl IntoExpr for Evaluation {}

op!(
    /// The various evaluation operations supported in mimas.
    EvaluationOp {
        Plus => Plus,
        Minus => Minus,
        Slash => Divide,
        Star => Multiply,
        TildeSlash => Div,
        // `÷` is the printed-friendly spelling of `~/` — `BinaryOp::of`
        // folds it to `Div`, so downstream only ever sees `Div`.
        Obelus => Obelus,
        Ident("mod") => Modulo,
        Ampersand => And,
        Pipe => Or,
        CircledPlus => Xor,
        Nand => Nand,
        Nor => Nor,
        DoubleLeftCaret => BitShiftLeft,
        DoubleRightCaret => BitShiftRight,
        // the named operators: no built-in meaning, a method each (r[impl mimas.op.glyphs])
        Wedge => Wedge,
        Vee => Vee,
        DotOp => Dot,
        Cross => Cross,
        Compose => Compose,
        Tensor => Tensor,
        // `⊡` is the boxed `⊙` — odot's infix spelling. `⊙` itself is
        // `outward`'s direction glyph: circles are directions, boxes are
        // operators. `a ⊙ b` applies `a` tagged outward, never hadamard.
        SqDot => Odot,
        SqPlus => Boxplus,
        SqMinus => Boxminus,
        Intersect => Intersect,
        Union => Union,
        Ominus => Ominus,
        Uplus => Uplus,
        // `≈`: binds like `==`, and only its method gives it meaning (r[impl mimas.op.approx])
        Approx => Approx,
    }
);

impl EvaluationOp {
    pub fn is_bitwise(&self) -> bool {
        self.is_binary() || self.is_bit_shift()
    }

    /// The bit/gate family: `&`, `|`, `⊕`, `⍲`, `⍱` — bitwise on ints,
    /// eager logic on bools (`nand`/`nor` spell the last two as words).
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            EvaluationOp::And
                | EvaluationOp::Or
                | EvaluationOp::Xor
                | EvaluationOp::Nand
                | EvaluationOp::Nor
        )
    }

    pub(crate) fn is_bit_shift(&self) -> bool {
        matches!(
            self,
            EvaluationOp::BitShiftLeft | EvaluationOp::BitShiftRight
        )
    }

    pub(crate) fn is_additive(&self) -> bool {
        matches!(self, EvaluationOp::Plus | EvaluationOp::Minus)
    }

    /// The named operators — `∧ ⋅ × ∘ ⊠ ⊡ ∩ ∨ ∪ ⊖ ⊎ ⊞ ⊟` — have no built-in
    /// meaning on any type: each is its method (`wedge`, `dot`, …) or an
    /// error. r[impl mimas.op.named]
    pub fn is_named(&self) -> bool {
        matches!(
            self,
            EvaluationOp::Wedge
                | EvaluationOp::Vee
                | EvaluationOp::Dot
                | EvaluationOp::Cross
                | EvaluationOp::Compose
                | EvaluationOp::Tensor
                | EvaluationOp::Odot
                | EvaluationOp::Boxplus
                | EvaluationOp::Boxminus
                | EvaluationOp::Intersect
                | EvaluationOp::Union
                | EvaluationOp::Ominus
                | EvaluationOp::Uplus
                | EvaluationOp::Approx
        )
    }

    /// The named operators that bind like `+`: `∨ ∪ ⊖ ⊎ ⊞ ⊟` (the rest
    /// bind like `*`).
    pub(crate) fn is_named_additive(&self) -> bool {
        matches!(
            self,
            EvaluationOp::Vee
                | EvaluationOp::Union
                | EvaluationOp::Ominus
                | EvaluationOp::Uplus
                | EvaluationOp::Boxplus
                | EvaluationOp::Boxminus
        )
    }
}

#[mutants::skip]
impl std::fmt::Display for Evaluation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&format!("{} {} {}", self.left, self.op, self.right))
    }
}

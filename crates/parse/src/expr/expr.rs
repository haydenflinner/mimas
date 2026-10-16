use crate::{IntoStmt, NodeId, StmtKind};

use super::*;
use shared::{Located, Location};

/// Declares the pieces of the `ExprKind` enum.
macro_rules! declare_expr_kinds {
    (#[doc = $doc:expr]$name:ident { $($tok:ident), * $(,)? }) => {
        #[derive(Debug, PartialEq, Clone)]
        pub enum $name {
            $(
                $tok($tok),
            )*
        }

        impl IntoExpr for ExprKind {}

        #[mutants::skip]
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                 match self {
                    $($name::$tok(v) => std::fmt::Display::fmt(v, f),)*
                }
            }
        }
    };
}

declare_expr_kinds!(
    /// todo
    ExprKind {
        Absolve,
        Access,
        Block,
        Break,
        Call,
        Closure,
        Collect,
        Continue,
        Equality,
        Evaluation,
        For,
        FString,
        Grouping,
        Ident,
        If,
        In,
        Literal,
        Logical,
        Loop,
        Match,
        Coalescence,
        Raise,
        Range,
        Return,
        Unary,
        Unwrap,
        While,
    }
);

/// A wrapper around an [ExprType], containing additional information discovered while parsing.
#[derive(Debug, Clone)]
pub struct Expr {
    kind: Box<ExprKind>,
    id: NodeId,
    location: Location,
}
impl Expr {
    /// Creates a new expr.
    pub fn new(kind: ExprKind, location: Location) -> Self {
        Self {
            kind: Box::new(kind),
            id: NodeId::new(),
            location,
        }
    }

    /// Get a reference to the inner ExprKind.
    pub fn kind(&self) -> &ExprKind {
        self.kind.as_ref()
    }

    /// Get the expr's id.
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn as_ident(&self) -> Option<&Ident> {
        if let ExprKind::Ident(ident) = self.kind() {
            Some(ident)
        } else {
            None
        }
    }

    /// True when this expr's access spine holds a `?` (an option access), so codegen has to
    /// null-check a following plain `[i]` / `.x` that rides the short-circuit. Purely syntactic.
    /// Used by codegen only; the solver decides what's *legal* to ride with its own (type-aware)
    /// rule, but every access it accepts as a ride sits under a `?` here, so this stays correct.
    pub fn taints_chain(&self) -> bool {
        match self.kind() {
            ExprKind::Access(
                Access::Square { left, kind, .. } | Access::Dot { left, kind, .. },
            ) => *kind == AccessKind::Option || left.taints_chain(),
            _ => false,
        }
    }
}

impl Located for Expr {
    fn location(&self) -> Location {
        self.location
    }
}

impl From<Expr> for StmtKind {
    fn from(expr: Expr) -> Self {
        Self::Expr(expr)
    }
}
impl IntoStmt for Expr {}

#[mutants::skip]
impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.kind(), f)
    }
}

// this used to only exist under #[cfg(test)], but now that all tests live at the base of the
// crate we can't gate this impl anymore. really shouldn't be used outside a test, so, todo
pub trait IntoExpr: Sized + Into<ExprKind> {
    fn into_expr(self) -> Expr
    where
        Self: Sized,
    {
        Expr::new(self.into(), Default::default())
    }
}

impl PartialEq<Expr> for Expr {
    fn eq(&self, other: &Expr) -> bool {
        self.kind == other.kind
    }
}

// #[cfg(test)]
// impl From<ExprKind> for Expr {
//     fn from(value: ExprKind) -> Self {
//         Expr {
//             kind: Box::new(value),
//             id: ExprId::default(),
//             location: Location::default(),
//             tag: None,
//         }
//     }
// }

impl Expr {
    /// Reassigns a fresh [`NodeId`] to this expr and every nested expr/pat/stmt.
    /// Needed when a parsed subtree is cloned into the AST more than once
    /// (e.g. desugared rule variants) -- the solver keys nodes by id.
    pub fn refresh_ids(&mut self) {
        self.id = NodeId::new();
        match &mut *self.kind {
            ExprKind::Absolve(Absolve { left, handler }) => {
                left.refresh_ids();
                handler.refresh_ids();
            }
            ExprKind::Access(a) => match a {
                Access::Dot { left, right, .. } => {
                    left.refresh_ids();
                    right.refresh_ids();
                }
                Access::Square { left, key, .. } => {
                    left.refresh_ids();
                    key.refresh_ids();
                }
                Access::DoubleColon { left, .. } => left.refresh_ids(),
                Access::Identity { .. } => {}
            },
            ExprKind::Block(Block {
                body, yielded_expr, ..
            }) => {
                for s in body {
                    s.refresh_ids();
                }
                if let Some(e) = yielded_expr {
                    e.refresh_ids();
                }
            }
            ExprKind::Break(Break { value }) | ExprKind::Return(Return { value }) => {
                if let Some(v) = value {
                    v.refresh_ids();
                }
            }
            ExprKind::Call(Call { left, arguments, .. }) => {
                left.refresh_ids();
                for a in arguments {
                    a.value.refresh_ids();
                }
            }
            ExprKind::Closure(Closure {
                parameters, body, ..
            }) => {
                for b in parameters {
                    b.left.refresh_ids();
                    if let Some(r) = &mut b.right {
                        r.refresh_ids();
                    }
                }
                body.refresh_ids();
            }
            ExprKind::Coalescence(Coalescence { left, right })
            | ExprKind::Equality(Equality { left, right, .. })
            | ExprKind::Evaluation(Evaluation { left, right, .. })
            | ExprKind::In(In { left, right, .. })
            | ExprKind::Logical(Logical { left, right, .. })
            | ExprKind::Range(Range {
                start: left,
                end: right,
                ..
            }) => {
                left.refresh_ids();
                right.refresh_ids();
            }
            ExprKind::Collect(Collect { value })
            | ExprKind::Grouping(Grouping { inner: value })
            | ExprKind::Raise(Raise { value })
            | ExprKind::Unary(Unary { right: value, .. })
            | ExprKind::Unwrap(Unwrap { expr: value }) => value.refresh_ids(),
            ExprKind::Continue(_) => {}
            ExprKind::For(For {
                binding,
                iterator,
                body,
            }) => {
                binding.refresh_ids();
                iterator.refresh_ids();
                body.refresh_ids();
            }
            ExprKind::FString(FString { parts }) => {
                for p in parts {
                    if let FStringPart::Expr(e) = p {
                        e.refresh_ids();
                    }
                }
            }
            ExprKind::Ident(_) => {}
            ExprKind::If(If {
                condition,
                main_body,
                else_expr,
                binding,
            }) => {
                condition.refresh_ids();
                main_body.refresh_ids();
                if let Some(e) = else_expr {
                    e.refresh_ids();
                }
                if let Some(p) = binding {
                    p.refresh_ids();
                }
            }
            ExprKind::Literal(lit) => refresh_literal_ids(lit),
            ExprKind::Loop(Loop { body }) => body.refresh_ids(),
            ExprKind::Match(Match {
                identity, cases, ..
            }) => {
                identity.refresh_ids();
                for c in cases {
                    c.refresh_ids();
                }
            }
            ExprKind::While(While {
                binding,
                header,
                body,
            }) => {
                if let Some(p) = binding {
                    p.refresh_ids();
                }
                header.refresh_ids();
                body.refresh_ids();
            }
            _ => {}
        }
    }
}

/// Recursively refreshes ids inside literal containers (arrays, tuples, dicts, structs).
pub(crate) fn refresh_literal_ids(lit: &mut Literal) {
    match lit {
        Literal::Array(items) | Literal::Tuple(items) => {
            for e in items {
                e.refresh_ids();
            }
        }
        Literal::Dictionary(fields) => {
            for (_, e) in fields {
                e.refresh_ids();
            }
        }
        Literal::Struct(StructLiteral { name, fields }) => {
            name.refresh_ids();
            for (_, e) in fields {
                e.refresh_ids();
            }
        }
        _ => {}
    }
}

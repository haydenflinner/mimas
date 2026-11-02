use super::*;
use crate::{NodeId, expr::*, item::Item};
use shared::{Located, Location};

/// Declares all the various needed pieces of the ExprKinds.
macro_rules! declare_stmt_kinds {
    (#[doc = $doc:expr]$name:ident { $($tok:ident), * $(,)? }) => {
        #[derive(Debug, PartialEq, Clone)]
        pub enum $name {
            $(
                $tok($tok),
            )*
        }

        impl IntoStmt for $name {}

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

declare_stmt_kinds!(
    /// if you delete this comment you get an error ;)
    StmtKind {
        Let,
        Module,
        Assignment,
        Expr,
        Item,
        Unit
    }
);

/// A wrapper around a Stmt, containing additional information discovered while parsing.
#[derive(Debug, Clone)]
pub struct Stmt {
    kind: Box<StmtKind>,
    #[allow(unused)]
    id: NodeId,
    location: Location,
}
impl Stmt {
    /// Creates a new stmt.
    pub(crate) fn new(kind: StmtKind, id: NodeId, location: Location) -> Self {
        Self {
            kind: Box::new(kind),
            id,
            location,
        }
    }

    /// Returns a reference to the inner StmtKind.
    pub fn kind(&self) -> &StmtKind {
        self.kind.as_ref()
    }

    /// A top-level expression statement `e` becomes `echo(e)`: the value
    /// of a bare expression reaches a host through the `Prints` fixture
    /// the way `print`'s text does — Jupyter's echo of a cell's last
    /// expression — and `echo` returns its argument, so a script's
    /// trailing expression is still its result. Loops, jumps and blocks
    /// are left alone: they yield nothing to show, and their control
    /// flow must not sit inside a call argument.
    /// r[impl mimas.echo]
    pub(crate) fn echo(self) -> Self {
        let Self { kind, id, location } = self;
        let kind = match *kind {
            StmtKind::Expr(e) if echoes(&e) => {
                let loc = e.location();
                let callee = Expr::new(ExprKind::Ident(Ident::synthetic("echo")), loc);
                let call = Call::new(callee, vec![Argument { name: None, value: e }]);
                StmtKind::Expr(Expr::new(ExprKind::Call(call), loc))
            }
            other => other,
        };
        Self { kind: Box::new(kind), id, location }
    }

    /// Get the stmt's id.
    #[allow(unused)]
    pub fn id(&self) -> NodeId {
        self.id
    }
}

impl Located for Stmt {
    fn location(&self) -> Location {
        self.location
    }
}

#[mutants::skip]
impl std::fmt::Display for Stmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.kind(), f)
    }
}

/// Derives two methods to convert the T into a [Stmt], supporting both a standard
/// `into_stmt` method, and a `into_stmt_lazy` for tests.
pub trait IntoStmt: Sized + Into<StmtKind> {
    /// Converts self into a statement box with a default span. Used in tests, where all spans are
    /// expected to be 0, 0.
    fn into_stmt(self) -> Stmt
    where
        Self: Sized,
    {
        Stmt {
            id: NodeId::default(),
            kind: Box::new(self.into()),
            location: Location::default(),
        }
    }
}

impl PartialEq<Stmt> for Stmt {
    fn eq(&self, other: &Stmt) -> bool {
        self.kind == other.kind
    }
}

/// Which expression statements are worth echoing (see [`Stmt::echo`]).
fn echoes(e: &Expr) -> bool {
    !matches!(
        e.kind(),
        ExprKind::For(_)
            | ExprKind::While(_)
            | ExprKind::Loop(_)
            | ExprKind::Return(_)
            | ExprKind::Break(_)
            | ExprKind::Continue(_)
            | ExprKind::Raise(_)
            | ExprKind::Block(_)
            | ExprKind::Poison(_)
    )
}

use itertools::Itertools;
use shared::Fx;

use crate::{Ident, IntoStmt, StmtKind};

/// A `needs { doc, net, … };` declaration: the page's own statement of
/// the effect authority its code requires. Top-level only. The same
/// vocabulary `granting {…}` writes — `Fx::named` validates at parse —
/// and the same set algebra: the checker refuses a page whose inferred
/// `Fx` escapes what its `needs` declared, and a `granting` clause can
/// never hand an include more than the page itself holds. Absent
/// `needs` there is no check — the page runs under ambient authority.
#[derive(Debug, PartialEq, Clone)]
pub struct Needs {
    /// The effect names as written (`net`, `time`), in source order.
    pub names: Vec<Ident>,
}

impl Needs {
    pub(crate) fn new(names: Vec<Ident>) -> Self {
        Self { names }
    }

    /// The declaration as an effect set. The parser validates the
    /// names, so an unknown word contributes nothing here (it already
    /// reported at parse).
    // r[impl mimas.fx.needs]
    pub fn needs_fx(&self) -> Fx {
        self.names
            .iter()
            .filter_map(|n| Fx::named(&n.lexeme))
            .fold(Fx::empty(), |acc, fx| acc | fx)
    }
}

impl From<Needs> for StmtKind {
    fn from(stmt: Needs) -> Self {
        Self::Needs(stmt)
    }
}

impl IntoStmt for Needs {}

#[mutants::skip]
impl std::fmt::Display for Needs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&format!(
            "needs {{ {} }};",
            self.names.iter().map(|n| n.lexeme.as_str()).join(", ")
        ))
    }
}

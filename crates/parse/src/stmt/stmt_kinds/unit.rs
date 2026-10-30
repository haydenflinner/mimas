use crate::{Ident, IntoStmt, StmtKind};

/// A `unit` declaration: `unit tile;` claims a fresh base dimension, and
/// `unit league = 3mi;` names a scale of known units. The table itself
/// (shared::units) is what the lexer and dims checker consult — the AST
/// keeps the decl so hashing and pretty printing see it.
#[derive(Debug, PartialEq, Clone)]
pub struct Unit {
    pub name: Ident,
    /// The `= spec` text as written (`3mi`), `None` for a bare decl.
    pub spec: Option<String>,
}

impl Unit {
    pub(crate) fn new(name: Ident, spec: Option<String>) -> Self {
        Self { name, spec }
    }
}
impl From<Unit> for StmtKind {
    fn from(stmt: Unit) -> Self {
        Self::Unit(stmt)
    }
}

impl IntoStmt for Unit {}

#[mutants::skip]
impl std::fmt::Display for Unit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&format!(
            "unit {}{};",
            self.name,
            self.spec
                .as_ref()
                .map(|s| format!(" = {s}"))
                .unwrap_or_default()
        ))
    }
}

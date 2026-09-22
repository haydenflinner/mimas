use itertools::Itertools;

use crate::{
    Expr, Ident, NodeId,
    item::{IntoItem, ItemKind},
};

/// One expression in a `#[tests]` list. Compiled as a 0-arg body the inspector runs; the name
/// is the source snippet so failures read as the check that failed.
#[derive(Debug, Clone)]
pub struct TestCase {
    pub id: NodeId,
    pub name: Ident,
    pub expr: Expr,
}

impl PartialEq for TestCase {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.expr == other.expr
    }
}

/// A `#[tests] [ expr, expr, ... ]` item. The inspector evaluates each expression; `false` or a
/// panic is a failure.
#[derive(Debug, PartialEq, Clone)]
pub struct Tests {
    pub cases: Vec<TestCase>,
}

impl Tests {
    pub(crate) fn new(cases: Vec<TestCase>) -> Self {
        Self { cases }
    }
}

impl From<Tests> for ItemKind {
    fn from(tests: Tests) -> Self {
        Self::Tests(tests)
    }
}

impl IntoItem for Tests {}

#[mutants::skip]
impl std::fmt::Display for Tests {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&format!(
            "[{}]",
            self.cases
                .iter()
                .map(|case| case.expr.to_string())
                .join(", ")
        ))
    }
}

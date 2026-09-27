use crate::{
    Expr,
    components::{Annotation, Pat},
    item::{IntoItem, ItemKind},
};

/// `const` declaration -- a name bound to a compile-time-foldable expression. Visibility lives
/// on the wrapping [Item]. The left side takes the same irrefutable patterns as `let` (idents and
/// tuples thereof), so `const (A, B) = (1, 2);` declares two constants.
#[derive(Debug, PartialEq, Clone)]
pub struct Const {
    pub left: Pat,
    pub annotation: Option<Annotation>,
    pub right: Expr,
}

impl Const {
    /// Creates a new const declaration.
    #[cfg(test)]
    pub(crate) fn new(left: impl Into<Pat>, right: Expr, annotation: Option<Annotation>) -> Self {
        Self {
            left: left.into(),
            right,
            annotation,
        }
    }
}

impl From<Const> for ItemKind {
    fn from(con: Const) -> Self {
        Self::Const(con)
    }
}
impl IntoItem for Const {}

#[mutants::skip]
impl std::fmt::Display for Const {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(&format!(
            "const {}{} = {};",
            self.left,
            self.annotation
                .as_ref()
                .map(|v| format!(": {v}"))
                .unwrap_or_default(),
            self.right,
        ))
    }
}

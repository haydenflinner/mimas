use std::{fmt::Display, ops::Deref};

use itertools::Itertools;

use crate::Ident;

/// One declared type parameter: `T`, or `T: Draw + Vary` -- the pacts any type
/// filling the slot must implement. Inside the item a bound is what lets
/// `x.draw()` resolve on an `x: T`: the call dispatches through the pact at
/// runtime, and every instantiation of `T` is checked against it.
///
/// Derefs to its name, so the many sites that only want the ident
/// (`param.lexeme`, `param.id`, `param.location`) read unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeParam {
    pub name: Ident,
    pub bounds: Vec<Ident>,
}

impl TypeParam {
    pub fn new(name: Ident, bounds: Vec<Ident>) -> Self {
        Self { name, bounds }
    }
}

impl Deref for TypeParam {
    type Target = Ident;
    fn deref(&self) -> &Ident {
        &self.name
    }
}

impl From<Ident> for TypeParam {
    fn from(name: Ident) -> Self {
        Self::new(name, vec![])
    }
}

#[mutants::skip]
impl Display for TypeParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.bounds.is_empty() {
            f.pad(&self.name.to_string())
        } else {
            f.pad(&format!(
                "{}: {}",
                self.name,
                self.bounds.iter().join(" + ")
            ))
        }
    }
}

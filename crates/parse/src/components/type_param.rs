use std::{fmt::Display, ops::Deref};

use itertools::Itertools;

use crate::{Ident, components::Annotation};

/// One pact a type parameter is held to: `Draw`, `Mul<float>`, or
/// `Add<Out = T>`. `args` fill the pact's own type parameters in order (a
/// missing one takes the pact's default), and `assoc` pins associated
/// types (`mimas.pact.params`, `mimas.pact.assoc-bound`).
///
/// Derefs to the pact's name, so a site that only wants the ident reads
/// unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct Bound {
    pub name: Ident,
    pub args: Vec<Annotation>,
    pub assoc: Vec<(Ident, Annotation)>,
}

impl Bound {
    pub fn is_plain(&self) -> bool {
        self.args.is_empty() && self.assoc.is_empty()
    }
}

impl From<Ident> for Bound {
    fn from(name: Ident) -> Self {
        Self {
            name,
            args: vec![],
            assoc: vec![],
        }
    }
}

impl Deref for Bound {
    type Target = Ident;
    fn deref(&self) -> &Ident {
        &self.name
    }
}

#[mutants::skip]
impl Display for Bound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_plain() {
            return f.pad(&self.name.to_string());
        }
        let parts = self
            .args
            .iter()
            .map(|a| a.to_string())
            .chain(self.assoc.iter().map(|(n, a)| format!("{n} = {a}")))
            .join(", ");
        f.pad(&format!("{}<{parts}>", self.name))
    }
}

/// One declared type parameter: `T`, or `T: Draw + Vary` -- the pacts any type
/// filling the slot must implement. Inside the item a bound is what lets
/// `x.draw()` resolve on an `x: T`: the call dispatches through the pact at
/// runtime, and every instantiation of `T` is checked against it.
///
/// Derefs to its name, so the many sites that only want the ident
/// (`param.lexeme`, `param.id`, `param.location`) read unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeParam {
    pub name: Ident,
    pub bounds: Vec<Bound>,
}

impl TypeParam {
    pub fn new(name: Ident, bounds: Vec<Bound>) -> Self {
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

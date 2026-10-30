use itertools::Itertools;

use crate::{
    Expr,
    components::{Annotation, Binding},
    expr::Ident,
    item::{IntoItem, ItemKind},
};

/// Representation of function declaration in mimas. Visibility lives on the wrapping [Item].
#[derive(Debug, PartialEq, Clone)]
pub struct Function {
    /// The name, if any, of this function. Anonymous functions do not have names.
    pub name: Ident,
    /// The declared type parameters, `<T, U>` after the name. Empty for non-generic fns.
    pub type_params: Vec<Ident>,
    /// The parameters of this function.
    pub parameters: Vec<Binding>,
    /// `where` predicates between the signature and the body, each checked against the
    /// bound arguments at call sites when provable and again when the call runs.
    /// `fn f(xs: [int]) where xs.len() > 2 { ... }` holds two here for `a, b` written
    /// comma-separated.
    pub wheres: Vec<Expr>,
    /// The body of the function declaration.
    pub body: Expr,
    /// The type bound for the return.
    pub return_type: Option<Annotation>,
}
impl Function {
    /// Creates a new function declaration.
    #[cfg(test)]
    pub(crate) fn new(
        name: Ident,
        // todo, dont think this should be binding, or at least, this shouldnt contain a pat
        parameters: Vec<Binding>,
        return_type: Option<Annotation>,
        body: Expr,
    ) -> Self {
        Self {
            name,
            type_params: vec![],
            parameters,
            wheres: vec![],
            body,
            return_type,
        }
    }

    pub fn is_method(&self) -> bool {
        self.parameters
            .first()
            .and_then(|b| b.left.as_ident())
            .is_some_and(|i| i.lexeme == "self")
    }
}
impl From<Function> for ItemKind {
    fn from(function: Function) -> Self {
        Self::Function(function)
    }
}
impl IntoItem for Function {}

#[mutants::skip]
impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let generics = if self.type_params.is_empty() {
            String::new()
        } else {
            format!("<{}>", self.type_params.iter().join(", "))
        };
        let param_str = self.parameters.iter().join(", ");
        let ret_str = self
            .return_type
            .as_ref()
            .map_or(String::new(), |ret| format!(" -> {ret}"));
        let where_str = if self.wheres.is_empty() {
            String::new()
        } else {
            format!(" where {}", self.wheres.iter().join(", "))
        };
        f.pad(&format!(
            "fn {}{generics}({param_str}){ret_str}{where_str} {}",
            self.name, self.body
        ))
    }
}

/// One of the inbuilt functions mimas's runtime supports.
#[derive(Debug, PartialEq, Clone)]
pub enum FfiFn {
    Print,
    Stringify,
}

impl TryFrom<String> for FfiFn {
    type Error = ();

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "print" => Ok(FfiFn::Print),
            "stringify" => Ok(FfiFn::Stringify),
            _ => Err(()),
        }
    }
}

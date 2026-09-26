use crate::{
    StructField,
    components::Annotation,
    expr::Ident,
    item::{IntoItem, ItemKind},
};
use itertools::Itertools;

/// Representation of an enum declaration in mimas. Visibility lives on the wrapping [Item].
#[derive(Debug, PartialEq, Clone)]
pub struct Enum {
    pub head: Ident,
    /// The declared type parameters, `<T>` after the name. Empty for non-generic enums.
    pub type_params: Vec<Ident>,
    /// The members of this enum.
    pub members: Vec<(Ident, Member)>,
}
impl Enum {
    #[cfg(test)]
    pub(crate) fn new(name: Ident, members: Vec<(Ident, Member)>) -> Self {
        Self {
            head: name,
            type_params: vec![],
            members,
        }
    }
}

impl From<Enum> for ItemKind {
    fn from(enu: Enum) -> Self {
        Self::Enum(enu)
    }
}
impl IntoItem for Enum {}

#[mutants::skip]
impl std::fmt::Display for Enum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let generics = if self.type_params.is_empty() {
            String::new()
        } else {
            format!("<{}>", self.type_params.iter().join(", "))
        };
        f.pad(&format!(
            "enum {}{generics} {{ {} }}",
            self.head,
            self.members
                .iter()
                .map(|(ident, member)| match member {
                    Member::Tuple(members) => format!(
                        "{ident}({})",
                        members.iter().map(|v| v.to_string()).join(", ")
                    ),
                    Member::Struct(fields) => {
                        format!(
                            "{ident} {{ {} }}",
                            fields
                                .iter()
                                .map(
                                    |StructField {
                                         name, annotation, ..
                                     }| {
                                        format!("{ident} {name}: {annotation}")
                                    },
                                )
                                .join(", ")
                        )
                    }
                })
                .join(", ")
        ))
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Member {
    Tuple(Vec<Annotation>),
    Struct(Vec<StructField>),
}

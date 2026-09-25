use crate::lex::TyKw;
use itertools::Itertools;

use crate::{
    components::{POISON, Poison},
    expr::Ident,
};

#[derive(Debug, PartialEq, Clone)]
pub enum Annotation {
    Unit,
    Kw(TyKw),
    Option(Box<Annotation>),
    Result(Box<Annotation>),
    Tuple(Vec<Annotation>),
    Array(Box<Annotation>),
    Dictionary(Box<Annotation>),
    Function(Vec<Annotation>, Box<Annotation>),
    Ty(Ident),
    Path(Vec<Ident>),
    Bounds(Vec<Ident>),
    /// A compound unit, `usd/kWh` or `m/s^2`: each unit with its signed exponent. Whether the
    /// names are units is for the checker to say. (A lone unit name is a plain `Ty`.)
    Quantity(Vec<(Ident, i8)>),
    Poison(Poison),
}

#[mutants::skip]
impl std::fmt::Display for Annotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Annotation::Unit => f.pad("()"),
            Annotation::Kw(tykw) => f.pad(&tykw.to_string()),
            Annotation::Option(t) => f.pad(&format!("{t}?")),
            Annotation::Result(t) => f.pad(&format!("{t}!")),
            Annotation::Tuple(members) => f.pad(&format!(
                "({})",
                members.iter().map(|p| p.to_string()).join(", "),
            )),
            Annotation::Array(e) => f.pad(&format!("[{e}]")),
            Annotation::Dictionary(e) => f.pad(&format!("~{{{e}}}")),
            Annotation::Function(params, ret) => f.pad(&format!(
                "({}) -> {ret}",
                params.iter().map(|p| p.to_string()).join(", "),
            )),
            Annotation::Ty(ident) => f.pad(&format!("{ident}")),
            Annotation::Path(idents) => f.pad(&idents.iter().map(|i| i.to_string()).join("::")),
            Annotation::Bounds(idents) => f.pad(&idents.iter().map(|i| i.to_string()).join(" + ")),
            Annotation::Quantity(parts) => f.pad(
                &parts
                    .iter()
                    .enumerate()
                    .map(|(i, (unit, exp))| match (i, exp) {
                        (0, 1) => unit.to_string(),
                        (0, e) => format!("{unit}^{e}"),
                        (_, 1) => format!("*{unit}"),
                        (_, -1) => format!("/{unit}"),
                        (_, e) if *e > 0 => format!("*{unit}^{e}"),
                        (_, e) => format!("/{unit}^{}", -e),
                    })
                    .join(""),
            ),
            Annotation::Poison(_) => f.pad(POISON),
        }
    }
}

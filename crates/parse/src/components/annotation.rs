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
    /// `[T]`, or a fixed-size `[T; n]` / `[T; m, n]`. The dims list is empty for a
    /// growable array; each `n` pins one level's length, outermost first -- so
    /// `[f32; 4, 8]` reads as "4 rows of `[f32; 8]`".
    Array(Box<Annotation>, Vec<usize>),
    Dictionary(Box<Annotation>),
    /// `{K: V}`: a dict keyed by `K` values instead of strings. The location is the key
    /// type's, for the checker's "cannot be a key" error.
    Map(Box<Annotation>, Box<Annotation>, shared::Location),
    Function(Vec<Annotation>, Box<Annotation>),
    Ty(Ident),
    Path(Vec<Ident>),
    Bounds(Vec<Ident>),
    /// A compound unit, `usd/kWh` or `m/s^2`: each unit with its signed exponent. Whether the
    /// names are units is for the checker to say. (A lone unit name is a plain `Ty`.)
    Quantity(Vec<(Ident, i8)>),
    /// A type applied to arguments in `<>`, `Pair<A, B>` or `mod::Pair<A, B>`: either real
    /// type arguments for a generic type, or a single unit (`Interval<kW>`) -- the checker
    /// knows which the head is.
    Applied(Vec<Ident>, Vec<Annotation>),
    /// `x: inout T` -- the `inout` call convention before a parameter's type: the callee
    /// binds the place the caller lent with `&`. The keyword keeps its ident for errors.
    Inout(Ident, Box<Annotation>),
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
            Annotation::Array(e, dims) if dims.is_empty() => f.pad(&format!("[{e}]")),
            Annotation::Array(e, dims) => f.pad(&format!("[{e}; {}]", dims.iter().join(", "))),
            Annotation::Dictionary(e) => f.pad(&format!("{{{e}}}")),
            Annotation::Map(k, v, _) => f.pad(&format!("{{{k}: {v}}}")),
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
                        // an axis part prints as its suffix: `px↕`, `usd@flow`
                        (_, 1) if unit.lexeme.starts_with('@') || unit.lexeme == "↔" || unit.lexeme == "↕" => {
                            unit.to_string()
                        }
                        (0, 1) => unit.to_string(),
                        (0, e) => format!("{unit}^{e}"),
                        (_, 1) => format!("*{unit}"),
                        (_, -1) => format!("/{unit}"),
                        (_, e) if *e > 0 => format!("*{unit}^{e}"),
                        (_, e) => format!("/{unit}^{}", -e),
                    })
                    .join(""),
            ),
            Annotation::Applied(ty, args) => f.pad(&format!(
                "{}<{}>",
                ty.iter().map(|i| i.to_string()).join("::"),
                args.iter().join(", ")
            )),
            Annotation::Inout(_, inner) => f.pad(&format!("inout {inner}")),
            Annotation::Poison(_) => f.pad(POISON),
        }
    }
}

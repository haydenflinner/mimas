//! The `name:type` spec strings that `df.schema("fare:float kwh:kWh")` accepts.
//!
//! One parser serves both consumers: the runtime validates columns against it (and
//! casts where it can), and the solver reads the same spec to type `pull` calls.

use crate::units::{self, Dim};

/// What a schema spec entry asserts about its column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SchemaTy {
    Int,
    Float,
    Str,
    Bool,
    /// A numeric column read as quantities in this dimension (`kwh:kWh`).
    Unit(Dim),
}

impl SchemaTy {
    /// The human-readable spelling for error messages (`kWh` prints as its base dims).
    pub fn describe(&self) -> String {
        match self {
            SchemaTy::Int => "int".into(),
            SchemaTy::Float => "float".into(),
            SchemaTy::Str => "str".into(),
            SchemaTy::Bool => "bool".into(),
            SchemaTy::Unit(_) => "a unit".into(),
        }
    }
}

/// Parse `zone:int, fare:float kwh:kWh` into `(name, ty)` pairs, in spec order.
/// Separators are commas and whitespace; each token is `name:type` where `type` is
/// `int`, `float`, `str`, `bool`, or a unit like `kWh` (which means floats in that
/// dimension).
pub fn parse_schema(spec: &str) -> Result<Vec<(String, SchemaTy)>, String> {
    let mut out = Vec::new();
    for token in spec.split(|c: char| c == ',' || c.is_whitespace()) {
        if token.is_empty() {
            continue;
        }
        let Some((name, ty)) = token.split_once(':') else {
            return Err(format!("schema entry `{token}` needs a `name:type` pair"));
        };
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!("`{name}` isn't a column name"));
        }
        let ty = match ty {
            "int" => SchemaTy::Int,
            "float" => SchemaTy::Float,
            "str" | "string" => SchemaTy::Str,
            "bool" => SchemaTy::Bool,
            other => match units::parse(other) {
                Some((dim, _)) => SchemaTy::Unit(dim),
                None => {
                    return Err(format!(
                        "`{other}` isn't a column type -- expected `int`, `float`, `str`, `bool`, or a unit like `kWh`"
                    ));
                }
            },
        };
        if out.iter().any(|(n, _): &(String, SchemaTy)| n == name) {
            return Err(format!("`{name}` appears twice in the schema"));
        }
        out.push((name.to_string(), ty));
    }
    if out.is_empty() {
        return Err("empty schema spec".into());
    }
    Ok(out)
}

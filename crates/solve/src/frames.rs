//! Dataframe schemas: `table { … }` literals, `df.schema("…")` assertions and `query { … }`
//! pipelines give the solver a column→type map, so `df.pull("fare")` returns a concrete
//! `[float]` instead of the unresolved generic the `pull` signature declares, and so a
//! misspelled column is a check-time error instead of a runtime raise.
//!
//! Schemas are tracked beside the type system, not in it: a frame's `Ty` stays the plain
//! nominal `DataFrame`, while [`Solver::frame_schemas`] / [`Solver::frame_decs`] carry the
//! column map for producing exprs and `let` bindings. `closed = false` marks frames that
//! may hold columns the checker can't see (a `$expr` splice among `derive` items, a join
//! with an opaque partner) -- `pull` misses there stay generic instead of erroring.

use std::rc::Rc;

use indexmap::IndexMap;
use parse::{
    Access, Call, Expr, ExprKind, Literal, NodeId,
    components::{Pat, PatKind},
    expr::{EqualityOp, EvaluationOp, UnaryOp},
};
use shared::{Located, Ty, units::Dim};

use crate::{
    Error, Result, Solver,
    components::TyExt,
    errors::{BadSchemaSpec, DimensionMismatch, NoSuchColumn},
    traits::Query,
};

/// What the checker knows about one column of a frame.
#[derive(Debug, Clone)]
pub(crate) struct FrameCol {
    /// The element type `pull("name")` returns. `None` for columns whose type isn't
    /// tracked (a `$expr` splice, an op the walker doesn't know) -- those keep the
    /// generic pull, exactly like an unschema'd frame.
    pub ty: Option<Ty>,
    /// The column's unit dimension, when it's a quantity column (`kwh:kWh`, or a
    /// `table {}` literal whose cells are unit literals). The dims pass reads this.
    pub dim: Option<Dim>,
}

impl FrameCol {
    fn known(ty: Ty) -> Self {
        Self {
            ty: Some(ty),
            dim: None,
        }
    }

    fn unknown() -> Self {
        Self {
            ty: None,
            dim: None,
        }
    }
}

/// A dataframe's columns as the solver sees them.
#[derive(Debug, Clone)]
pub(crate) struct FrameSchema {
    pub cols: IndexMap<String, FrameCol>,
    /// `false` when unseen columns may also be present (splices, opaque mixed sources).
    /// `pull` on an unlisted name only errors when closed.
    pub closed: bool,
}

impl FrameSchema {
    fn empty() -> Self {
        Self {
            cols: IndexMap::new(),
            closed: true,
        }
    }

    /// The `name` column, or a `NoSuchColumn` error when the schema is closed and lacks it.
    fn col(&self, solver: &Solver, at: &Expr, name: &str) -> Result<FrameCol> {
        match self.cols.get(name) {
            Some(col) => Ok(col.clone()),
            None if !self.closed => Ok(FrameCol::unknown()),
            None => Err(solver.no_such_column(at, name, self)),
        }
    }
}

fn numeric(ty: &Ty) -> bool {
    matches!(ty, Ty::Int | Ty::Float)
}

/// `x <op> y` column arithmetic promotes like scalar mimas: `int/int` is still float
/// division (`/`), `~/` keeps ints, floats absorb ints, `str + _` concatenates.
fn numeric_join(op: EvaluationOp, a: &Option<Ty>, b: &Option<Ty>) -> Option<Ty> {
    let (Some(a), Some(b)) = (a, b) else {
        return None;
    };
    match op {
        EvaluationOp::Divide => (numeric(a) || numeric(b)).then_some(Ty::Float),
        EvaluationOp::Div | EvaluationOp::Modulo => match (a, b) {
            (Ty::Int, Ty::Int) => Some(Ty::Int),
            _ if numeric(a) && numeric(b) => Some(Ty::Float),
            _ => None,
        },
        EvaluationOp::Plus => match (a, b) {
            (Ty::Str, _) | (_, Ty::Str) => Some(Ty::Str),
            (Ty::Int, Ty::Int) => Some(Ty::Int),
            _ if numeric(a) && numeric(b) => Some(Ty::Float),
            _ => None,
        },
        EvaluationOp::Minus | EvaluationOp::Multiply => match (a, b) {
            (Ty::Int, Ty::Int) => Some(Ty::Int),
            _ if numeric(a) && numeric(b) => Some(Ty::Float),
            _ => None,
        },
        EvaluationOp::And | EvaluationOp::Or => match (a, b) {
            (Ty::Bool, Ty::Bool) => Some(Ty::Bool),
            _ => None,
        },
        EvaluationOp::BitShiftLeft | EvaluationOp::BitShiftRight | EvaluationOp::Xor => {
            Some(Ty::Int)
        }
    }
}

/// How certainly a column expression's dimension is known. `Q` is a unit-carrying column
/// or literal, `Plain` is definitely a plain number (a bare literal, a `float`/`int`
/// column, `count()`), and `Any` can't be told (a `$` splice, an opaque call) -- `Any`
/// never causes an error, same as the dims pass.
#[derive(Debug, Clone, Copy, PartialEq)]
enum CDim {
    Any,
    Plain,
    Q(Dim),
}

impl CDim {
    /// The unit a column carries, for the schema it ends up in (`Any`/`Plain` carry none).
    fn dim(self) -> Option<Dim> {
        match self {
            CDim::Q(d) => Some(d),
            _ => None,
        }
    }
}

/// A column's dimension status: a declared unit is `Q`, a numeric column with no unit is
/// `Plain`, and anything else is out of the checker's reach.
fn cdim(col: &FrameCol) -> CDim {
    match (&col.ty, col.dim) {
        (_, Some(d)) => CDim::Q(d),
        (Some(Ty::Int) | Some(Ty::Float), _) => CDim::Plain,
        _ => CDim::Any,
    }
}

/// `a` and `b` side by side where the dims pass demands the same dimension (`+ - %`,
/// comparisons, `??`, `if` arms): two *known* sides that disagree. `Plain` counts as
/// `Dim::NONE`, so `kg_col + 3` is caught while `kg_col + $x` stays permissive.
fn dim_clash(a: CDim, b: CDim) -> Option<(Dim, Dim)> {
    match (a, b) {
        (CDim::Q(x), CDim::Q(y)) if x != y => Some((x, y)),
        (CDim::Q(x), CDim::Plain) if !x.is_none() => Some((x, Dim::NONE)),
        (CDim::Plain, CDim::Q(y)) if !y.is_none() => Some((Dim::NONE, y)),
        _ => None,
    }
}

/// The [`CDim`] `a <op> b` computes, after `dim_clash` has had its say on `+ - %`.
/// `*`/`/` compose dims like the dims pass (`kg * usd/kg` is `usd`), a `Plain` operand
/// scales without changing the dimension, and `Any` defers to the known side.
fn cdim_join(op: EvaluationOp, a: CDim, b: CDim) -> CDim {
    match op {
        EvaluationOp::Multiply => match (a, b) {
            (CDim::Q(x), CDim::Q(y)) => CDim::Q(x.mul(y)),
            (CDim::Q(x), _) | (_, CDim::Q(x)) => CDim::Q(x),
            (CDim::Plain, CDim::Plain) => CDim::Plain,
            _ => CDim::Any,
        },
        EvaluationOp::Divide | EvaluationOp::Div => match (a, b) {
            (CDim::Q(x), CDim::Q(y)) => CDim::Q(x.div(y)),
            (CDim::Q(x), _) => CDim::Q(x),
            (CDim::Plain, CDim::Q(y)) => CDim::Q(Dim::NONE.div(y)),
            (CDim::Plain, CDim::Plain) => CDim::Plain,
            _ => CDim::Any,
        },
        // `+ - %` only reach here without a clash: an agreed dimension, a plain pair, or
        // something unknown -- the known side wins where there is one
        EvaluationOp::Plus | EvaluationOp::Minus | EvaluationOp::Modulo => match (a, b) {
            (CDim::Q(x), CDim::Q(_)) | (CDim::Q(x), CDim::Any) | (CDim::Any, CDim::Q(x)) => {
                CDim::Q(x)
            }
            (CDim::Plain, CDim::Plain) => CDim::Plain,
            _ => CDim::Any,
        },
        _ => CDim::Any,
    }
}

/// A literal `0`, which dims allows on either side of a comparison (`mass > 0`).
fn is_zero(e: &Expr) -> bool {
    match e.kind() {
        ExprKind::Literal(Literal::Int(0)) => true,
        ExprKind::Literal(Literal::Float(f)) => *f == 0.0,
        ExprKind::Grouping(g) => is_zero(&g.inner),
        _ => false,
    }
}

/// The column an aggregation/`PlExpr` method produces from `base`.
fn apply_result_col(f: &str, base: FrameCol) -> FrameCol {
    match f {
        "count" | "n" | "n_unique" | "len" | "arg_min" | "arg_max" | "rank" => {
            FrameCol::known(Ty::Int)
        }
        // a cast changes the storage, not what the numbers measure: `cast_int` is an
        // int quantity (like `.to_int()`), `cast_float` keeps whatever unit it had
        "cast_int" => FrameCol {
            ty: Some(Ty::Int),
            dim: base.dim,
        },
        "mean" | "average" | "median" | "quantile" | "std" | "var" | "cast_float" => FrameCol {
            ty: Some(Ty::Float),
            dim: match f {
                "mean" | "average" | "median" | "quantile" | "std" | "cast_float" => base.dim,
                "var" => base.dim.map(|d| d.mul(d)),
                _ => None,
            },
        },
        "is_null" | "is_not_null" | "contains" | "contains_many" | "starts_with" | "ends_with"
        | "is_in" | "is_between" | "eq" | "neq" => FrameCol::known(Ty::Bool),
        "to_upper" | "to_lower" | "strip" | "cast_str" | "str_slice" | "str_replace" => {
            FrameCol::known(Ty::Str)
        }
        "unique" | "sort" | "sort_desc" | "reverse" => FrameCol {
            ty: base.ty.clone().map(|t| Ty::Array(Box::new(t))),
            dim: base.dim,
        },
        // sum/min/max/first/last/fill_null and anything unlisted: same shape out as in
        _ => base,
    }
}

impl Solver {
    /// The schema of the frame `e` evaluates to, when tracked.
    pub(crate) fn frame_schema_of(&self, e: &Expr) -> Option<Rc<FrameSchema>> {
        match e.kind() {
            ExprKind::Grouping(g) => self.frame_schema_of(&g.inner),
            ExprKind::Unwrap(u) => self.frame_schema_of(&u.expr),
            ExprKind::Demote(d) => self.frame_schema_of(&d.expr),
            ExprKind::Absolve(a) => self.frame_schema_of(&a.left),
            ExprKind::Ident(ident) => self
                .node_decs
                .get(&e.id())
                .or_else(|| self.node_decs.get(&ident.id))
                .and_then(|dec| self.frame_decs.get(dec))
                .cloned(),
            _ => self.frame_schemas.get(&e.id()).cloned(),
        }
    }

    /// The `(input schema, keys)` of a `group_by`/`__q_group` result.
    fn group_schema_of(&self, e: &Expr) -> Option<Rc<(Rc<FrameSchema>, Vec<String>)>> {
        match e.kind() {
            ExprKind::Grouping(g) => self.group_schema_of(&g.inner),
            ExprKind::Unwrap(u) => self.group_schema_of(&u.expr),
            ExprKind::Demote(d) => self.group_schema_of(&d.expr),
            ExprKind::Absolve(a) => self.group_schema_of(&a.left),
            ExprKind::Ident(ident) => self
                .node_decs
                .get(&e.id())
                .or_else(|| self.node_decs.get(&ident.id))
                .and_then(|dec| self.group_decs.get(dec))
                .cloned(),
            _ => self.group_schemas.get(&e.id()).cloned(),
        }
    }

    /// Called from the `let` handler: propagate the RHS's frame/groupby schema onto the
    /// declared binding, so `let df = table {…}` makes `df.pull` typed by name.
    pub(crate) fn propagate_frame_schemas(&mut self, pat: &Pat, rhs: &Expr) {
        let frame = self.frame_schema_of(rhs);
        let group = self.group_schema_of(rhs);
        if frame.is_none() && group.is_none() {
            return;
        }
        match pat.kind() {
            PatKind::Ident(ident) => {
                let dec = self
                    .node_decs
                    .get(&pat.id())
                    .or_else(|| self.node_decs.get(&ident.id))
                    .copied();
                if let Some(dec) = dec {
                    if let Some(s) = frame {
                        self.frame_decs.insert(dec, s);
                    }
                    if let Some(g) = group {
                        self.group_decs.insert(dec, g);
                    }
                }
            }
            PatKind::NullBind(inner) => self.propagate_frame_schemas(inner, rhs),
            _ => {}
        }
    }

    /// A `"lit"` string argument at `call.arguments[n]`.
    fn str_arg<'c>(call: &'c Call, n: usize) -> Option<&'c str> {
        match call.arguments.get(n).map(|a| a.value.kind()) {
            Some(ExprKind::Literal(Literal::String(s))) => Some(s.as_str()),
            _ => None,
        }
    }

    /// A `["a","b"]` string-array argument (the `q_str_list` the parser emits for verb keys).
    fn str_list_arg<'c>(call: &'c Call, n: usize) -> Option<Vec<&'c str>> {
        match call.arguments.get(n).map(|a| a.value.kind()) {
            Some(ExprKind::Literal(Literal::Array(items))) => items
                .iter()
                .map(|i| match i.kind() {
                    ExprKind::Literal(Literal::String(s)) => Some(s.as_str()),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }

    pub(crate) fn schema_columns(schema: &FrameSchema) -> String {
        schema.cols.keys().cloned().collect::<Vec<_>>().join(", ")
    }

    pub(crate) fn no_such_column(&self, at: &Expr, name: &str, schema: &FrameSchema) -> Error {
        NoSuchColumn {
            src: self.src(at.location()),
            at: at.location().into(),
            name: name.to_string(),
            columns: Self::schema_columns(schema),
        }
        .into()
    }

    /// `df.schema("fare:float kwh:kWh")` -- parse the spec into a schema; the runtime
    /// casts/checks the same spec so check-time and run-time agree.
    fn schema_from_spec(&self, spec: &str, at: &Expr) -> Result<FrameSchema> {
        let entries = shared::schema::parse_schema(spec).map_err(|msg| BadSchemaSpec {
            src: self.src(at.location()),
            at: at.location().into(),
            msg,
        })?;
        let mut cols = IndexMap::new();
        for (name, ty) in entries {
            let (ty, dim) = match ty {
                shared::schema::SchemaTy::Int => (Ty::Int, None),
                shared::schema::SchemaTy::Float => (Ty::Float, None),
                shared::schema::SchemaTy::Str => (Ty::Str, None),
                shared::schema::SchemaTy::Bool => (Ty::Bool, None),
                shared::schema::SchemaTy::Unit(d) => (Ty::Float, Some(d)),
            };
            cols.insert(name, FrameCol { ty: Some(ty), dim });
        }
        Ok(FrameSchema { cols, closed: true })
    }

    /// A `cannot <verb> X and Y`-style dimension error inside a column expression -- the
    /// same `DimensionMismatch` the dims pass gives scalars.
    fn dim_err(&self, at: &Expr, what: String, label: String, help: &str) -> Error {
        DimensionMismatch {
            src: self.src(at.location()),
            at: at.location().into(),
            what,
            label,
            help: help.to_string(),
        }
        .into()
    }

    /// The column a query/derive/agg column-expression computes, given the input schema.
    /// Handles the `__q_*`/`col()`/`lit()`/`n()` forms, `| alias` / `.alias()`, method
    /// chains on `col(…)`, and arithmetic between them. A bare `Ident` here is a `$expr`
    /// splice (or a plain value in the method api) -- an outer-scope value, so its solved
    /// type broadcasts to the column.
    fn col_expr_col(&mut self, e: &Expr, input: &FrameSchema) -> Result<FrameCol> {
        self.col_expr(e, input).map(|(col, _)| col)
    }

    /// `col_expr_col` plus the expression's [`CDim`], so compound column expressions can
    /// enforce the dims pass's same-dimension rules inside `filter`/`derive`/friends:
    /// `mass > 2kg` is a fine predicate on a `kg` column, `mass > 2s` is a check-time
    /// error, and `derive cost = price * n` gives `cost` the composed dimension.
    fn col_expr(&mut self, e: &Expr, input: &FrameSchema) -> Result<(FrameCol, CDim)> {
        let opaque = |this: &mut Self, e: &Expr| -> Result<(FrameCol, CDim)> {
            // a splice escape or a plain call -- an ordinary value in scope whose solved
            // ty broadcasts to every row of the column, with a dim we can't see
            Ok((
                FrameCol {
                    ty: Some(e.query(this)?.normalized(this)),
                    dim: None,
                },
                CDim::Any,
            ))
        };
        match e.kind() {
            ExprKind::Grouping(g) => self.col_expr(&g.inner, input),
            ExprKind::Unwrap(u) => self.col_expr(&u.expr, input),
            ExprKind::Demote(d) => self.col_expr(&d.expr, input),
            ExprKind::Absolve(a) => self.col_expr(&a.left, input),
            ExprKind::Literal(Literal::Int(_)) => Ok((FrameCol::known(Ty::Int), CDim::Plain)),
            ExprKind::Literal(Literal::Float(_)) => {
                let d = self
                    .ast_quantities
                    .get(&e.id())
                    .map_or(CDim::Plain, |d| CDim::Q(*d));
                Ok((
                    FrameCol {
                        ty: Some(Ty::Float),
                        dim: d.dim(),
                    },
                    d,
                ))
            }
            ExprKind::Literal(Literal::String(_)) => Ok((FrameCol::known(Ty::Str), CDim::Any)),
            ExprKind::Literal(Literal::True) | ExprKind::Literal(Literal::False) => {
                Ok((FrameCol::known(Ty::Bool), CDim::Any))
            }
            ExprKind::Call(c) => {
                if let Some(ident) = c.left.as_ident() {
                    return match ident.lexeme.as_str() {
                        "__q_col" | "col" => match Self::str_arg(c, 0) {
                            Some(col) => {
                                let col = input.col(self, e, col)?;
                                let d = cdim(&col);
                                Ok((col, d))
                            }
                            None => Ok((FrameCol::unknown(), CDim::Any)),
                        },
                        "__q_n" | "n" => Ok((FrameCol::known(Ty::Int), CDim::Plain)),
                        // `lit(2kg)` keeps the literal's own type and unit
                        "lit" => match c.arguments.first() {
                            Some(a) => self.col_expr(&a.value, input),
                            None => Ok((FrameCol::unknown(), CDim::Any)),
                        },
                        "__q_when" => {
                            if let Some(cond) = c.arguments.first() {
                                self.col_expr(&cond.value, input)?;
                            }
                            let (t, t_d) = match c.arguments.get(1) {
                                Some(a) => self.col_expr(&a.value, input)?,
                                None => (FrameCol::unknown(), CDim::Any),
                            };
                            let (f, f_d) = match c.arguments.get(2) {
                                Some(a) => self.col_expr(&a.value, input)?,
                                None => (FrameCol::unknown(), CDim::Any),
                            };
                            // `if c { mass } else { 3s }` -- the arms are a column each,
                            // so they share the dims pass's branches-agree rule
                            if let Some((x, y)) = dim_clash(t_d, f_d) {
                                return Err(self.dim_err(
                                    e,
                                    format!(
                                        "branches disagree: {} and {}",
                                        x.describe(),
                                        y.describe()
                                    ),
                                    format!("this branch is {}", y.describe()),
                                    "both arms of an `if` column have to give the same dimension",
                                ));
                            }
                            let d = match (t_d, f_d) {
                                (CDim::Q(x), CDim::Q(y)) if x == y => CDim::Q(x),
                                (CDim::Plain, CDim::Plain) => CDim::Plain,
                                _ => CDim::Any,
                            };
                            Ok((
                                FrameCol {
                                    ty: t.ty.or(f.ty),
                                    dim: d.dim(),
                                },
                                d,
                            ))
                        }
                        "__q_apply" => {
                            let (base, base_d) = match c.arguments.first() {
                                Some(a) => self.col_expr(&a.value, input)?,
                                None => (FrameCol::unknown(), CDim::Any),
                            };
                            let func = Self::str_arg(c, 1).unwrap_or("");
                            if let Some(extra) = c.arguments.get(2) {
                                match func {
                                    // `eq(mass, 2s)` is a comparison against the column
                                    "eq" | "neq" => {
                                        let (_, arg_d) =
                                            self.col_expr(&extra.value, input)?;
                                        if !is_zero(&extra.value)
                                            && let Some((x, y)) = dim_clash(base_d, arg_d)
                                        {
                                            return Err(self.dim_err(
                                                &extra.value,
                                                format!(
                                                    "cannot compare {} and {}",
                                                    x.describe(),
                                                    y.describe()
                                                ),
                                                format!(
                                                    "{} against {}",
                                                    x.describe(),
                                                    y.describe()
                                                ),
                                                "quantities are only comparable in the same dimension; convert one side with `.to(unit)`",
                                            ));
                                        }
                                    }
                                    // `is_in(mass, [2kg, 5kg])` compares each listed
                                    // cell against the column
                                    "is_in" => {
                                        if let ExprKind::Literal(Literal::Array(items)) =
                                            extra.value.kind()
                                        {
                                            for item in items {
                                                let (_, item_d) =
                                                    self.col_expr(item, input)?;
                                                if let Some((x, y)) =
                                                    dim_clash(base_d, item_d)
                                                {
                                                    return Err(self.dim_err(
                                                        item,
                                                        format!(
                                                            "cannot compare {} and {}",
                                                            x.describe(),
                                                            y.describe()
                                                        ),
                                                        format!(
                                                            "{} against {}",
                                                            x.describe(),
                                                            y.describe()
                                                        ),
                                                        "`is_in` cells are compared to the column, so they share its dimension",
                                                    ));
                                                }
                                            }
                                        } else {
                                            self.col_expr(&extra.value, input)?;
                                        }
                                    }
                                    // `is_in(discount, $codes)`-style extras can hold
                                    // outer values
                                    _ => {
                                        self.col_expr(&extra.value, input)?;
                                    }
                                }
                            }
                            let col = apply_result_col(func, base);
                            let d = cdim(&col);
                            Ok((col, d))
                        }
                        "__q_named" => match c.arguments.first() {
                            Some(a) => self.col_expr(&a.value, input),
                            None => Ok((FrameCol::unknown(), CDim::Any)),
                        },
                        _ => opaque(self, e),
                    };
                }
                if let ExprKind::Access(Access::Dot {
                    left: recv, right, ..
                }) = c.left.kind()
                    && let Some(m) = right.as_ident().map(|i| i.lexeme.as_str())
                {
                    if m == "alias" {
                        return self.col_expr(recv, input);
                    }
                    let (base, base_d) = self.col_expr(recv, input)?;
                    // `mass.is_in([2kg, 3s])` -- the listed cells compare against the column
                    if m == "is_in"
                        && let Some(arg) = c.arguments.first()
                        && let ExprKind::Literal(Literal::Array(items)) = arg.value.kind()
                    {
                        for item in items {
                            let (_, item_d) = self.col_expr(item, input)?;
                            if let Some((x, y)) = dim_clash(base_d, item_d) {
                                return Err(self.dim_err(
                                    item,
                                    format!(
                                        "cannot compare {} and {}",
                                        x.describe(),
                                        y.describe()
                                    ),
                                    format!("{} against {}", x.describe(), y.describe()),
                                    "`is_in` cells are compared to the column, so they share its dimension",
                                ));
                            }
                        }
                    }
                    // `col("kwh").cast("int")` is an int quantity column; other casts
                    // that rename the storage keep whatever it measured
                    if m == "cast" {
                        let col = match Self::str_arg(c, 0) {
                            Some("int") => FrameCol {
                                ty: Some(Ty::Int),
                                dim: base_d.dim(),
                            },
                            Some("float") => FrameCol {
                                ty: Some(Ty::Float),
                                dim: base_d.dim(),
                            },
                            _ => base,
                        };
                        let d = cdim(&col);
                        return Ok((col, d));
                    }
                    let col = apply_result_col(m, base);
                    let d = cdim(&col);
                    return Ok((col, d));
                }
                Ok((FrameCol::unknown(), CDim::Any))
            }
            ExprKind::Evaluation(ev) => {
                let (a, a_d) = self.col_expr(&ev.left, input)?;
                let (b, b_d) = self.col_expr(&ev.right, input)?;
                if matches!(
                    ev.op,
                    EvaluationOp::Plus | EvaluationOp::Minus | EvaluationOp::Modulo
                ) && let Some((x, y)) = dim_clash(a_d, b_d)
                {
                    let verb = match ev.op {
                        EvaluationOp::Plus => "add",
                        EvaluationOp::Minus => "subtract",
                        _ => "combine",
                    };
                    return Err(self.dim_err(
                        e,
                        format!("cannot {verb} {} and {}", x.describe(), y.describe()),
                        format!("{} with {}", x.describe(), y.describe()),
                        "quantities only add and subtract in the same dimension -- convert one with `.to(unit)`, or multiply/divide to make a new quantity",
                    ));
                }
                let d = cdim_join(ev.op, a_d, b_d);
                Ok((
                    FrameCol {
                        ty: numeric_join(ev.op, &a.ty, &b.ty),
                        dim: d.dim(),
                    },
                    d,
                ))
            }
            // `a > 10`, `x & y`: the result is bool, but the operands' column refs still
            // need checking -- descend for errors, apply the same-dimension rule
            ExprKind::Equality(eq) => {
                let (_, a_d) = self.col_expr(&eq.left, input)?;
                let (_, b_d) = self.col_expr(&eq.right, input)?;
                if !is_zero(&eq.left)
                    && !is_zero(&eq.right)
                    && let Some((x, y)) = dim_clash(a_d, b_d)
                {
                    let verb = match eq.op {
                        EqualityOp::Equal | EqualityOp::NotEqual => "compare",
                        _ => "order",
                    };
                    return Err(self.dim_err(
                        e,
                        format!("cannot {verb} {} and {}", x.describe(), y.describe()),
                        format!("{} against {}", x.describe(), y.describe()),
                        "quantities are only comparable in the same dimension; convert one side with `.to(unit)`",
                    ));
                }
                Ok((FrameCol::known(Ty::Bool), CDim::Any))
            }
            ExprKind::Logical(l) => {
                self.col_expr(&l.left, input)?;
                self.col_expr(&l.right, input)?;
                Ok((FrameCol::known(Ty::Bool), CDim::Any))
            }
            ExprKind::Coalescence(co) => {
                let (l, l_d) = self.col_expr(&co.left, input)?;
                let (_, r_d) = self.col_expr(&co.right, input)?;
                if let Some((x, y)) = dim_clash(l_d, r_d) {
                    return Err(self.dim_err(
                        &co.right,
                        format!(
                            "`??` fallback has a different dimension: {} vs {}",
                            x.describe(),
                            y.describe()
                        ),
                        format!("this is {}", y.describe()),
                        "the fallback of `??` has to measure the same thing as the value it's replacing",
                    ));
                }
                Ok((l, l_d))
            }
            ExprKind::Unary(u) => match u.op {
                UnaryOp::Negative | UnaryOp::Positive => self.col_expr(&u.right, input),
                UnaryOp::Not => {
                    self.col_expr(&u.right, input)?;
                    Ok((FrameCol::known(Ty::Bool), CDim::Any))
                }
                UnaryOp::BitwiseNot => {
                    self.col_expr(&u.right, input)?;
                    Ok((FrameCol::known(Ty::Int), CDim::Plain))
                }
            },
            // a bare ident in a column expr is an outer-scope value (all column-name reads
            // were already rewritten to `__q_col`), so its own type broadcasts
            ExprKind::Ident(_) => opaque(self, e),
            _ => Ok((FrameCol::unknown(), CDim::Any)),
        }
    }

    /// The name a select/derive/agg item produces: `__q_named(_, "n")`, `.alias("n")`,
    /// or the root name polars keeps (`col("n")`, `col("n").sum()` → `n`).
    fn col_expr_name(&self, e: &Expr) -> Option<String> {
        match e.kind() {
            ExprKind::Call(c) => {
                if let Some(ident) = c.left.as_ident() {
                    return match ident.lexeme.as_str() {
                        "__q_named" => Self::str_arg(c, 1).map(str::to_string),
                        "__q_col" | "col" => Self::str_arg(c, 0).map(str::to_string),
                        "__q_n" | "n" => Some("n".into()),
                        _ => None,
                    };
                }
                if let ExprKind::Access(Access::Dot { left, right, .. }) = c.left.kind() {
                    if right.as_ident().is_some_and(|i| i.lexeme == "alias") {
                        return Self::str_arg(c, 0).map(str::to_string);
                    }
                    return self.col_expr_name(left);
                }
                None
            }
            ExprKind::Grouping(g) => self.col_expr_name(&g.inner),
            _ => None,
        }
    }

    /// The schema `__q_agg`/`gb.agg(…)` produces: group keys in input order, then the
    /// named aggregates -- polars's output column order. `at` locates key errors.
    fn agg_schema(
        &mut self,
        input: &FrameSchema,
        keys: &[String],
        list: Option<&Expr>,
        at: &Expr,
    ) -> Result<FrameSchema> {
        let mut cols = IndexMap::new();
        for key in keys {
            cols.insert(key.clone(), input.col(self, at, key)?);
        }
        // a list we can't enumerate (a splice, a variable) may grow columns we
        // can't name -- keep what we derived but stay open
        let mut closed = match list {
            Some(list) if matches!(list.kind(), ExprKind::Literal(Literal::Array(_))) => {
                input.closed
            }
            _ => false,
        };
        if let Some(list) = list
            && let ExprKind::Literal(Literal::Array(items)) = list.kind()
        {
            for item in items {
                match self.col_expr_name(item) {
                    Some(name) => {
                        cols.insert(name, self.col_expr_col(item, input)?);
                    }
                    None => closed = false,
                }
            }
        }
        Ok(FrameSchema { cols, closed })
    }

    /// Post-solve hook for calls (runs at the tail of `Call::solve`, after arguments are
    /// fulfilled): record propagated schemas and refine `pull`'s generic return into the
    /// column's known `[T]`.
    pub(crate) fn frame_call(&mut self, id: NodeId, call: &Call, ty: Ty) -> Result<Ty> {
        match call.left.kind() {
            // `df.pull("x")`, `df.schema("…")`, `df.filter(…)`, `gb.agg(…)`
            ExprKind::Access(Access::Dot {
                left: recv, right, ..
            }) => {
                let Some(method) = right.as_ident().map(|i| i.lexeme.clone()) else {
                    return Ok(ty);
                };
                self.frame_method_call(id, call, recv, &method, ty)
            }
            // query-mode natives: `__q_filter(df, …)`, `__table_col(df, "x", […])`
            ExprKind::Ident(callee) => {
                let arg = |n: usize| call.arguments.get(n).map(|a| &a.value);
                match callee.lexeme.as_str() {
                    "__table_new" => {
                        self.frame_schemas.insert(id, Rc::new(FrameSchema::empty()));
                    }
                    "__table_col" => self.table_col(id, call)?,
                    "__q_filter" => {
                        if let Some(schema) = arg(0).and_then(|a| self.frame_schema_of(a)) {
                            if let Some(cond) = arg(1) {
                                self.col_expr_col(cond, &schema)?;
                            }
                            self.frame_schemas.insert(id, schema);
                        }
                    }
                    // keys checked when they're a literal list; all keep every column
                    "__q_sort" | "__q_distinct" => {
                        if let Some(schema) = arg(0).and_then(|a| self.frame_schema_of(a)) {
                            if let (Some(names), Some(at)) = (Self::str_list_arg(call, 1), arg(1)) {
                                for n in names {
                                    schema.col(self, at, n)?;
                                }
                            }
                            self.frame_schemas.insert(id, schema);
                        }
                    }
                    "__q_take" => {
                        if let Some(schema) = arg(0).and_then(|a| self.frame_schema_of(a)) {
                            self.frame_schemas.insert(id, schema);
                        }
                    }
                    "__q_select" => self.q_select(id, call)?,
                    "__q_rename" => self.q_rename(id, call)?,
                    "__q_mutate" => self.q_mutate(id, call)?,
                    "__q_group" => self.q_group(id, call)?,
                    "__q_agg" => self.q_agg(id, call)?,
                    "__q_join" => self.q_join(id, call)?,
                    _ => {}
                }
                Ok(ty)
            }
            _ => Ok(ty),
        }
    }

    /// `df.<method>(…)` schema handling.
    fn frame_method_call(
        &mut self,
        id: NodeId,
        call: &Call,
        recv: &Expr,
        method: &str,
        ty: Ty,
    ) -> Result<Ty> {
        match method {
            "pull" => {
                let (Some(schema), Some(name)) =
                    (self.frame_schema_of(recv), Self::str_arg(call, 0))
                else {
                    return Ok(ty);
                };
                match schema.col(self, &call.arguments[0].value, name)? {
                    FrameCol { ty: Some(t), .. } => {
                        Ok(Ty::Result(Box::new(Ty::Array(Box::new(t)))))
                    }
                    _ => Ok(ty),
                }
            }
            // a unit column pulled as a unit is still `[float]` at the Ty layer
            "pull_as" => Ok(match self.frame_schema_of(recv) {
                Some(_) => Ty::Result(Box::new(Ty::Array(Box::new(Ty::Float)))),
                None => ty,
            }),
            // `x.to("usd")`: the only runtime failure is an unparseable unit
            // name, so a literal the units table knows can't raise — keep the
            // plain inner type. A computed unit string keeps the honest `T!`.
            "to" => {
                if let Some(ExprKind::Literal(Literal::String(u))) =
                    call.arguments.first().map(|a| a.value.kind())
                    && shared::units::parse(u).is_some()
                    && let Ty::Result(inner) = &ty
                {
                    return Ok(inner.as_ref().clone());
                }
                Ok(ty)
            }
            "schema" => self.apply_schema_spec(id, call, recv, ty),
            "filter" => {
                let Some(schema) = self.frame_schema_of(recv) else {
                    return Ok(ty);
                };
                for arg in &call.arguments {
                    self.col_expr_col(&arg.value, &schema)?;
                }
                self.frame_schemas.insert(id, schema);
                Ok(ty)
            }
            "sort" | "arrange" | "distinct" | "drop_nulls" => {
                let Some(schema) = self.frame_schema_of(recv) else {
                    return Ok(ty);
                };
                for i in 0..call.arguments.len() {
                    if let Some(names) = Self::str_list_arg(call, i) {
                        for n in names {
                            schema.col(self, &call.arguments[i].value, n)?;
                        }
                    }
                }
                self.frame_schemas.insert(id, schema);
                Ok(ty)
            }
            "head" | "tail" | "slice" | "take" | "sample" => {
                if let Some(s) = self.frame_schema_of(recv) {
                    self.frame_schemas.insert(id, s);
                }
                Ok(ty)
            }
            "select_names" => self.q_select_names(id, call, recv, ty),
            "rename" => {
                let Some(schema) = self.frame_schema_of(recv) else {
                    return Ok(ty);
                };
                let (Some(from), Some(to)) = (Self::str_arg(call, 0), Self::str_arg(call, 1))
                else {
                    return Ok(ty);
                };
                let col = schema.col(self, &call.arguments[0].value, from)?;
                let mut cols = schema.cols.clone();
                if let Some(idx) = cols.get_index_of(from) {
                    cols.shift_remove(from);
                    cols.insert_before(idx.min(cols.len()), to.to_string(), col);
                }
                self.frame_schemas.insert(
                    id,
                    Rc::new(FrameSchema {
                        cols,
                        closed: schema.closed,
                    }),
                );
                Ok(ty)
            }
            "select" => self.select_exprs(id, call, recv, ty),
            "mutate" | "with_columns" => {
                let Some(schema) = self.frame_schema_of(recv) else {
                    return Ok(ty);
                };
                let mut out = schema.as_ref().clone();
                if let Some(ExprKind::Literal(Literal::Array(items))) =
                    call.arguments.first().map(|a| a.value.kind())
                {
                    for item in items {
                        match self.col_expr_name(item) {
                            Some(name) => {
                                out.cols.insert(name, self.col_expr_col(item, &schema)?);
                            }
                            None => {
                                self.col_expr_col(item, &schema)?;
                                out.closed = false;
                            }
                        }
                    }
                }
                self.frame_schemas.insert(id, Rc::new(out));
                Ok(ty)
            }
            "group_by" => {
                let Some(schema) = self.frame_schema_of(recv) else {
                    return Ok(ty);
                };
                let keys: Vec<String> = Self::str_list_arg(call, 0)
                    .map(|v| v.into_iter().map(str::to_string).collect())
                    .unwrap_or_default();
                if let Some(at) = call.arguments.first().map(|a| &a.value) {
                    for k in &keys {
                        schema.col(self, at, k)?;
                    }
                }
                self.group_schemas.insert(id, Rc::new((schema, keys)));
                Ok(ty)
            }
            "agg" | "aggregate" => {
                let Some(grouped) = self.group_schema_of(recv) else {
                    return Ok(ty);
                };
                let (input, keys) = grouped.as_ref().clone();
                let schema = self.agg_schema(
                    &input,
                    &keys,
                    call.arguments.first().map(|a| &a.value),
                    &call.left,
                )?;
                self.frame_schemas.insert(id, Rc::new(schema));
                Ok(ty)
            }
            "join" => {
                self.join_schemas(
                    id,
                    call,
                    Some(recv),
                    call.arguments.first().map(|a| &a.value),
                    1,
                )?;
                Ok(ty)
            }
            _ => Ok(ty),
        }
    }

    /// `df.select([col("a"), (col("b") * 2).alias("c")])` -- project to the named items;
    /// an item we can't name still gets its column refs checked.
    fn select_exprs(&mut self, id: NodeId, call: &Call, recv: &Expr, ty: Ty) -> Result<Ty> {
        let Some(schema) = self.frame_schema_of(recv) else {
            return Ok(ty);
        };
        let Some(ExprKind::Literal(Literal::Array(items))) =
            call.arguments.first().map(|a| a.value.kind())
        else {
            return Ok(ty);
        };
        let mut cols = IndexMap::new();
        let mut closed = schema.closed;
        for item in items {
            match self.col_expr_name(item) {
                Some(name) => {
                    cols.insert(name, self.col_expr_col(item, &schema)?);
                }
                // a splice may add columns we can't name
                None => {
                    self.col_expr_col(item, &schema)?;
                    closed = false;
                }
            }
        }
        self.frame_schemas
            .insert(id, Rc::new(FrameSchema { cols, closed }));
        Ok(ty)
    }

    /// `df.select_names(["a","b"])` -- project by name list.
    fn q_select_names(&mut self, id: NodeId, call: &Call, recv: &Expr, ty: Ty) -> Result<Ty> {
        let Some(schema) = self.frame_schema_of(recv) else {
            return Ok(ty);
        };
        let Some(names) = Self::str_list_arg(call, 0) else {
            return Ok(ty);
        };
        let mut cols = IndexMap::new();
        for n in names {
            cols.insert(
                n.to_string(),
                schema.col(self, &call.arguments[0].value, n)?,
            );
        }
        self.frame_schemas.insert(
            id,
            Rc::new(FrameSchema {
                cols,
                closed: schema.closed,
            }),
        );
        Ok(ty)
    }

    /// `df.schema("…")` -- parse the spec, error against a closed receiver schema when it
    /// names a missing column, and record the schema on the call's result.
    fn apply_schema_spec(&mut self, id: NodeId, call: &Call, recv: &Expr, ty: Ty) -> Result<Ty> {
        let Some(spec) = Self::str_arg(call, 0) else {
            return Ok(ty);
        };
        let at = &call.arguments[0].value;
        let schema = self.schema_from_spec(spec, at)?;
        if let Some(recv_schema) = self.frame_schema_of(recv)
            && recv_schema.closed
        {
            for name in schema.cols.keys() {
                if !recv_schema.cols.contains_key(name) {
                    return Err(self.no_such_column(at, name, &recv_schema));
                }
            }
        }
        self.frame_schemas.insert(id, Rc::new(schema));
        Ok(ty)
    }

    /// `__table_col(df, "name", [cells])` -- extend the schema with the literal's element
    /// type (and the unit, when the cells are `5kWh`-style literals).
    fn table_col(&mut self, id: NodeId, call: &Call) -> Result<()> {
        let Some(schema) = call
            .arguments
            .first()
            .and_then(|a| self.frame_schema_of(&a.value))
        else {
            return Ok(());
        };
        let mut out = schema.as_ref().clone();
        let Some(name) = Self::str_arg(call, 1) else {
            out.closed = false;
            self.frame_schemas.insert(id, Rc::new(out));
            return Ok(());
        };
        let (ty, dim) = match call.arguments.get(2).map(|a| &a.value) {
            Some(e) => {
                let dim = match e.kind() {
                    ExprKind::Literal(Literal::Array(items)) => items
                        .iter()
                        .find_map(|i| self.ast_quantities.get(&i.id()).copied()),
                    _ => self.ast_quantities.get(&e.id()).copied(),
                };
                let ty = match e.query(self)?.normalized(self) {
                    Ty::Array(inner) => Some(*inner),
                    _ => None,
                };
                (ty, dim)
            }
            None => (None, None),
        };
        out.cols.insert(name.to_string(), FrameCol { ty, dim });
        self.frame_schemas.insert(id, Rc::new(out));
        Ok(())
    }

    /// `__q_select(t, ["a","b"])` -- project in the listed order.
    fn q_select(&mut self, id: NodeId, call: &Call) -> Result<()> {
        let Some(schema) = call
            .arguments
            .first()
            .and_then(|a| self.frame_schema_of(&a.value))
        else {
            return Ok(());
        };
        let (Some(names), Some(at)) = (
            Self::str_list_arg(call, 1),
            call.arguments.get(1).map(|a| &a.value),
        ) else {
            return Ok(());
        };
        let mut cols = IndexMap::new();
        for n in names {
            cols.insert(n.to_string(), schema.col(self, at, n)?);
        }
        self.frame_schemas.insert(
            id,
            Rc::new(FrameSchema {
                cols,
                closed: schema.closed,
            }),
        );
        Ok(())
    }

    /// `__q_rename(t, "old", "new")`.
    fn q_rename(&mut self, id: NodeId, call: &Call) -> Result<()> {
        let Some(schema) = call
            .arguments
            .first()
            .and_then(|a| self.frame_schema_of(&a.value))
        else {
            return Ok(());
        };
        let Some((from, to)) = Self::str_arg(call, 1).zip(Self::str_arg(call, 2)) else {
            return Ok(());
        };
        let col = schema.col(self, &call.arguments[1].value, from)?;
        let mut cols = schema.cols.clone();
        if let Some(idx) = cols.get_index_of(from) {
            cols.shift_remove(from);
            cols.insert_before(idx.min(cols.len()), to.to_string(), col);
        }
        self.frame_schemas.insert(
            id,
            Rc::new(FrameSchema {
                cols,
                closed: schema.closed,
            }),
        );
        Ok(())
    }

    /// `__q_mutate(t, [__q_named(expr, "name")…])` -- the `derive` verb.
    fn q_mutate(&mut self, id: NodeId, call: &Call) -> Result<()> {
        let Some(schema) = call
            .arguments
            .first()
            .and_then(|a| self.frame_schema_of(&a.value))
        else {
            return Ok(());
        };
        let mut out = schema.as_ref().clone();
        if let Some(ExprKind::Literal(Literal::Array(items))) =
            call.arguments.get(1).map(|a| a.value.kind())
        {
            for item in items {
                match self.col_expr_name(item) {
                    Some(name) => {
                        out.cols.insert(name, self.col_expr_col(item, &schema)?);
                    }
                    None => {
                        self.col_expr_col(item, &schema)?;
                        out.closed = false;
                    }
                }
            }
        }
        self.frame_schemas.insert(id, Rc::new(out));
        Ok(())
    }

    /// `__q_group(t, ["k1","k2"])` -- stash `(schema, keys)` for the `__q_agg` that follows.
    fn q_group(&mut self, id: NodeId, call: &Call) -> Result<()> {
        let Some(schema) = call
            .arguments
            .first()
            .and_then(|a| self.frame_schema_of(&a.value))
        else {
            return Ok(());
        };
        let keys: Vec<String> = Self::str_list_arg(call, 1)
            .map(|v| v.into_iter().map(str::to_string).collect())
            .unwrap_or_default();
        if let Some(at) = call.arguments.get(1).map(|a| &a.value) {
            for k in &keys {
                schema.col(self, at, k)?;
            }
        }
        self.group_schemas.insert(id, Rc::new((schema, keys)));
        Ok(())
    }

    /// `__q_agg(grouped, [__q_named(col_expr, "name")…])`.
    fn q_agg(&mut self, id: NodeId, call: &Call) -> Result<()> {
        let Some(grouped) = call
            .arguments
            .first()
            .and_then(|a| self.group_schema_of(&a.value))
        else {
            return Ok(());
        };
        let (input, keys) = grouped.as_ref().clone();
        let schema = self.agg_schema(
            &input,
            &keys,
            call.arguments.get(1).map(|a| &a.value),
            &call.left,
        )?;
        self.frame_schemas.insert(id, Rc::new(schema));
        Ok(())
    }

    /// `__q_join(t, other, ["key"], "kind")` -- left cols + the right's non-key cols.
    fn q_join(&mut self, id: NodeId, call: &Call) -> Result<()> {
        self.join_schemas(
            id,
            call,
            call.arguments.first().map(|a| &a.value),
            call.arguments.get(1).map(|a| &a.value),
            2,
        )
    }

    /// Shared join logic: keys are the str-list arg at `keys_arg` and must exist on the
    /// left; the output is the left schema plus the right's non-key columns.
    fn join_schemas(
        &mut self,
        id: NodeId,
        call: &Call,
        left: Option<&Expr>,
        right: Option<&Expr>,
        keys_arg: usize,
    ) -> Result<()> {
        let (Some(l), Some(r)) = (left, right) else {
            return Ok(());
        };
        let (Some(a), Some(b)) = (self.frame_schema_of(l), self.frame_schema_of(r)) else {
            return Ok(());
        };
        let (Some(keys), Some(at)) = (
            Self::str_list_arg(call, keys_arg),
            call.arguments.get(keys_arg).map(|a| &a.value),
        ) else {
            return Ok(());
        };
        for k in &keys {
            a.col(self, at, k)?;
            b.col(self, at, k)?;
        }
        let mut cols = a.cols.clone();
        for (name, col) in &b.cols {
            if keys.iter().any(|k| *k == name.as_str()) {
                continue;
            }
            // polars suffixes a duplicated non-key column with `_right`
            if cols.contains_key(name) {
                cols.insert(format!("{name}_right"), col.clone());
            } else {
                cols.insert(name.clone(), col.clone());
            }
        }
        self.frame_schemas.insert(
            id,
            Rc::new(FrameSchema {
                cols,
                closed: a.closed && b.closed,
            }),
        );
        Ok(())
    }
}

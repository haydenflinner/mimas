//! `std::polars` -- a thin, fluent-method-chain surface over a real `polars::frame::DataFrame`.
//! `DataFrame`/`PlExpr` are genuine `Val` variants (see `vm::val::DataFrame`/`vm::val::PlExpr`),
//! not `#[mimas] struct`s decomposed into `Val::Instance` fields -- `vm`'s own operator dispatch
//! (`bin()` in `crates/vm/src/val.rs`) already knows how to build a `polars::prelude::Expr` tree
//! out of `col("age") > 30 & col("city") == "SF"`, so this module only needs to wire up the
//! DataFrame-shaped verbs (`filter`/`select`/`sort`) plus the two ways to get a `DataFrame` in
//! the first place: `to_dataframe` (array-of-structs -> columns) and `from_csv` (an in-memory
//! CSV string -> columns, via polars' own reader).

use macros::native;
use vm::{
    Ctx, Val,
    api::Api,
    conversion::{DataFrameTy, GroupByTy, PlExprTy, Raisable},
};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    // must run before any `add`/`add_method` below whose signature mentions these types --
    // field/parameter types resolve eagerly through the registry.
    api.add_adt::<DataFrameTy>();
    api.add_adt::<PlExprTy>();
    api.add_adt::<GroupByTy>();
    {
        let mut m = api.module("std::polars");
        m.add(col);
        m.add(to_dataframe);
        m.add(from_csv);
    }
    api.add_method(filter);
    api.add_method(select);
    api.add_method(sort);
    api.add_method(group_by);
    api.add_method(agg);
    api.add_method(join);
    // Expr aggregation/naming methods -- `col("age").mean().alias("avg_age")`, used inside
    // `agg([..])`. Like `col()`, none of these can fail (they build a plan, they don't run one).
    api.add_method(sum);
    api.add_method(mean);
    api.add_method(median);
    api.add_method(min);
    api.add_method(max);
    api.add_method(count);
    api.add_method(n_unique);
    api.add_method(alias);
}

#[native]
fn col<'gc>(ctx: Ctx<'gc>, name: &str) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(polars::prelude::col(name))
}

// every query built here is a small in-memory transform -- `collect_with_engine(InMemory)`
// instead of plain `collect()` sidesteps `Engine::Auto` reaching for the streaming engine, which
// needs a polars feature this workspace doesn't enable (see the comment on the `polars`
// dependency in Cargo.toml).
fn collect_in_memory(lazy: polars::prelude::LazyFrame) -> polars::prelude::PolarsResult<polars::frame::DataFrame> {
    // `collect()`'s `Engine::Auto` picks `InMemory` whenever `opt_state.eager()` is set --
    // `_with_eager` is the only public way to set that flag (`QueryResult`/`Engine::InMemory`
    // aren't reachable through the `polars` umbrella crate's public API).
    lazy._with_eager(true).collect()
}

#[native]
fn filter<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    predicate: vm::PlExpr<'gc>,
) -> Raisable<vm::DataFrame<'gc>> {
    use polars::prelude::IntoLazy;
    let lazy = df.0.borrow().0.clone().lazy();
    collect_in_memory(lazy.filter(predicate.0.0.clone()))
        .map(|d| ctx.new_dataframe(d))
        .into()
}

#[native]
fn select<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    cols: Vec<vm::PlExpr<'gc>>,
) -> Raisable<vm::DataFrame<'gc>> {
    use polars::prelude::IntoLazy;
    let exprs: Vec<polars::prelude::Expr> = cols.into_iter().map(|e| e.0.0.clone()).collect();
    let lazy = df.0.borrow().0.clone().lazy();
    collect_in_memory(lazy.select(exprs))
        .map(|d| ctx.new_dataframe(d))
        .into()
}

#[native]
fn sort<'gc>(ctx: Ctx<'gc>, df: vm::DataFrame<'gc>, by: Vec<String>) -> Raisable<vm::DataFrame<'gc>> {
    let opts = polars::prelude::SortMultipleOptions::new();
    df.0.borrow()
        .0
        .sort(by, opts)
        .map(|d| ctx.new_dataframe(d))
        .into()
}

/// `df.group_by(["dept"]).agg([col("age").mean().alias("avg_age")])`. `group_by` alone can't
/// fail -- like `col()`, it just builds a plan -- `agg` is what actually runs it.
#[native]
fn group_by<'gc>(ctx: Ctx<'gc>, df: vm::DataFrame<'gc>, by: Vec<String>) -> vm::GroupBy<'gc> {
    use polars::prelude::{IntoLazy, col};
    let by: Vec<polars::prelude::Expr> = by.iter().map(|n| col(n.as_str())).collect();
    let lazy = df.0.borrow().0.clone().lazy();
    ctx.new_group_by(lazy.group_by(by))
}

#[native]
fn agg<'gc>(
    ctx: Ctx<'gc>,
    gb: vm::GroupBy<'gc>,
    aggs: Vec<vm::PlExpr<'gc>>,
) -> Raisable<vm::DataFrame<'gc>> {
    let exprs: Vec<polars::prelude::Expr> = aggs.into_iter().map(|e| e.0.0.clone()).collect();
    collect_in_memory(gb.0.0.clone().agg(exprs))
        .map(|d| ctx.new_dataframe(d))
        .into()
}

/// `df.join(other, ["id"], "inner")` -- the join key(s) must share a name on both sides (the
/// common case); for differently-named keys, `select`/`alias` one side to match first.
#[native]
fn join<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    other: vm::DataFrame<'gc>,
    on: Vec<String>,
    how: &str,
) -> Raisable<vm::DataFrame<'gc>> {
    use polars::prelude::{IntoLazy, JoinArgs, JoinType, col};
    let join_type = match how {
        "inner" => JoinType::Inner,
        "left" => JoinType::Left,
        "right" => JoinType::Right,
        "full" | "outer" => JoinType::Full,
        other => {
            return Raisable::Raised(format!(
                "join: unknown join type {other:?} (expected \"inner\", \"left\", \"right\", or \"full\")"
            ));
        }
    };
    let on: Vec<polars::prelude::Expr> = on.iter().map(|n| col(n.as_str())).collect();
    let left = df.0.borrow().0.clone().lazy();
    let right = other.0.borrow().0.clone().lazy();
    let joined = match left.join(right, on.clone(), on, JoinArgs::new(join_type)) {
        Ok(lf) => lf,
        Err(e) => return Raisable::Raised(e.to_string()),
    };
    collect_in_memory(joined).map(|d| ctx.new_dataframe(d)).into()
}

macro_rules! pl_expr_reducer {
    ($($name:ident),+ $(,)?) => {$(
        #[native]
        fn $name<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>) -> vm::PlExpr<'gc> {
            ctx.new_plexpr(e.0.0.clone().$name())
        }
    )+};
}
pl_expr_reducer!(sum, mean, median, min, max, count, n_unique);

#[native]
fn alias<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>, name: &str) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(e.0.0.clone().alias(name))
}

/// An array of same-shaped struct instances -> a `DataFrame` whose columns are named after the
/// struct's declared fields, in declaration order. Column dtype is inferred per-field from the
/// first row's value (int/float/bool/str only -- nested structs/arrays/dicts aren't flattened).
#[native]
fn to_dataframe<'gc>(ctx: Ctx<'gc>, rows: Vec<Val<'gc>>) -> Raisable<vm::DataFrame<'gc>> {
    let Some(first) = rows.first().and_then(|v| v.as_instance()) else {
        return Raisable::Raised(
            "to_dataframe: expected a non-empty array of struct instances".into(),
        );
    };
    let struct_id = first.0.borrow().struct_id;
    let field_names = {
        let all = ctx.state().field_names.borrow();
        match all.get(struct_id as usize) {
            Some(names) => names.clone(),
            None => {
                return Raisable::Raised(
                    "to_dataframe: no declared fields for this struct".into(),
                );
            }
        }
    };

    let mut columns: Vec<Vec<Val<'gc>>> = vec![Vec::with_capacity(rows.len()); field_names.len()];
    for row in &rows {
        let Some(inst) = row.as_instance() else {
            return Raisable::Raised(
                "to_dataframe: every element must be a struct instance".into(),
            );
        };
        let data = inst.0.borrow();
        if data.struct_id != struct_id {
            return Raisable::Raised(
                "to_dataframe: every element must be the same struct type".into(),
            );
        }
        for (slot, &v) in data.fields.iter().enumerate() {
            columns[slot].push(v);
        }
    }

    let mut pl_columns = Vec::with_capacity(field_names.len());
    for (name, vals) in field_names.iter().zip(columns) {
        match column_from_vals(name, &vals) {
            Ok(c) => pl_columns.push(c),
            Err(e) => return Raisable::Raised(e),
        }
    }
    polars::frame::DataFrame::new_infer_height(pl_columns)
        .map(|d| ctx.new_dataframe(d))
        .into()
}

/// Parses an in-memory CSV string (header row required) into a `DataFrame`, inferring each
/// column's dtype from its values the same way polars' own file-based CSV reader does.
#[native]
fn from_csv<'gc>(ctx: Ctx<'gc>, csv: &str) -> Raisable<vm::DataFrame<'gc>> {
    use polars_io::prelude::{CsvReadOptions, SerReader};
    CsvReadOptions::default()
        .with_has_header(true)
        .into_reader_with_file_handle(std::io::Cursor::new(csv.as_bytes()))
        .finish()
        .map(|d| ctx.new_dataframe(d))
        .into()
}

fn column_from_vals(name: &str, vals: &[Val<'_>]) -> Result<polars::prelude::Column, String> {
    use polars::prelude::Column;
    let name: polars::prelude::PlSmallStr = name.into();
    let mismatch = || format!("to_dataframe: column {name:?} has a mix of incompatible types");
    match vals.first() {
        None => Err(format!("to_dataframe: column {name:?} has no rows")),
        Some(Val::Int(_)) => {
            let v: Vec<i64> = vals
                .iter()
                .map(|v| v.as_int().ok_or_else(mismatch))
                .collect::<Result<_, _>>()?;
            Ok(Column::new(name, v))
        }
        Some(Val::Float(_)) => {
            let v: Vec<f64> = vals
                .iter()
                .map(|v| v.as_float().ok_or_else(mismatch))
                .collect::<Result<_, _>>()?;
            Ok(Column::new(name, v))
        }
        Some(Val::Bool(_)) => {
            let v: Vec<bool> = vals
                .iter()
                .map(|v| v.as_bool().ok_or_else(mismatch))
                .collect::<Result<_, _>>()?;
            Ok(Column::new(name, v))
        }
        Some(Val::Str(_)) => {
            let v: Vec<String> = vals
                .iter()
                .map(|v| v.as_str().map(|s| s.as_str().to_string()).ok_or_else(mismatch))
                .collect::<Result<_, _>>()?;
            Ok(Column::new(name, v))
        }
        Some(other) => Err(format!(
            "to_dataframe: column {name:?} has an unsupported field type: {other:?}"
        )),
    }
}

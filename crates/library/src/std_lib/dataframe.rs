//! `std::polars` -- a thin, fluent-method-chain surface over a real `polars::frame::DataFrame`.
//! `DataFrame`/`PlExpr` are genuine `Val` variants (see `vm::val::DataFrame`/`vm::val::PlExpr`),
//! not `#[mimas] struct`s decomposed into `Val::Instance` fields -- `vm`'s own operator dispatch
//! (`bin()` in `crates/vm/src/val.rs`) already knows how to build a `polars::prelude::Expr` tree
//! out of `col("age") > 30 & col("city") == "SF"`, so this module only needs to wire up the
//! DataFrame-shaped verbs (`filter`/`select`/`sort`/`group_by`+`agg`/`join`/`pivot`) plus the two
//! ways to get a `DataFrame` in the first place: `to_dataframe` (array-of-structs -> columns) and
//! `from_csv` (an in-memory CSV string -> columns, via polars' own reader).

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
    // the `table { … }` literal lowers to these (see `Parser::table_literal`); prelude-level so
    // the desugaring needs no `use`
    api.add(__table_new);
    api.add(__table_col);
    // the `query { … }` block lowers to these (see `Parser::query_literal`)
    api.add(__q_col);
    api.add(__q_n);
    api.add(__q_named);
    api.add(__q_apply);
    api.add(__q_when);
    api.add(__q_filter);
    api.add(__q_mutate);
    api.add(__q_select);
    api.add(__q_sort);
    api.add(__q_take);
    api.add(__q_group);
    api.add(__q_agg);
    api.add(__q_rename);
    api.add(__q_distinct);
    api.add(__q_join);
    api.add(__q_splice);
    {
        let mut m = api.module("std::polars");
        m.add(col);
        m.add(lit);
        m.add(when);
        m.add(n);
        m.add(to_dataframe);
        m.add(from_csv);
        // the verbs are also free functions so `df |> filter(..)` pipes
        // resolve -- `|>` desugars `x |> f(a)` to `f(x, a)`, and methods alone
        // would leave `f` unbound.
        m.add(filter);
        m.add(select);
        m.add(select_names);
        m.add(sort);
        m.add(arrange);
        m.add(mutate);
        m.add(distinct);
        m.add(drop_nulls);
        m.add(head);
        m.add(tail);
        m.add(slice);
        m.add(rename);
        m.add(col_names);
        m.add(pull);
        m.add(pull_as);
        m.add(row);
        m.add(rows);
        m.add(schema);
        m.add(group_by);
        m.add(agg);
        m.add(join);
        m.add(pivot);
        m.add(unpivot);
    }
    api.add_method(filter);
    api.add_method(select);
    api.add_method(select_names);
    api.add_method(sort);
    api.add_method(arrange);
    api.add_method(mutate);
    api.add_method(distinct);
    api.add_method(drop_nulls);
    api.add_method(head);
    api.add_method(tail);
    api.add_method(slice);
    api.add_method(rename);
    api.add_method(col_names);
    api.add_method(pull);
    api.add_method(pull_as);
    api.add_method(row);
    api.add_method(rows);
    api.add_method(schema);
    api.add_method(group_by);
    api.add_method(agg);
    api.add_method(join);
    api.add_method(pivot);
    // Expr aggregation/naming methods -- `col("age").mean().alias("avg_age")`, used inside
    // `agg([..])`. Like `col()`, none of these can fail (they build a plan, they don't run one).
    api.add_method(sum);
    api.add_method(mean);
    api.add_method(median);
    api.add_method(min);
    api.add_method(max);
    api.add_method(count);
    api.add_method(n_unique);
    api.add_method(first);
    api.add_method(last);
    api.add_method(alias);
    api.add_method(shift);
    api.add_method(diff);
    api.add_method(is_null);
    api.add_method(is_not_null);
    api.add_method(fill_null);
    api.add_method(unpivot);
    api.add_method(str_to_upper);
    api.add_method(str_to_lower);
    api.add_method(str_contains);
    api.add_method(str_starts_with);
    api.add_method(str_ends_with);
    api.add_method(str_len);
    api.add_method(cast);
    api.add_method(is_in);
}

#[native]
fn col<'gc>(ctx: Ctx<'gc>, name: &str) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(polars::prelude::col(name))
}

/// `n()` — the row-count expr for `agg`/`summarise`: `gb.agg([n().alias("n")])`
/// (polars `len()`; dplyr's `n()`).
#[native]
fn n<'gc>(ctx: Ctx<'gc>) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(polars::prelude::len())
}

// every query built here is a small in-memory transform -- `collect_with_engine(InMemory)`
// instead of plain `collect()` sidesteps `Engine::Auto` reaching for the streaming engine, which
// needs a polars feature this workspace doesn't enable (see the comment on the `polars`
// dependency in Cargo.toml).
fn collect_in_memory(
    lazy: polars::prelude::LazyFrame,
) -> polars::prelude::PolarsResult<polars::frame::DataFrame> {
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
fn sort<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    by: Vec<String>,
) -> Raisable<vm::DataFrame<'gc>> {
    let opts = polars::prelude::SortMultipleOptions::new();
    df.0.borrow()
        .0
        .sort(by, opts)
        .map(|d| ctx.new_dataframe(d))
        .into()
}

/// `df.select_names(["name", "age"])` — `select` by bare column names, no `col()`
/// exprs needed (the tidy layer's name-driven `select`).
#[native]
fn select_names<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    names: Vec<String>,
) -> Raisable<vm::DataFrame<'gc>> {
    use polars::prelude::{IntoLazy, col};
    let exprs: Vec<polars::prelude::Expr> = names.iter().map(|n| col(n.as_str())).collect();
    let lazy = df.0.borrow().0.clone().lazy();
    collect_in_memory(lazy.select(exprs))
        .map(|d| ctx.new_dataframe(d))
        .into()
}

/// `df.arrange(["laps", "elapsed"], [true, false])` — multi-key sort with a
/// per-key direction; a single-element `desc` broadcasts to every key.
#[native]
fn arrange<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    by: Vec<String>,
    desc: Vec<bool>,
) -> Raisable<vm::DataFrame<'gc>> {
    if desc.len() != by.len() && desc.len() != 1 {
        return Raisable::Raised(format!(
            "arrange: expected {} or 1 direction flags, got {}",
            by.len(),
            desc.len()
        ));
    }
    let opts = polars::prelude::SortMultipleOptions::new().with_order_descending_multi(desc);
    df.0.borrow()
        .0
        .sort(by, opts)
        .map(|d| ctx.new_dataframe(d))
        .into()
}

/// `df.mutate([col("price") * col("qty") | alias "gross"])` — append or replace
/// columns without dropping the rest (polars `with_columns`, dplyr's `mutate`).
#[native]
fn mutate<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    exprs: Vec<vm::PlExpr<'gc>>,
) -> Raisable<vm::DataFrame<'gc>> {
    use polars::prelude::IntoLazy;
    let exprs: Vec<polars::prelude::Expr> = exprs.into_iter().map(|e| e.0.0.clone()).collect();
    let lazy = df.0.borrow().0.clone().lazy();
    collect_in_memory(lazy.with_columns(exprs))
        .map(|d| ctx.new_dataframe(d))
        .into()
}

/// `df.distinct(["dept"])` — first row per unique key combination; an empty
/// `by` dedups whole rows.
#[native]
fn distinct<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    by: Vec<String>,
) -> Raisable<vm::DataFrame<'gc>> {
    use polars::prelude::UniqueKeepStrategy;
    let subset = if by.is_empty() { None } else { Some(by.as_slice()) };
    df.0.borrow()
        .0
        .unique_stable(subset, UniqueKeepStrategy::First, None)
        .map(|d| ctx.new_dataframe(d))
        .into()
}

/// `df.drop_nulls(["age"])` — drop rows with nulls in the named columns; an
/// empty `by` drops rows with a null anywhere.
#[native]
fn drop_nulls<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    by: Vec<String>,
) -> Raisable<vm::DataFrame<'gc>> {
    let subset = if by.is_empty() { None } else { Some(by.as_slice()) };
    df.0.borrow()
        .0
        .drop_nulls(subset)
        .map(|d| ctx.new_dataframe(d))
        .into()
}

#[native]
fn head<'gc>(ctx: Ctx<'gc>, df: vm::DataFrame<'gc>, n: i64) -> vm::DataFrame<'gc> {
    ctx.new_dataframe(df.0.borrow().0.head(Some(n.max(0) as usize)))
}

#[native]
fn tail<'gc>(ctx: Ctx<'gc>, df: vm::DataFrame<'gc>, n: i64) -> vm::DataFrame<'gc> {
    ctx.new_dataframe(df.0.borrow().0.tail(Some(n.max(0) as usize)))
}

#[native]
fn slice<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    offset: i64,
    len: i64,
) -> vm::DataFrame<'gc> {
    ctx.new_dataframe(df.0.borrow().0.slice(offset, len.max(0) as usize))
}

/// `df.rename("old", "new")` — rename one column, non-destructively.
#[native]
fn rename<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    from: &str,
    to: &str,
) -> Raisable<vm::DataFrame<'gc>> {
    let mut out = df.0.borrow().0.clone();
    out.rename(from, to.into())
        .map(|d| ctx.new_dataframe(d.clone()))
        .into()
}

/// `df.col_names()` — the column names, in order, as a mimas `[str]` (for
/// inspection and for driving table renderers/query UIs).
#[native]
fn col_names<'gc>(ctx: Ctx<'gc>, df: vm::DataFrame<'gc>) -> vm::Array<'gc> {
    let names: Vec<Val<'gc>> = {
        let d = df.0.borrow();
        d.0.get_column_names()
            .iter()
            .map(|n| Val::Str(ctx.intern(n.as_str())))
            .collect()
    };
    ctx.new_array(names)
}

/// `df.pull("name")` — one column as a plain mimas array (ints, floats, bools
/// and strings come through natively; anything else raises).
#[native]
fn pull<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    name: &str,
) -> Raisable<vm::Array<'gc>> {
    let col = {
        let d = df.0.borrow();
        match d.0.column(name) {
            Ok(c) => c.clone(),
            Err(e) => return Raisable::Raised(format!("pull: {e}")),
        }
    };
    let s = col.as_materialized_series();
    let mut out = Vec::with_capacity(s.len());
    for v in s.iter() {
        match cell_val(ctx, v, "pull", name) {
            Ok(v) => out.push(v),
            Err(e) => return Raisable::Raised(e),
        }
    }
    Raisable::Ok(ctx.new_array(out))
}

/// AnyValue -> Val, the cell conversion `pull`/`row`/`rows` share. `what` names the
/// verb the user typed so error text points at the right call.
fn cell_val<'gc>(
    ctx: Ctx<'gc>,
    v: polars::prelude::AnyValue,
    what: &str,
    col: &str,
) -> Result<Val<'gc>, String> {
    use polars::prelude::AnyValue;
    Ok(match v {
        AnyValue::Null => Val::Null,
        AnyValue::Boolean(b) => Val::Bool(b),
        AnyValue::Int8(x) => Val::Int(x as i64),
        AnyValue::Int16(x) => Val::Int(x as i64),
        AnyValue::Int32(x) => Val::Int(x as i64),
        AnyValue::Int64(x) => Val::Int(x),
        AnyValue::UInt8(x) => Val::Int(x as i64),
        AnyValue::UInt16(x) => Val::Int(x as i64),
        AnyValue::UInt32(x) => Val::Int(x as i64),
        AnyValue::UInt64(x) => match i64::try_from(x) {
            Ok(x) => Val::Int(x),
            Err(_) => return Err(format!("{what}: {x} overflows int")),
        },
        AnyValue::Float32(x) => Val::Float(x as f64),
        AnyValue::Float64(x) => Val::Float(x),
        AnyValue::String(x) => Val::Str(ctx.intern(x)),
        AnyValue::StringOwned(x) => Val::Str(ctx.intern(x.as_str())),
        other => {
            return Err(format!(
                "{what}: column {col:?} has an unsupported dtype ({other:?})"
            ));
        }
    })
}

/// The struct_id whose declared fields are exactly `cols` (order-free), i.e. the
/// record shape a row of this frame becomes. `field_names`/`struct_names` are
/// indexed by `struct_id` and filled at `load_program` from every adt the program
/// compiled -- including enum-variant layouts, whose fields match too.
fn record_struct(ctx: Ctx, cols: &[String], what: &str) -> Result<u32, String> {
    let field_names = ctx.state().field_names.borrow();
    let mut want: Vec<&str> = cols.iter().map(String::as_str).collect();
    want.sort_unstable();
    let mut found = vec![];
    for (id, fields) in field_names.iter().enumerate() {
        let mut have: Vec<&str> = fields.iter().map(String::as_str).collect();
        have.sort_unstable();
        if have == want {
            found.push(id as u32);
        }
    }
    match found.as_slice() {
        [id] => Ok(*id),
        [] => Err(format!(
            "{what}: no declared struct has fields [{}] -- declare one for the row shape first",
            cols.join(", ")
        )),
        ids => {
            let names = ctx.state().struct_names.borrow();
            let list: Vec<String> = ids
                .iter()
                .map(|&i| {
                    names
                        .get(i as usize)
                        .cloned()
                        .unwrap_or_else(|| format!("#{i}"))
                })
                .collect();
            Err(format!(
                "{what}: columns [{}] match more than one declared struct ({})",
                cols.join(", "),
                list.join(", ")
            ))
        }
    }
}

/// The struct a frame's rows record into, plus one materialized column per field in
/// the struct's declared order, so a row fills its fields by name.
fn record_cols(
    ctx: Ctx,
    df: &polars::frame::DataFrame,
    what: &str,
) -> Result<(u32, Vec<(String, polars::prelude::Series)>), String> {
    let cols: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|n| n.as_str().to_string())
        .collect();
    let sid = record_struct(ctx, &cols, what)?;
    let fields = ctx.state().field_names.borrow()[sid as usize].clone();
    let mut out = Vec::with_capacity(fields.len());
    for name in fields {
        let series = df
            .column(&name)
            .map_err(|e| format!("{what}: {e}"))?
            .as_materialized_series()
            .clone();
        out.push((name, series));
    }
    Ok((sid, out))
}

/// Row `i` of a frame as an instance of `struct_id`, fields filled from `cols`.
fn record_at<'gc>(
    ctx: Ctx<'gc>,
    struct_id: u32,
    cols: &[(String, polars::prelude::Series)],
    i: usize,
    what: &str,
) -> Result<Val<'gc>, String> {
    let mut vals = Vec::with_capacity(cols.len());
    for (name, col) in cols {
        let v = col.get(i).map_err(|e| format!("{what}: {e}"))?;
        vals.push(cell_val(ctx, v, what, name)?);
    }
    Ok(Val::Instance(ctx.new_instance(struct_id, vm::Fields::new(vals))))
}

/// `df.row(2)` -- row `i` as a record: an instance of the declared struct whose
/// fields are exactly this frame's columns, filled by name. Missing cells come
/// through as `null`. Raises when no declared struct matches the columns (or when
/// several do), and on an out-of-bounds index.
#[native]
fn row<'gc>(ctx: Ctx<'gc>, df: vm::DataFrame<'gc>, i: i64) -> Raisable<vm::anon::Anon<'gc, 0>> {
    let d = df.0.borrow();
    (|| {
        let (sid, cols) = record_cols(ctx, &d.0, "row")?;
        let n = d.0.height();
        if i < 0 || i as usize >= n {
            return Err(format!("row: index {i} out of bounds ({n} rows)"));
        }
        record_at(ctx, sid, &cols, i as usize, "row")
    })()
    .map(vm::anon::Anon)
    .into()
}

/// `df.rows()` -- every row as a record, resolved the same way as [`row`].
#[native]
fn rows<'gc>(ctx: Ctx<'gc>, df: vm::DataFrame<'gc>) -> Raisable<vm::anon::ArrayOf<'gc, 0>> {
    let d = df.0.borrow();
    (|| {
        let (sid, cols) = record_cols(ctx, &d.0, "rows")?;
        let mut out = Vec::with_capacity(d.0.height());
        for i in 0..d.0.height() {
            out.push(record_at(ctx, sid, &cols, i, "rows")?);
        }
        Ok::<_, String>(ctx.new_array(out))
    })()
    .map(vm::anon::ArrayOf)
    .into()
}

/// `df.pull_as("kwh", "kWh")` -- a numeric column read as quantities in `unit`. The table holds
/// bare numbers; this says what they measure. Each value is scaled into base units, exactly as
/// `25kW` would be, so the result is a `[float]` the dimension checker sees as a list of
/// quantities (and it checks nothing else about the column). Ints come through as floats.
#[native]
fn pull_as<'gc>(
    _ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    name: &str,
    unit: &str,
) -> Raisable<Vec<f64>> {
    use polars::prelude::AnyValue;
    let Some((_, scale)) = shared::units::parse(unit) else {
        return Raisable::Raised(format!("pull_as: `{unit}` isn't a unit"));
    };
    let col = {
        let d = df.0.borrow();
        match d.0.column(name) {
            Ok(c) => c.clone(),
            Err(e) => return Raisable::Raised(format!("pull_as: {e}")),
        }
    };
    let s = col.as_materialized_series();
    let mut out = Vec::with_capacity(s.len());
    for v in s.iter() {
        let x = match v {
            AnyValue::Int8(x) => x as f64,
            AnyValue::Int16(x) => x as f64,
            AnyValue::Int32(x) => x as f64,
            AnyValue::Int64(x) => x as f64,
            AnyValue::UInt8(x) => x as f64,
            AnyValue::UInt16(x) => x as f64,
            AnyValue::UInt32(x) => x as f64,
            AnyValue::UInt64(x) => x as f64,
            AnyValue::Float32(x) => x as f64,
            AnyValue::Float64(x) => x,
            other => {
                return Raisable::Raised(format!(
                    "pull_as: column {name:?} isn't numeric ({other:?})"
                ));
            }
        };
        out.push(x * scale);
    }
    Raisable::Ok(out)
}

/// `df.schema("zone:int pay:int fare:float kwh:kWh")` -- declare the column types you
/// expect. Unknown columns and cells that can't convert raise; entries that already match
/// are free. A unit (`kwh:kWh`) means "floats in that dimension" -- the runtime checks
/// numeric, and the checker tracks the unit for you from then on. More than validation:
/// the solver reads the same spec, so `df.pull("fare")` after this is a plain `[float]`
/// with no annotation.
#[native]
fn schema<'gc>(ctx: Ctx<'gc>, df: vm::DataFrame<'gc>, spec: &str) -> Raisable<vm::DataFrame<'gc>> {
    use polars::prelude::{DataType, IntoLazy, col};
    let entries = match shared::schema::parse_schema(spec) {
        Ok(entries) => entries,
        Err(e) => return Raisable::Raised(format!("schema: {e}")),
    };
    let frame = df.0.borrow().0.clone();
    let mut casts = vec![];
    for (name, ty) in &entries {
        let want = match ty {
            shared::schema::SchemaTy::Int => DataType::Int64,
            shared::schema::SchemaTy::Float | shared::schema::SchemaTy::Unit(_) => {
                DataType::Float64
            }
            shared::schema::SchemaTy::Str => DataType::String,
            shared::schema::SchemaTy::Bool => DataType::Boolean,
        };
        match frame.column(name) {
            Ok(existing) if existing.dtype() == &want => {}
            Ok(_) => casts.push(col(name).cast(want)),
            Err(_) => {
                let have = frame
                    .get_column_names()
                    .iter()
                    .map(|n| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Raisable::Raised(format!(
                    "schema: no column {name:?} (this dataframe's columns: {have})"
                ));
            }
        }
    }
    if casts.is_empty() {
        return Raisable::Ok(df);
    }
    collect_in_memory(frame.lazy().with_columns(casts))
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
    collect_in_memory(joined)
        .map(|d| ctx.new_dataframe(d))
        .into()
}

/// `df.pivot(["quarter"], ["region"], ["revenue"], "sum")` -- wide-format table: one row per
/// distinct `index` combination, one column per distinct `on` value, cells filled by aggregating
/// matching `values` with `agg` ("sum"/"mean"/"median"/"min"/"max"/"count"/"n_unique"). `agg`
/// takes a function name rather than a `PlExpr` (unlike `group_by`'s `agg`) because pivot's
/// aggregate expression isn't allowed to reference a column by name at all -- it runs against an
/// anonymous per-group "element" placeholder polars builds internally (`values` already says
/// which column). `on`'s distinct values must be known up front to name the output columns
/// (pivot's own requirement, same as Python polars' `DataFrame.pivot`), so this computes them
/// eagerly (`select(on).unique()`) before building the pivot itself.
#[native]
fn pivot<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    on: Vec<String>,
    index: Vec<String>,
    values: Vec<String>,
    agg: &str,
) -> Raisable<vm::DataFrame<'gc>> {
    use polars::frame::PivotColumnNaming;
    use polars::prelude::{IntoLazy, UniqueKeepStrategy, cols, element};
    let agg_expr = match agg {
        "sum" => element().sum(),
        "mean" => element().mean(),
        "median" => element().median(),
        "min" => element().min(),
        "max" => element().max(),
        "count" => element().count(),
        "n_unique" => element().n_unique(),
        other => {
            return Raisable::Raised(format!(
                "pivot: unknown aggregate function {other:?} (expected \"sum\", \"mean\", \"median\", \"min\", \"max\", \"count\", or \"n_unique\")"
            ));
        }
    };
    let on_selector = cols(on);
    let base = df.0.borrow().0.clone();
    // `unique` (unlike `unique_stable`) doesn't maintain row order, which would make the pivot's
    // output *column* order nondeterministic between runs -- `on_columns`' row order is what
    // decides it.
    let on_columns = match collect_in_memory(
        base.clone()
            .lazy()
            .select([on_selector.clone().into()])
            .unique_stable(None, UniqueKeepStrategy::First),
    ) {
        Ok(d) => std::sync::Arc::new(d),
        Err(e) => return Raisable::Raised(e.to_string()),
    };
    let pivoted = base.lazy().pivot(
        on_selector,
        on_columns,
        cols(index),
        cols(values),
        agg_expr,
        false,
        "_".into(),
        PivotColumnNaming::default(),
    );
    collect_in_memory(pivoted)
        .map(|d| ctx.new_dataframe(d))
        .into()
}

macro_rules! pl_expr_reducer {
    ($($name:ident),+ $(,)?) => {$(
        #[native]
        fn $name<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>) -> vm::PlExpr<'gc> {
            ctx.new_plexpr(e.0.0.clone().$name())
        }
    )+};
}
pl_expr_reducer!(sum, mean, median, min, max, count, n_unique, first, last);

#[native]
fn alias<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>, name: &str) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(e.0.0.clone().alias(name))
}

/// `col("t").shift(1)` — the value `n` rows above (a lag); negative `n`
/// reaches the rows below (a lead). The shifted-in cells are null.
/// Sort first: "the row above" is only meaningful in row order.
#[native]
fn shift<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>, n: i64) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(e.0.0.clone().shift(polars::prelude::lit(n)))
}

/// `col("t").diff(1)` — each cell minus the cell `n` rows above; the
/// first `n` cells are null (polars' `NullBehavior::Ignore`).
#[native]
fn diff<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>, n: i64) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(
        e.0.0.clone().diff(polars::prelude::lit(n), polars::series::ops::NullBehavior::Ignore),
    )
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
                return Raisable::Raised("to_dataframe: no declared fields for this struct".into());
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
                .map(|v| {
                    v.as_str()
                        .map(|s| s.as_str().to_string())
                        .ok_or_else(mismatch)
                })
                .collect::<Result<_, _>>()?;
            Ok(Column::new(name, v))
        }
        Some(other) => Err(format!(
            "to_dataframe: column {name:?} has an unsupported field type: {other:?}"
        )),
    }
}

/// `lit(3)` / `lit("x")` -- a constant as an expression, for the places a scalar isn't
/// auto-promoted (`when(..).then` arms, `fill_null`).
#[native]
fn lit<'gc>(ctx: Ctx<'gc>, v: Val<'gc>) -> Result<vm::PlExpr<'gc>, vm::RtErr> {
    use polars::prelude::lit;
    let e = match v {
        Val::Int(i) => lit(i),
        Val::Float(f) => lit(f),
        Val::Bool(b) => lit(b),
        Val::Str(s) => lit(s.as_str()),
        _ => return Err(vm::RtErr::Custom("lit: expected an int, float, bool or str".into())),
    };
    Ok(ctx.new_plexpr(e))
}

/// `when(cond, then, otherwise)` -- polars' conditional column: `then` where `cond` holds,
/// `otherwise` elsewhere (dplyr's `if_else`). Arms are expressions; wrap constants in `lit`.
#[native]
fn when<'gc>(
    ctx: Ctx<'gc>,
    cond: vm::PlExpr<'gc>,
    then: vm::PlExpr<'gc>,
    otherwise: vm::PlExpr<'gc>,
) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(
        polars::prelude::when(cond.0.0.clone())
            .then(then.0.0.clone())
            .otherwise(otherwise.0.0.clone()),
    )
}

#[native]
fn is_null<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(e.0.0.clone().is_null())
}

#[native]
fn is_not_null<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(e.0.0.clone().is_not_null())
}

/// `col("discount").fill_null(lit("none"))` -- replace missing cells.
#[native]
fn fill_null<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>, with: vm::PlExpr<'gc>) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(e.0.0.clone().fill_null(with.0.0.clone()))
}

/// `df.unpivot(["region"], ["q1", "q2"])` -- wide to tall: the `index` columns stay, every `on`
/// column becomes rows of `variable` (its name) and `value` (its cell). The inverse of `pivot`.
#[native]
fn unpivot<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    index: Vec<String>,
    on: Vec<String>,
) -> Raisable<vm::DataFrame<'gc>> {
    use polars::prelude::{IntoLazy, UnpivotArgsDSL, cols};
    let args = UnpivotArgsDSL {
        on: Some(cols(on)),
        index: cols(index),
        variable_name: Some("variable".into()),
        value_name: Some("value".into()),
    };
    let lf = df.0.borrow().0.clone().lazy().unpivot(args);
    collect_in_memory(lf).map(|d| ctx.new_dataframe(d)).into()
}

// text operations on a string column, as expressions (`col("email").str_ends_with(".org")`)
#[native]
fn str_to_upper<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(e.0.0.clone().str().to_uppercase())
}

#[native]
fn str_to_lower<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(e.0.0.clone().str().to_lowercase())
}

/// does the cell contain this literal text (not a pattern)?
#[native]
fn str_contains<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>, text: &str) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(e.0.0.clone().str().contains_literal(polars::prelude::lit(text)))
}

#[native]
fn str_starts_with<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>, text: &str) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(e.0.0.clone().str().starts_with(polars::prelude::lit(text)))
}

#[native]
fn str_ends_with<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>, text: &str) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(e.0.0.clone().str().ends_with(polars::prelude::lit(text)))
}

/// number of characters in each cell
#[native]
fn str_len<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(e.0.0.clone().str().len_chars())
}

/// `col("numtix").cast("int")` -- convert a column's type: "int", "float", "str" or "bool".
/// A cell that can't convert becomes null (strict conversion is a later refinement).
#[native]
fn cast<'gc>(ctx: Ctx<'gc>, e: vm::PlExpr<'gc>, to: &str) -> Result<vm::PlExpr<'gc>, vm::RtErr> {
    use polars::prelude::DataType;
    let dt = match to {
        "int" => DataType::Int64,
        "float" => DataType::Float64,
        "str" => DataType::String,
        "bool" => DataType::Boolean,
        other => {
            return Err(vm::RtErr::Custom(format!(
                "cast: unknown type {other:?} (expected \"int\", \"float\", \"str\" or \"bool\")"
            )));
        }
    };
    Ok(ctx.new_plexpr(e.0.0.clone().cast(dt)))
}

/// `col("discount").is_in(["birthday", "student"])` -- is the cell one of these values?
#[native]
fn is_in<'gc>(
    ctx: Ctx<'gc>,
    e: vm::PlExpr<'gc>,
    values: Vec<Val<'gc>>,
) -> Result<vm::PlExpr<'gc>, vm::RtErr> {
    use polars::prelude::lit;
    let mut out = lit(false);
    for v in values {
        let one = match v {
            Val::Int(i) => lit(i),
            Val::Float(f) => lit(f),
            Val::Bool(b) => lit(b),
            Val::Str(s) => lit(s.as_str()),
            _ => return Err(vm::RtErr::Custom("is_in: expected ints, floats, bools or strs".into())),
        };
        out = out.or(e.0.0.clone().eq(one));
    }
    Ok(ctx.new_plexpr(out))
}

/// `table { … }`'s starting point: a table with no columns.
#[native]
fn __table_new<'gc>(ctx: Ctx<'gc>) -> vm::DataFrame<'gc> {
    ctx.new_dataframe(polars::frame::DataFrame::empty())
}

/// Adds one column (its dtype inferred from the values) to a table under construction. A
/// problem (ragged columns, mixed types) is a runtime error, so `table { … }` is a plain
/// `DataFrame`, not a result.
#[native]
fn __table_col<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    name: &str,
    values: Vec<Val<'gc>>,
) -> Result<vm::DataFrame<'gc>, vm::RtErr> {
    let column = column_from_vals(name, &values)
        .map_err(|e| vm::RtErr::Custom(e.replacen("to_dataframe", "table", 1)))?;
    let mut columns = df.0.borrow().0.columns().to_vec();
    columns.push(column);
    let out = polars::frame::DataFrame::new_infer_height(columns)
        .map_err(|e| vm::RtErr::Custom(format!("table: {e}")))?;
    Ok(ctx.new_dataframe(out))
}

// ---- the `query { … }` block ---------------------------------------------------------------
// PRQL-flavored: bare names are columns, verbs read top to bottom. Every native here reports a
// problem as a runtime error (not a result), so a whole query is one plain `DataFrame`.

fn q_err(e: impl std::fmt::Display) -> vm::RtErr {
    vm::RtErr::Custom(format!("query: {e}"))
}

/// A scalar or an expression, as an expression (`lit` for scalars).
fn q_expr<'gc>(v: Val<'gc>) -> Result<polars::prelude::Expr, vm::RtErr> {
    use polars::prelude::lit;
    Ok(match v {
        Val::PlExpr(e) => e.0.0.clone(),
        Val::Int(i) => lit(i),
        Val::Float(f) => lit(f),
        Val::Bool(b) => lit(b),
        Val::Str(s) => lit(s.as_str()),
        other => return Err(q_err(format!("expected a column expression or a value, got {other:?}"))),
    })
}

fn q_run<'gc>(
    ctx: Ctx<'gc>,
    lazy: polars::prelude::LazyFrame,
) -> Result<vm::DataFrame<'gc>, vm::RtErr> {
    collect_in_memory(lazy).map(|d| ctx.new_dataframe(d)).map_err(q_err)
}

/// a column, by name
#[native]
fn __q_col<'gc>(ctx: Ctx<'gc>, name: &str) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(polars::prelude::col(name))
}

/// the row count (`count()`)
#[native]
fn __q_n<'gc>(ctx: Ctx<'gc>) -> vm::PlExpr<'gc> {
    ctx.new_plexpr(polars::prelude::len())
}

/// name a computed column (`derive total = price * qty`)
#[native]
fn __q_named<'gc>(ctx: Ctx<'gc>, value: Val<'gc>, name: &str) -> Result<vm::PlExpr<'gc>, vm::RtErr> {
    Ok(ctx.new_plexpr(q_expr(value)?.alias(name)))
}

/// Function-call syntax on columns: `sum(x)`, `to_lower(name)`, `contains(name, "a")`, …
/// `arg` is the extra argument (or null).
#[native]
fn __q_apply<'gc>(
    ctx: Ctx<'gc>,
    value: Val<'gc>,
    func: &str,
    arg: Val<'gc>,
) -> Result<vm::PlExpr<'gc>, vm::RtErr> {
    let e = q_expr(value)?;
    let text = |a: Val<'gc>| match a {
        Val::Str(s) => Ok(s.as_str().to_string()),
        _ => Err(q_err(format!("{func} needs a text argument"))),
    };
    let out = match func {
        "sum" => e.sum(),
        "mean" | "average" => e.mean(),
        "median" => e.median(),
        "min" => e.min(),
        "max" => e.max(),
        "count" => e.count(),
        "n_unique" => e.n_unique(),
        "first" => e.first(),
        "last" => e.last(),
        "is_null" => e.is_null(),
        "is_not_null" => e.is_not_null(),
        "to_upper" => e.str().to_uppercase(),
        "to_lower" => e.str().to_lowercase(),
        "len" => e.str().len_chars(),
        "contains" => e.str().contains_literal(polars::prelude::lit(text(arg)?)),
        "starts_with" => e.str().starts_with(polars::prelude::lit(text(arg)?)),
        "ends_with" => e.str().ends_with(polars::prelude::lit(text(arg)?)),
        "fill_null" => e.fill_null(q_expr(arg)?),
        "is_in" => {
            let Val::Array(items) = arg else {
                return Err(q_err("is_in needs a list of values"));
            };
            let mut out = polars::prelude::lit(false);
            for v in items.0.borrow().iter() {
                let one = match v {
                    Val::Int(i) => polars::prelude::lit(*i),
                    Val::Float(f) => polars::prelude::lit(*f),
                    Val::Bool(b) => polars::prelude::lit(*b),
                    Val::Str(s) => polars::prelude::lit(s.as_str()),
                    _ => return Err(q_err("is_in: expected ints, floats, bools or strs")),
                };
                out = out.or(e.clone().eq(one));
            }
            out
        }
        "cast_int" => e.cast(polars::prelude::DataType::Int64),
        "cast_float" => e.cast(polars::prelude::DataType::Float64),
        "cast_str" => e.cast(polars::prelude::DataType::String),
        // `lag(x)`, `lead(x)`, `difference(x)` — the row `n` above/below
        // or the gap to it (`n` defaults to 1); first/last cells come
        // back null. Order-sensitive: `sort` first.
        "lag" | "lead" | "difference" => {
            let n = match arg {
                Val::Null => 1,
                Val::Int(i) => i,
                _ => return Err(q_err(format!("{func} takes an optional int step"))),
            };
            if func == "lag" {
                e.shift(polars::prelude::lit(n))
            } else if func == "lead" {
                e.shift(polars::prelude::lit(-n))
            } else {
                e.diff(polars::prelude::lit(n), polars::series::ops::NullBehavior::Ignore)
            }
        }
        // `eq(a, b)` compares two columns; for a column against an outside
        // value write `a == $who` (the `$` splice) or `eq(a, $who)`
        "eq" => e.eq(q_expr(arg)?),
        "neq" => e.neq(q_expr(arg)?),
        other => return Err(q_err(format!("unknown column function `{other}`"))),
    };
    Ok(ctx.new_plexpr(out))
}

/// `$expr` outside a `query { }` — `q_lower` unwraps the marker inside one,
/// so reaching this call means the escape ran where it isn't an escape.
#[native]
fn __q_splice<'gc>(_ctx: Ctx<'gc>, _value: Val<'gc>) -> Result<Val<'gc>, vm::RtErr> {
    Err(q_err("`$` names an outside value only inside `query { … }`"))
}

/// `if c { a } else { b }` inside a query
#[native]
fn __q_when<'gc>(
    ctx: Ctx<'gc>,
    cond: Val<'gc>,
    then: Val<'gc>,
    otherwise: Val<'gc>,
) -> Result<vm::PlExpr<'gc>, vm::RtErr> {
    Ok(ctx.new_plexpr(
        polars::prelude::when(q_expr(cond)?)
            .then(q_expr(then)?)
            .otherwise(q_expr(otherwise)?),
    ))
}

#[native]
fn __q_filter<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    pred: Val<'gc>,
) -> Result<vm::DataFrame<'gc>, vm::RtErr> {
    use polars::prelude::IntoLazy;
    let lazy = df.0.borrow().0.clone().lazy().filter(q_expr(pred)?);
    q_run(ctx, lazy)
}

#[native]
fn __q_mutate<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    exprs: Vec<vm::PlExpr<'gc>>,
) -> Result<vm::DataFrame<'gc>, vm::RtErr> {
    use polars::prelude::IntoLazy;
    let exprs: Vec<_> = exprs.into_iter().map(|e| e.0.0.clone()).collect();
    q_run(ctx, df.0.borrow().0.clone().lazy().with_columns(exprs))
}

#[native]
fn __q_select<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    names: Vec<String>,
) -> Result<vm::DataFrame<'gc>, vm::RtErr> {
    use polars::prelude::{IntoLazy, col};
    let exprs: Vec<_> = names.iter().map(|n| col(n.as_str())).collect();
    q_run(ctx, df.0.borrow().0.clone().lazy().select(exprs))
}

#[native]
fn __q_sort<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    names: Vec<String>,
    descending: Vec<bool>,
) -> Result<vm::DataFrame<'gc>, vm::RtErr> {
    let opts = polars::prelude::SortMultipleOptions::new().with_order_descending_multi(descending);
    df.0.borrow()
        .0
        .sort(names, opts)
        .map(|d| ctx.new_dataframe(d))
        .map_err(q_err)
}

#[native]
fn __q_take<'gc>(ctx: Ctx<'gc>, df: vm::DataFrame<'gc>, n: i64) -> vm::DataFrame<'gc> {
    ctx.new_dataframe(df.0.borrow().0.head(Some(n.max(0) as usize)))
}

#[native]
fn __q_group<'gc>(ctx: Ctx<'gc>, df: vm::DataFrame<'gc>, names: Vec<String>) -> vm::GroupBy<'gc> {
    use polars::prelude::{IntoLazy, col};
    let by: Vec<_> = names.iter().map(|n| col(n.as_str())).collect();
    ctx.new_group_by(df.0.borrow().0.clone().lazy().group_by(by))
}

#[native]
fn __q_agg<'gc>(
    ctx: Ctx<'gc>,
    gb: vm::GroupBy<'gc>,
    aggs: Vec<vm::PlExpr<'gc>>,
) -> Result<vm::DataFrame<'gc>, vm::RtErr> {
    let exprs: Vec<_> = aggs.into_iter().map(|e| e.0.0.clone()).collect();
    q_run(ctx, gb.0.0.clone().agg(exprs))
}

#[native]
fn __q_rename<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    from: &str,
    to: &str,
) -> Result<vm::DataFrame<'gc>, vm::RtErr> {
    let mut out = df.0.borrow().0.clone();
    out.rename(from, to.into()).map_err(q_err)?;
    Ok(ctx.new_dataframe(out))
}

#[native]
fn __q_distinct<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    names: Vec<String>,
) -> Result<vm::DataFrame<'gc>, vm::RtErr> {
    use polars::prelude::UniqueKeepStrategy;
    let subset = if names.is_empty() { None } else { Some(names.as_slice()) };
    df.0.borrow()
        .0
        .unique_stable(subset, UniqueKeepStrategy::First, None)
        .map(|d| ctx.new_dataframe(d))
        .map_err(q_err)
}

#[native]
fn __q_join<'gc>(
    ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    other: vm::DataFrame<'gc>,
    on: Vec<String>,
    how: &str,
) -> Result<vm::DataFrame<'gc>, vm::RtErr> {
    use polars::prelude::{IntoLazy, JoinArgs, JoinType, col};
    let join_type = match how {
        "inner" => JoinType::Inner,
        "left" => JoinType::Left,
        "right" => JoinType::Right,
        "full" | "outer" => JoinType::Full,
        other => return Err(q_err(format!("unknown join type {other:?}"))),
    };
    let on: Vec<_> = on.iter().map(|n| col(n.as_str())).collect();
    let left = df.0.borrow().0.clone().lazy();
    let right = other.0.borrow().0.clone().lazy();
    let joined = left.join(right, on.clone(), on, JoinArgs::new(join_type));
    q_run(ctx, joined.map_err(q_err)?)
}

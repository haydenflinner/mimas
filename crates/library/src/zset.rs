//! Z-set algebra over polars frames, the runtime half of `query z { … }`.
//!
//! A Z-set frame is an ordinary frame with an integer weight column, `w`.
//! A row with weight 2 stands for two copies of that row, and a row with
//! weight -1 stands for one copy taken away. A plain frame enters with
//! weight 1 on each row. A frame that arrives with an integer `w` column is
//! a Z-set already, and that column holds its weights. A `w` column of any
//! other type is an error, because it cannot hold weights. The parser
//! lowers each `query z` verb to the `__q_*` native it uses in plain mode,
//! or to one of these, when the verb must see the weights (see
//! `Parser::query_body`).
//!
//! Z-sets form an abelian group under `+`: `-z` undoes a change, and the
//! empty Z-set is zero. With `k * z` for an integer `k` they form a
//! Z-module, so a Z-set is a weighted multiset (a bag) that may also hold
//! negative copies. `added`/`removed` split a Z-set into its positive and
//! negative parts, `size` is the sum of the absolute weights, and
//! `integrate`, `differentiate` and `delay` are DBSP's I, D and z⁻¹ over a
//! list of Z-sets.
//!
//! These are pure frame transforms, so the VM natives
//! (`std_lib::dataframe`) and the generated-Rust runtime (`host/eval`'s
//! `pl.rs`) share one implementation.
//! r[impl mimas.query.zset]

use polars::prelude::*;

/// The weight column's name.
pub const W: &str = "w";

// Every transform here is a small in-memory one. `_with_eager` keeps
// `collect()` off the streaming engine, which this workspace does not
// compile in (see `std_lib::dataframe::collect_in_memory`).
fn run(lazy: LazyFrame) -> PolarsResult<DataFrame> {
    lazy._with_eager(true).collect()
}

// A bare `lit(1i64)` is a dynamic int literal that a frame stores as `i32`;
// weights are always `i64`.
fn one() -> Expr {
    lit(1i64).cast(DataType::Int64)
}

fn weight() -> Expr {
    col(W).cast(DataType::Int64)
}

/// The error for a frame whose `w` column is not whole numbers. The solver
/// rejects the same frame at check time when it knows the column's type.
pub fn not_int_message(ty: &str) -> String {
    format!(
        "query z keeps the weights in a column named `w`, and this table's `w` is {ty}: \
         rename it before the query (`.rename(\"w\", \"width\")!`) or make it whole numbers"
    )
}

/// `df` with a weight column. A frame that has an integer `w` already keeps
/// it as its weights, and any other frame gets weight 1 on each row. A `w`
/// of another type is an error.
/// r[impl mimas.query.zset.weight]
/// r[impl mimas.query.zset.adopt]
/// r[impl mimas.query.zset.reject]
pub fn weigh(df: &DataFrame) -> PolarsResult<DataFrame> {
    if let Ok(c) = df.column(W) {
        let dt = c.dtype();
        if !dt.is_integer() {
            let ty = match dt {
                DataType::String => "str".to_string(),
                DataType::Boolean => "bool".to_string(),
                dt if dt.is_float() => "float".to_string(),
                dt => dt.to_string(),
            };
            return Err(PolarsError::ComputeError(not_int_message(&ty).into()));
        }
        return Ok(df.clone());
    }
    run(df.clone().lazy().with_column(one().alias(W)))
}

/// Identical rows merged into one row whose weight is the sum of theirs.
/// Rows whose weights sum to zero drop out. A row keeps the position where
/// it first appears, so a sorted frame stays sorted. The weight column
/// moves to the end.
/// r[impl mimas.query.zset.consolidate]
pub fn consolidate(df: &DataFrame) -> PolarsResult<DataFrame> {
    let df = weigh(df)?;
    let keys: Vec<Expr> = df
        .get_column_names()
        .into_iter()
        .filter(|n| n.as_str() != W)
        .map(|n| col(n.clone()))
        .collect();
    let summed = if keys.is_empty() {
        // a frame of weights alone: one row that holds their sum
        df.lazy().select([weight().sum()])
    } else {
        df.lazy().group_by_stable(keys).agg([weight().sum()])
    };
    run(summed.filter(col(W).neq(lit(0i64))))
}

/// The `query z` name of the aggregate filter: the parser lowers a
/// non-linear aggregate in `group … { aggregate { … } }` (`min`, `max`,
/// `unweighted_median`, `first`, `last`, `n_unique`) to
/// `__q_apply(x, POSITIVE, null)` under the aggregate, and both runtimes
/// answer it with [`positive`].
pub const POSITIVE: &str = "__z_positive";

/// `e` on the rows of positive weight only. A non-linear aggregate reads
/// its column through this, so a row taken away (weight 0 or less) is not
/// a value. Each row counts once, whatever its weight: `min`, `max`,
/// `first`, `last` and `n_unique` do not change with copies, and
/// `unweighted_median` counts each distinct row once.
/// r[impl mimas.query.zset.aggregate.other]
pub fn positive(e: Expr) -> Expr {
    e.filter(col(W).gt(lit(0i64)))
}

/// The `query z` name of the weighted median: the parser lowers `median(x)`
/// in a `query z` aggregate to `__q_apply(x, MEDIAN, null)`, and both
/// runtimes answer it with [`weighted_median`].
pub const MEDIAN: &str = "__z_median";

/// The median of `e` with each row counted `w` times: the median of the
/// copies the rows stand for. It reads only the rows of positive weight
/// whose value is not null. A negative weight takes a copy away, and a
/// median of copies that are not there has no meaning, so those rows are
/// not values.
///
/// Sort the values, add up the weights in that order, and let `total` be
/// the sum of all of them. The lower middle is the first value where twice
/// the running weight reaches `total`, and the upper middle is the first
/// value where it passes `total`. The median is the mean of the two. When
/// the copies are an odd count, or a value's copies span the middle, the
/// two are the same value. When the copies are an even count and the
/// middle falls between two values, the median is halfway between them,
/// as the plain `median` does for an even count of rows. So weights of 1
/// on every row give the plain `median`. An aggregate over no rows is null.
/// r[impl mimas.query.zset.aggregate.median]
pub fn weighted_median(e: Expr) -> Expr {
    let keep = col(W).gt(lit(0i64)).and(e.clone().is_not_null());
    let x = e.filter(keep.clone());
    let wt = weight().filter(keep);
    // a stable sort, so the weights stay paired with their values
    let order = SortMultipleOptions::default().with_maintain_order(true);
    let xs = x.clone().sort_by([x.clone()], order.clone());
    let twice_run = wt.clone().sort_by([x], order).cum_sum(false) * lit(2i64);
    let total = wt.sum();
    let lower = xs
        .clone()
        .filter(twice_run.clone().gt_eq(total.clone()))
        .first();
    let upper = xs.filter(twice_run.gt(total)).first();
    (lower.cast(DataType::Float64) + upper.cast(DataType::Float64)) / lit(2.0f64)
}

/// The polars join kind a query names.
pub fn join_type(how: &str) -> Option<JoinType> {
    Some(match how {
        "inner" => JoinType::Inner,
        "left" => JoinType::Left,
        "right" => JoinType::Right,
        "full" | "outer" => JoinType::Full,
        _ => return None,
    })
}

/// The join of two Z-sets: each output row weighs the product of the two
/// rows it pairs. A side with no match (a left join's unmatched row) has a
/// null weight, which counts as 1, so the matched side's weight carries
/// through unchanged.
/// r[impl mimas.query.zset.join]
pub fn join(
    left: &DataFrame,
    right: &DataFrame,
    on: &[String],
    how: JoinType,
) -> PolarsResult<DataFrame> {
    let right_w = format!("{W}_right");
    let keys: Vec<Expr> = on.iter().map(|n| col(n.as_str())).collect();
    let joined =
        weigh(left)?
            .lazy()
            .join(weigh(right)?.lazy(), keys.clone(), keys, JoinArgs::new(how))?;
    let product =
        weight().fill_null(one()) * col(right_w.as_str()).cast(DataType::Int64).fill_null(one());
    run(joined.with_column(product.alias(W)))?.drop(&right_w)
}

/// Which row of a key `distinct a b` keeps when several rows share it.
/// `First` and `Last` go by input order. `Any` leaves the choice open, so
/// the run may take the cheapest one, and the rows may come out in any
/// order. The query verb defaults to `First`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keep {
    First,
    Last,
    Any,
}

impl Keep {
    /// The policy a `distinct` line names: `first`, `last` or `any`.
    pub fn parse(word: &str) -> Option<Keep> {
        Some(match word {
            "first" => Keep::First,
            "last" => Keep::Last,
            "any" => Keep::Any,
            _ => return None,
        })
    }

    /// The policy a `distinct` native receives as a string. The parser
    /// passes only the three words, so anything else is a fault.
    pub fn of_arg(word: &str) -> PolarsResult<Keep> {
        Keep::parse(word).ok_or_else(|| {
            PolarsError::ComputeError(
                format!("distinct keeps `first`, `last` or `any` row of a key, not {word:?}")
                    .into(),
            )
        })
    }
}

/// One row for each distinct value of the `names` columns, chosen by
/// `keep`. With no names, every distinct row once, in input order. This is
/// plain `distinct`: it reads no weights.
/// r[impl mimas.query.zset.coalesce]
pub fn unique_by(df: &DataFrame, names: &[String], keep: Keep) -> PolarsResult<DataFrame> {
    let subset = if names.is_empty() { None } else { Some(names) };
    match keep {
        Keep::First => df.unique_stable(subset, UniqueKeepStrategy::First, None),
        Keep::Last => df.unique_stable(subset, UniqueKeepStrategy::Last, None),
        Keep::Any => df.unique_impl(
            false,
            subset.map(|v| v.iter().map(|n| PlSmallStr::from_str(n)).collect()),
            UniqueKeepStrategy::Any,
            None,
        ),
    }
}

/// The set of rows with positive weight, each at weight 1. With `names`,
/// one row for each distinct value of those columns, chosen by `keep`
/// from the consolidated rows of positive weight.
/// r[impl mimas.query.zset.distinct]
/// r[impl mimas.query.zset.coalesce]
pub fn distinct(df: &DataFrame, names: &[String], keep: Keep) -> PolarsResult<DataFrame> {
    let present = added(df)?;
    let present = if names.is_empty() {
        present
    } else {
        unique_by(&present, names, keep)?
    };
    run(present.lazy().with_column(one().alias(W)))
}

/// `z.as_set()`: the method spelling of the `distinct` verb with no keys.
/// The rows of positive weight, each at weight 1.
/// r[impl mimas.query.zset.as_set]
pub fn as_set(df: &DataFrame) -> PolarsResult<DataFrame> {
    distinct(df, &[], Keep::First)
}

/// `-z`: every weight negated, so `z + -z` is empty. It undoes a change.
/// The result is consolidated.
/// r[impl mimas.query.zset.negate]
pub fn negate(df: &DataFrame) -> PolarsResult<DataFrame> {
    scale(df, -1)
}

/// `k * z`: every weight times `k`, the change applied `k` times. `k = 0`
/// gives the empty Z-set, which keeps the columns. The result is
/// consolidated. A weight that overflows an `i64` is an error.
/// r[impl mimas.query.zset.scale]
pub fn scale(df: &DataFrame, k: i64) -> PolarsResult<DataFrame> {
    let z = consolidate(df)?;
    if k == 0 {
        return Ok(z.clear());
    }
    let w = z.column(W)?.as_materialized_series().i64()?.clone();
    let mut out = Vec::with_capacity(w.len());
    for i in 0..w.len() {
        let v = w.get(i).unwrap_or(0);
        out.push(v.checked_mul(k).ok_or_else(|| {
            PolarsError::ComputeError(
                format!("{k} * a Z-set: the weight {v} times {k} is too large for an int").into(),
            )
        })?);
    }
    let mut z = z;
    z.with_column(Column::new(W.into(), out))?;
    Ok(z)
}

/// The error for `k * z` where `k` is not a whole number.
pub fn scale_not_int_message(ty: &str) -> String {
    format!(
        "scaling a Z-set applies a change a whole number of times, and this factor is {ty}: \
         write an int (`2 * z`), or round it first (`x.round() * z`)"
    )
}

/// `z.added()`: the positive part, Z⁺. The consolidated rows of positive
/// weight, at their weights.
/// r[impl mimas.query.zset.parts]
pub fn added(df: &DataFrame) -> PolarsResult<DataFrame> {
    run(consolidate(df)?.lazy().filter(col(W).gt(lit(0i64))))
}

/// `z.removed()`: the negative part, Z⁻, at positive weights. The
/// consolidated rows of negative weight, each weight negated. So
/// `z == z.added() - z.removed()`.
/// r[impl mimas.query.zset.parts]
pub fn removed(df: &DataFrame) -> PolarsResult<DataFrame> {
    let gone = run(consolidate(df)?.lazy().filter(col(W).lt(lit(0i64))))?;
    run(gone.lazy().with_column((lit(0i64) - weight()).alias(W)))
}

/// `z.size()`: ‖Z‖, the sum of the absolute weights of the consolidated
/// rows. It counts the copies a change adds and takes away, so
/// `z.size() == z.added().size() + z.removed().size()`, and a Z-set is
/// empty exactly when its size is 0.
/// r[impl mimas.query.zset.size]
pub fn size(df: &DataFrame) -> PolarsResult<i64> {
    let z = consolidate(df)?;
    let w = z.column(W)?.as_materialized_series().i64()?.clone();
    let mut total: i64 = 0;
    for v in (0..w.len()).filter_map(|i| w.get(i)) {
        total = v
            .checked_abs()
            .and_then(|a| total.checked_add(a))
            .ok_or_else(|| {
                PolarsError::ComputeError("the size of this Z-set is too large for an int".into())
            })?;
    }
    Ok(total)
}

/// `z.unweighted()`: the weights dropped. A plain frame of the consolidated
/// rows of positive weight, each once, with no `w` column.
/// r[impl mimas.query.zset.unweighted]
pub fn unweighted(df: &DataFrame) -> PolarsResult<DataFrame> {
    added(df)?.drop(W)
}

/// `zs.integrate()`: DBSP's I over a stream of Z-sets. Item `i` is the sum
/// of items `0..=i`, so the list keeps its length. Each item is
/// consolidated. All the items need the same column names.
/// r[impl mimas.query.zset.integrate]
pub fn integrate(zs: &[DataFrame]) -> PolarsResult<Vec<DataFrame>> {
    let mut out: Vec<DataFrame> = Vec::with_capacity(zs.len());
    for z in zs {
        let next = match out.last() {
            Some(sum) => add(sum, z)?,
            None => consolidate(z)?,
        };
        out.push(next);
    }
    Ok(out)
}

/// `zs.delay()`: DBSP's z⁻¹ over a stream of Z-sets, the stream one step
/// late. Item 0 is the empty Z-set with item 0's columns, and item `i` is
/// input item `i - 1`, so the list keeps its length and drops its last
/// item. Each item is consolidated. Item by item,
/// `zs.differentiate()` is `zs` minus `zs.delay()` (D = 1 − z⁻¹).
/// r[impl mimas.query.zset.delay]
pub fn delay(zs: &[DataFrame]) -> PolarsResult<Vec<DataFrame>> {
    let Some(first) = zs.first() else {
        return Ok(vec![]);
    };
    let mut out = Vec::with_capacity(zs.len());
    out.push(consolidate(first)?.clear());
    for z in &zs[..zs.len() - 1] {
        out.push(consolidate(z)?);
    }
    Ok(out)
}

/// `zs.differentiate()`: DBSP's D over a stream of Z-sets. Item 0 stays,
/// and item `i` is item `i` minus item `i - 1` (the stream minus its
/// delay, z⁻¹). Each item is consolidated, so
/// `zs.integrate().differentiate()` equals `zs` item by item as Z-sets.
/// r[impl mimas.query.zset.differentiate]
pub fn differentiate(zs: &[DataFrame]) -> PolarsResult<Vec<DataFrame>> {
    let mut out = Vec::with_capacity(zs.len());
    for (i, z) in zs.iter().enumerate() {
        out.push(match i {
            0 => consolidate(z)?,
            _ => sub(z, &zs[i - 1])?,
        });
    }
    Ok(out)
}

/// The error for `+` or `-` between two frames whose column names differ.
/// The solver gives the same error at check time when it knows both schemas.
pub fn columns_differ_message(op: &str, left_only: &[String], right_only: &[String]) -> String {
    let list = |names: &[String]| {
        names
            .iter()
            .map(|n| format!("`{n}`"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut parts = vec![];
    if !left_only.is_empty() {
        parts.push(format!("only the left side has {}", list(left_only)));
    }
    if !right_only.is_empty() {
        parts.push(format!("only the right side has {}", list(right_only)));
    }
    format!(
        "`{op}` on two tables needs the same column names on both sides: {}",
        parts.join(", and ")
    )
}

/// The column names (without `w`) that only one side has, as `(left, right)`.
pub fn column_difference(left: &[String], right: &[String]) -> (Vec<String>, Vec<String>) {
    let only = |a: &[String], b: &[String]| {
        a.iter()
            .filter(|n| n.as_str() != W && !b.contains(n))
            .cloned()
            .collect::<Vec<_>>()
    };
    (only(left, right), only(right, left))
}

/// `a` and `b` stacked into one Z-set and consolidated. With `negate`,
/// `b`'s weights count negated. `b`'s columns are matched to `a`'s by name
/// and cast to `a`'s types, so the column order may differ. `op` names the
/// operator in the error for column names that differ.
fn combine(a: &DataFrame, b: &DataFrame, negate: bool, op: &str) -> PolarsResult<DataFrame> {
    let a = weigh(a)?;
    let b = weigh(b)?;
    let names = |df: &DataFrame| {
        df.get_column_names()
            .into_iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
    };
    let (left_only, right_only) = column_difference(&names(&a), &names(&b));
    if !left_only.is_empty() || !right_only.is_empty() {
        return Err(PolarsError::ComputeError(
            columns_differ_message(op, &left_only, &right_only).into(),
        ));
    }
    let cols: Vec<Expr> = a
        .columns()
        .iter()
        .map(|c| {
            let e = col(c.name().clone()).cast(c.dtype().clone());
            if negate && c.name().as_str() == W {
                (lit(0i64) - e).alias(W)
            } else {
                e
            }
        })
        .collect();
    let b = run(b.lazy().select(cols))?;
    consolidate(&a.vstack(&b)?)
}

/// `a + b`: every row of both, consolidated, so the weights of equal rows
/// add and a row whose weights come to 0 drops out. A frame with no `w`
/// counts each row at weight 1.
/// r[impl mimas.query.zset.add]
pub fn add(a: &DataFrame, b: &DataFrame) -> PolarsResult<DataFrame> {
    combine(a, b, false, "+")
}

/// `a - b`: `a + b` with `b`'s weights negated. It equals `changes(a, b)`.
/// r[impl mimas.query.zset.add]
pub fn sub(a: &DataFrame, b: &DataFrame) -> PolarsResult<DataFrame> {
    combine(a, b, true, "-")
}

/// Whether `df` is a Z-set frame: it has a `w` column of whole numbers.
pub fn is_zset(df: &DataFrame) -> bool {
    df.column(W).is_ok_and(|c| c.dtype().is_integer())
}

/// `a == b` on two Z-set frames: the same Z-set, whatever order the rows
/// come in and however the equal rows are split. It holds exactly when
/// `(a - b).size() == 0`. Frames whose column names differ are not equal.
/// r[impl mimas.query.zset.equal]
pub fn equal(a: &DataFrame, b: &DataFrame) -> bool {
    let names = |df: &DataFrame| {
        let mut n: Vec<String> = df.get_column_names().iter().map(|n| n.to_string()).collect();
        n.sort();
        n
    };
    names(a) == names(b) && sub(a, b).and_then(|d| size(&d)).is_ok_and(|n| n == 0)
}

/// `a ⋈ b`, `a.natural_join(b)`: the inner join of two Z-sets on every
/// column both have except `w`. Each output row weighs the product of the
/// two rows it pairs. The result is consolidated, `w` last. Two tables with
/// no column in common are an error, since a join on nothing pairs every
/// row with every row.
/// r[impl mimas.query.zset.join-op]
pub fn natural_join(a: &DataFrame, b: &DataFrame) -> PolarsResult<DataFrame> {
    let right: Vec<String> = b.get_column_names().iter().map(|n| n.to_string()).collect();
    let on: Vec<String> = a
        .get_column_names()
        .iter()
        .map(|n| n.to_string())
        .filter(|n| n != W && right.contains(n))
        .collect();
    if on.is_empty() {
        return Err(PolarsError::ComputeError(
            "a ⋈ b joins on the columns both tables have, and these two have none in common: \
             rename a column so they share one, or join with `query z { a join b key }`"
                .into(),
        ));
    }
    consolidate(&join(a, b, &on, JoinType::Inner)?)
}

/// What changed from `old` to `new`, as a Z-set: the rows of `new` at their
/// weights and the rows of `old` at negated weights, consolidated. A row in
/// both cancels; a row only in `new` has a positive weight; a row only in
/// `old` has a negative weight. It is `new - old`.
/// r[impl mimas.query.zset.changes]
pub fn changes(new: &DataFrame, old: &DataFrame) -> PolarsResult<DataFrame> {
    combine(new, old, true, "changes")
}

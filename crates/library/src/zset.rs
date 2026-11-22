//! Z-set algebra over polars frames, the runtime half of `query z { … }`.
//!
//! A Z-set frame is an ordinary frame with an integer weight column, `__w`.
//! A row with weight 2 stands for two copies of that row, and a row with
//! weight -1 stands for one copy taken away. A plain frame enters with
//! weight 1 on each row. The parser lowers each `query z` verb to the
//! `__q_*` native it uses in plain mode, or to one of these, when the verb
//! must see the weights (see `Parser::query_body`).
//!
//! These are pure frame transforms, so the VM natives
//! (`std_lib::dataframe`) and the generated-Rust runtime (`host/eval`'s
//! `pl.rs`) share one implementation.
//! r[impl mimas.query.zset]

use polars::prelude::*;

/// The weight column's name.
pub const W: &str = "__w";

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

/// `df` with a weight column. A frame that has `__w` already keeps it, and
/// any other frame gets weight 1 on each row.
/// r[impl mimas.query.zset.weight]
pub fn weigh(df: &DataFrame) -> PolarsResult<DataFrame> {
    if df.get_column_index(W).is_some() {
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

/// One plain row for each unit of positive weight, without the weight
/// column: a row of weight 3 comes out three times, and rows of weight 0 or
/// less do not come out. An ordinary aggregate over the result (`count`,
/// `sum`, `min`, `max`, `mean`) then counts each row as often as it weighs.
/// r[impl mimas.query.zset.aggregate]
pub fn expand(df: &DataFrame) -> PolarsResult<DataFrame> {
    let df = weigh(df)?;
    let w = df
        .column(W)?
        .as_materialized_series()
        .cast(&DataType::Int64)?;
    let mut idx: Vec<IdxSize> = Vec::with_capacity(df.height());
    for (i, w) in w.i64()?.iter().enumerate() {
        for _ in 0..w.unwrap_or(0).max(0) {
            idx.push(i as IdxSize);
        }
    }
    df.drop(W)?.take(&IdxCa::from_vec(PlSmallStr::EMPTY, idx))
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

/// The set of rows with positive weight, each at weight 1. With `names`,
/// one row for each distinct value of those columns, the first one seen.
/// r[impl mimas.query.zset.distinct]
pub fn distinct(df: &DataFrame, names: &[String]) -> PolarsResult<DataFrame> {
    let present = run(consolidate(df)?.lazy().filter(col(W).gt(lit(0i64))))?;
    let present = if names.is_empty() {
        present
    } else {
        present.unique_stable(Some(names), UniqueKeepStrategy::First, None)?
    };
    run(present.lazy().with_column(one().alias(W)))
}

/// What changed from `old` to `new`, as a Z-set: the rows of `new` at their
/// weights and the rows of `old` at negated weights, consolidated. A row in
/// both cancels; a row only in `new` has a positive weight; a row only in
/// `old` has a negative weight. `old`'s columns are matched to `new`'s by
/// name and cast to `new`'s types.
/// r[impl mimas.query.zset.changes]
pub fn changes(new: &DataFrame, old: &DataFrame) -> PolarsResult<DataFrame> {
    let new = weigh(new)?;
    let old = weigh(old)?;
    let cols: Vec<Expr> = new
        .columns()
        .iter()
        .map(|c| {
            let e = col(c.name().clone()).cast(c.dtype().clone());
            if c.name().as_str() == W {
                (lit(0i64) - e).alias(W)
            } else {
                e
            }
        })
        .collect();
    let old = run(old.lazy().select(cols))?;
    consolidate(&new.vstack(&old)?)
}

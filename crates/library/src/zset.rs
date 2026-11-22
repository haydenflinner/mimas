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
/// `median`, `first`, `last`, `n_unique`) to `__q_apply(x, POSITIVE, null)`
/// under the aggregate, and both runtimes answer it with [`positive`].
pub const POSITIVE: &str = "__z_positive";

/// `e` on the rows of positive weight only. A non-linear aggregate reads
/// its column through this, so a row taken away (weight 0 or less) is not
/// a value. Each row counts once, whatever its weight: `min`, `max`,
/// `first`, `last` and `n_unique` do not change with copies, and `median`
/// counts each distinct row once.
/// r[impl mimas.query.zset.aggregate.other]
pub fn positive(e: Expr) -> Expr {
    e.filter(col(W).gt(lit(0i64)))
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

/// What changed from `old` to `new`, as a Z-set: the rows of `new` at their
/// weights and the rows of `old` at negated weights, consolidated. A row in
/// both cancels; a row only in `new` has a positive weight; a row only in
/// `old` has a negative weight. It is `new - old`.
/// r[impl mimas.query.zset.changes]
pub fn changes(new: &DataFrame, old: &DataFrame) -> PolarsResult<DataFrame> {
    combine(new, old, true, "changes")
}

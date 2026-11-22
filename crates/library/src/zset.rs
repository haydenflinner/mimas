//! Z-set algebra over polars frames, the runtime half of `query z { … }`.
//!
//! A frame keeps its weights in a column with a name no page can write,
//! [`W`]. Its type is the frame's kind (`shared::weights`): none for a
//! plain table, `bool` for a set, `int` for a Z-set, `float` for a
//! weighted table. A row with weight 2 stands for two copies of that row,
//! and a row with weight -1 stands for one copy taken away. A page writes
//! `w` for the weights inside `query z`, and the display shows them as `w`.
//!
//! The kinds do not cross on their own. `query z { t }` lifts a plain table
//! to a Z-set ([`weigh`]): each row gets weight 1, or a column of its own
//! named `w` becomes the weights. `weighted()` lifts a Z-set to real
//! weights. `counts()`, `unweighted()`, `as_set()` and `expand()` go down,
//! and each says what it loses. An operation on two kinds that do not meet
//! is an error that names the conversion ([`shared::weights::mix_message`]).
//! A set is presence, so it counts as weight 1 beside any weights.
//!
//! A weighted table is the same algebra over the reals (an R-module), so a
//! weight can be a rank, a share or a probability. The reals are a group
//! under `+` too, so negation, undo and the laws hold, apart from float
//! rounding: `0.1 + 0.2 - 0.3` is about 5.6e-17, not 0, so that row stays.
//! Consolidation drops a row only when its weight is exactly 0. The parser
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
//! r[impl mimas.query.zset.kinds]

use polars::prelude::*;
pub use shared::weights::{Kind, USER_W};
use shared::weights::{compatible, float_scale_message, mix_message, plain_message};

/// The weight column's name (`shared::weights::WEIGHT_COL`).
pub const W: &str = shared::weights::WEIGHT_COL;

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

fn fail<T>(msg: String) -> PolarsResult<T> {
    Err(PolarsError::ComputeError(msg.into()))
}

/// The kind of `df`: what its weight column holds, or `Plain` without one.
/// r[impl mimas.query.zset.kinds]
pub fn kind_of(df: &DataFrame) -> Kind {
    match df.column(W) {
        Ok(c) => {
            let dt = c.dtype();
            shared::weights::kind_of_weight(dt == &DataType::Boolean, dt.is_integer(), dt.is_float())
                .unwrap_or(Kind::Plain)
        }
        Err(_) => Kind::Plain,
    }
}

/// An error unless a `a` and a `b` may meet in `op` without a conversion.
/// r[impl mimas.query.zset.kind-mix]
pub fn check_mix(op: &str, a: &DataFrame, b: &DataFrame) -> PolarsResult<()> {
    let (ka, kb) = (kind_of(a), kind_of(b));
    if compatible(ka, kb) { Ok(()) } else { fail(mix_message(op, ka, kb)) }
}

/// `df`'s weights as numbers: a set's `true` becomes 1, and a Z-set or a
/// weighted table stays as it is. A plain table has no weights, so `what`
/// on it is the plain-table error.
fn weights_of(df: &DataFrame, what: &str) -> PolarsResult<DataFrame> {
    match kind_of(df) {
        Kind::Plain => fail(plain_message(what)),
        Kind::Set => run(df.clone().lazy().with_column(col(W).cast(DataType::Int64))),
        Kind::ZSet | Kind::Weighted => Ok(df.clone()),
    }
}

/// Whether `df`'s weights are real numbers: its weight column holds floats.
/// r[impl mimas.query.zset.real]
pub fn is_real(df: &DataFrame) -> bool {
    df.column(W).is_ok_and(|c| c.dtype().is_float())
}

/// The cells of `s` as `f64`s, `None` for a null.
fn floats(s: &Series) -> PolarsResult<Vec<Option<f64>>> {
    let s = s.cast(&DataType::Float64)?;
    let ca = s.f64()?;
    Ok((0..ca.len()).map(|i| ca.get(i)).collect())
}

/// The type weights take: `i64` for a Z-set, `f64` for a weighted table.
fn weight_type(real: bool) -> DataType {
    if real { DataType::Float64 } else { DataType::Int64 }
}

/// The weight column as `i64`, or as `f64` when `real`.
fn weight_as(real: bool) -> Expr {
    col(W).cast(weight_type(real))
}

/// The zero weight, typed to match.
fn zero(real: bool) -> Expr {
    if real { lit(0.0f64) } else { lit(0i64) }
}

/// A weight, as `size` and `item` give it back: whole for a Z-set, real for
/// a weighted table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Weight {
    Int(i64),
    Float(f64),
}

impl Weight {
    pub fn as_f64(self) -> f64 {
        match self {
            Weight::Int(i) => i as f64,
            Weight::Float(f) => f,
        }
    }
}

/// The error for a plain table whose own `w` column is not numbers, when
/// `query z` lifts it. The solver rejects the same frame at check time when
/// it knows the column's type.
pub fn not_int_message(ty: &str) -> String {
    format!(
        "query z reads a column named `w` as the weights, and this table's `w` is {ty}: \
         rename it before the query (`.rename(\"w\", \"width\")!`) or make it numbers"
    )
}

/// `query z { t }`: `t` lifted to weights. A plain table gets weight 1 on
/// each row, or, when it has a column of its own named `w` of whole or real
/// numbers, that column becomes the weights. A `w` of another type is an
/// error. A set's `true` becomes 1. A Z-set or a weighted table stays as it
/// is. This is the one place a plain table becomes a Z-set.
/// r[impl mimas.query.zset.weight]
/// r[impl mimas.query.zset.adopt]
/// r[impl mimas.query.zset.reject]
/// r[impl mimas.query.zset.real]
/// r[impl mimas.query.zset.lift]
pub fn weigh(df: &DataFrame) -> PolarsResult<DataFrame> {
    match kind_of(df) {
        Kind::ZSet | Kind::Weighted => return Ok(df.clone()),
        Kind::Set => return weights_of(df, "lifting"),
        Kind::Plain => {}
    }
    if let Ok(c) = df.column(USER_W) {
        let dt = c.dtype();
        if !dt.is_integer() && !dt.is_float() {
            let ty = match dt {
                DataType::String => "str".to_string(),
                DataType::Boolean => "bool".to_string(),
                dt => dt.to_string(),
            };
            return fail(not_int_message(&ty));
        }
        let mut out = df.clone();
        out.rename(USER_W, W.into())?;
        return Ok(out);
    }
    run(df.clone().lazy().with_column(one().alias(W)))
}

/// Identical rows merged into one row whose weight is the sum of theirs.
/// Rows whose weights sum to zero drop out; for real weights only an exact
/// 0 does. A row keeps the position where it first appears, so a sorted
/// frame stays sorted. The weight column moves to the end.
/// r[impl mimas.query.zset.consolidate]
pub fn consolidate(df: &DataFrame) -> PolarsResult<DataFrame> {
    let df = weigh(df)?;
    let real = is_real(&df);
    let keys: Vec<Expr> = df
        .get_column_names()
        .into_iter()
        .filter(|n| n.as_str() != W)
        .map(|n| col(n.clone()))
        .collect();
    let summed = if keys.is_empty() {
        // a frame of weights alone: one row that holds their sum
        df.lazy().select([weight_as(real).sum()])
    } else {
        df.lazy().group_by_stable(keys).agg([weight_as(real).sum()])
    };
    run(summed.filter(col(W).neq(zero(real))))
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

/// The median of `e` with each row counted `w` times: the point `m` that
/// makes `Σ w·|x − m|` least. It reads every row whose value is not null.
///
/// When every weight is positive this is the median of the copies the rows
/// stand for. Sort the values, add up the weights in that order, and let
/// `total` be the sum of all of them. The lower middle is the first value
/// where twice the running weight reaches `total`, and the upper middle is
/// the first value where it passes `total`. The median is the mean of the
/// two. When the copies are an odd count, or a value's copies span the
/// middle, the two are the same value. When the copies are an even count
/// and the middle falls between two values, the median is halfway between
/// them, as the plain `median` does for an even count of rows. So weights of
/// 1 on every row give the plain `median`.
///
/// A negative weight takes copies away, and the same sum still has a least
/// point when the weights add up to more than 0. When they add up to less
/// than 0, removals outweigh additions, the sum falls without limit as `m`
/// moves past the values, and the median is an error that says so. When they
/// add up to exactly 0, or when the least sum holds at two places with more
/// between them, there is no single middle, and that is an error too. An
/// aggregate over no rows is null.
/// r[impl mimas.query.zset.aggregate.median]
/// r[impl mimas.query.zset.median.signed]
pub fn weighted_median(e: Expr) -> Expr {
    as_struct(vec![e.alias("x"), col(W).cast(DataType::Float64).alias(W)]).agg_with_fmt_str(
        |c: Column| {
            let s = c.as_materialized_series();
            let parts = s.struct_()?.fields_as_series();
            let pairs: Vec<(f64, f64)> = floats(&parts[0])?
                .into_iter()
                .zip(floats(&parts[1])?)
                .filter_map(|(x, w)| Some((x?, w.unwrap_or(1.0))))
                .collect();
            let m = median_of(pairs)?;
            Ok(Column::new(c.name().clone(), [m]))
        },
        |_, f| Ok(Field::new(f.name().clone(), DataType::Float64)),
        MEDIAN,
    )
}

/// The weighted median of `(value, weight)` pairs, as [`weighted_median`]
/// describes it. `None` when no pair has a weight other than 0.
pub fn median_of(mut pairs: Vec<(f64, f64)>) -> PolarsResult<Option<f64>> {
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    // equal values merge, and a value whose weights cancel is not there
    let mut v: Vec<(f64, f64)> = Vec::with_capacity(pairs.len());
    for (x, w) in pairs {
        match v.last_mut() {
            Some(last) if last.0 == x => last.1 += w,
            _ => v.push((x, w)),
        }
    }
    v.retain(|p| p.1 != 0.0);
    if v.is_empty() {
        return Ok(None);
    }
    let total: f64 = v.iter().map(|p| p.1).sum();
    if v.iter().all(|p| p.1 > 0.0) {
        let mut run = 0.0;
        let (mut lower, mut upper) = (None, None);
        for &(x, w) in &v {
            run += w;
            if lower.is_none() && 2.0 * run >= total {
                lower = Some(x);
            }
            if upper.is_none() && 2.0 * run > total {
                upper = Some(x);
            }
        }
        let lower = lower.unwrap_or(v[v.len() - 1].0);
        let upper = upper.unwrap_or(lower);
        return Ok(Some((lower + upper) / 2.0));
    }
    let err = |msg: String| PolarsError::ComputeError(msg.into());
    if total < 0.0 {
        return Err(err(format!(
            "median: the weights add up to {total}, so removals outweigh additions: \
             Σ w·|x − m| falls without limit as m moves past the values, and no middle exists"
        )));
    }
    // Σ w·|x − m| at each value, from running sums of the weights and of w·x
    let s: f64 = v.iter().map(|p| p.0 * p.1).sum();
    let (mut c, mut p) = (0.0, 0.0);
    let f: Vec<f64> = v
        .iter()
        .map(|&(x, w)| {
            c += w;
            p += w * x;
            x * c - p + (s - p) - x * (total - c)
        })
        .collect();
    let least = f.iter().cloned().fold(f64::INFINITY, f64::min);
    let scale: f64 = v.iter().map(|p| p.1.abs()).sum::<f64>() * (v[v.len() - 1].0 - v[0].0);
    let near = |y: f64| (y - least).abs() <= 1e-12 * scale.max(least.abs()).max(1e-300);
    let at: Vec<usize> = (0..f.len()).filter(|&k| near(f[k])).collect();
    let (lo, hi) = (at[0], at[at.len() - 1]);
    if total == 0.0 && (lo == 0 || hi == v.len() - 1) {
        let edge = if lo == 0 { v[0].0 } else { v[v.len() - 1].0 };
        return Err(err(format!(
            "median: the weights add up to 0, so Σ w·|x − m| is least on a whole half-line \
             from {edge}, and there is no single middle"
        )));
    }
    if hi - lo + 1 != at.len() {
        let gap = (lo..=hi).find(|k| !at.contains(k)).unwrap_or(lo);
        return Err(err(format!(
            "median: Σ w·|x − m| is least at {} and at {}, with more at {} between them: \
             the removals split the middle, and there is no single middle",
            v[lo].0, v[hi].0, v[gap].0
        )));
    }
    Ok(Some((v[lo].0 + v[hi].0) / 2.0))
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
/// through unchanged. The product is real when either side's weights are.
/// r[impl mimas.query.zset.join]
pub fn join(
    left: &DataFrame,
    right: &DataFrame,
    on: &[String],
    how: JoinType,
) -> PolarsResult<DataFrame> {
    join_on(left, right, on, on, how)
}

/// [`join`] on key columns that may have other names on the right:
/// `join links (node == src)` pairs a left row with a right row when the
/// left's `node` equals the right's `src`. The right's key columns drop out,
/// since on every matched row they hold what the left's key columns hold.
/// r[impl mimas.query.zset.join]
/// r[impl mimas.query.join-on]
pub fn join_on(
    left: &DataFrame,
    right: &DataFrame,
    left_on: &[String],
    right_on: &[String],
    how: JoinType,
) -> PolarsResult<DataFrame> {
    let right_w = format!("{W}_right");
    // a `query z` block lifts what it reads: both sides enter with weights
    // (a plain table at weight 1), and a Z-set beside a weighted table joins
    // as reals. The no-mixing rule is for the operators on values outside a
    // query block.
    // r[impl mimas.query.zset.block-lifts]
    let (left, right) = (weigh(left)?, weigh(right)?);
    let real = is_real(&left) || is_real(&right);
    let lk: Vec<Expr> = left_on.iter().map(|n| col(n.as_str())).collect();
    let rk: Vec<Expr> = right_on.iter().map(|n| col(n.as_str())).collect();
    let joined = left.lazy().join(right.lazy(), lk, rk, JoinArgs::new(how))?;
    let one = || lit(1i64).cast(weight_type(real));
    let product = weight_as(real).fill_null(one())
        * col(right_w.as_str()).cast(weight_type(real)).fill_null(one());
    run(joined.with_column(product.alias(W)))?.drop(&right_w)
}

/// The error for a `join` whose two key lists differ in length.
pub fn join_keys_message(left: usize, right: usize) -> String {
    format!(
        "this join names {left} key column(s) on the left and {right} on the right: \
         pair each left key with one right key, `join t (a == x && b == y)`"
    )
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

/// `t.as_set()`: presence, a set. The rows of positive weight of a Z-set or
/// a weighted table, or the distinct rows of a plain table, each once with
/// weight `true`. It drops the counts, so it goes down a kind.
/// r[impl mimas.query.zset.as_set]
pub fn as_set(df: &DataFrame) -> PolarsResult<DataFrame> {
    let rows = match kind_of(df) {
        Kind::Plain => unique_by(df, &[], Keep::First)?,
        Kind::Set => return Ok(df.clone()),
        Kind::ZSet | Kind::Weighted => added(df)?,
    };
    run(rows.lazy().with_column(lit(true).alias(W)))
}

/// A frame of numeric weights back to a set: weight `true` on every row.
fn to_set(df: DataFrame) -> PolarsResult<DataFrame> {
    run(df.lazy().with_column(lit(true).alias(W)))
}

/// `-z`: every weight negated, so `z + -z` is empty. It undoes a change.
/// The result is consolidated.
/// r[impl mimas.query.zset.negate]
pub fn negate(df: &DataFrame) -> PolarsResult<DataFrame> {
    let df = &weights_of(df, "negating")?;
    if is_real(df) {
        return scale_real(df, -1.0);
    }
    scale(df, -1)
}

/// `k * z`: every weight times `k`, the change applied `k` times. `k = 0`
/// gives the empty Z-set, which keeps the columns. The result is
/// consolidated. A weight that overflows an `i64` is an error. A weighted
/// table with real weights scales as reals.
/// r[impl mimas.query.zset.scale]
pub fn scale(df: &DataFrame, k: i64) -> PolarsResult<DataFrame> {
    let df = &weights_of(df, "scaling")?;
    if is_real(df) {
        return scale_real(df, k as f64);
    }
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

/// `k * z` with a real `k` on a weighted table: every weight times `k`. A
/// product that comes to exactly 0 drops its row, so `0.0 * z` is empty
/// with `z`'s columns. A set scales as weight 1, so `0.5 · s` is a weighted
/// table. A Z-set's weights are counts, so a real factor on one is an error
/// that names `k · z.weighted()`.
/// r[impl mimas.query.zset.real]
/// r[impl mimas.query.zset.scale-real]
pub fn scale_real(df: &DataFrame, k: f64) -> PolarsResult<DataFrame> {
    if kind_of(df) == Kind::ZSet {
        return fail(float_scale_message());
    }
    let z = consolidate(&weights_of(df, "scaling")?)?;
    let scaled = z.lazy().with_column((weight_as(true) * lit(k)).alias(W));
    run(scaled.filter(col(W).neq(zero(true))))
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
    run(consolidate(&weights_of(df, "taking the positive part of")?)?.lazy().filter(col(W).gt(lit(0i64))))
}

/// `z.removed()`: the negative part, Z⁻, at positive weights. The
/// consolidated rows of negative weight, each weight negated. So
/// `z == z.added() - z.removed()`.
/// r[impl mimas.query.zset.parts]
pub fn removed(df: &DataFrame) -> PolarsResult<DataFrame> {
    let df = &weights_of(df, "taking the negative part of")?;
    let real = is_real(df);
    let gone = run(consolidate(df)?.lazy().filter(col(W).lt(zero(real))))?;
    run(gone.lazy().with_column((zero(real) - weight_as(real)).alias(W)))
}

/// `z.size()`: ‖Z‖, the sum of the absolute weights of the consolidated
/// rows. It counts the copies a change adds and takes away, so
/// `z.size() == z.added().size() + z.removed().size()`, and a Z-set is
/// empty exactly when its size is 0. A weighted table's size is the real
/// sum, its ℓ¹ norm, which [`norm`] gives.
/// r[impl mimas.query.zset.size]
pub fn size(df: &DataFrame) -> PolarsResult<i64> {
    let df = &weights_of(df, "the size of")?;
    if is_real(df) {
        return Err(PolarsError::ComputeError(
            "this table's weights are real numbers, so its size is a float, not an int".into(),
        ));
    }
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

/// ‖Z‖ for either kind of weight: [`size`] as a whole number for a Z-set,
/// and the sum of the absolute real weights for a weighted table.
/// r[impl mimas.query.zset.size]
/// r[impl mimas.query.zset.real]
pub fn norm(df: &DataFrame) -> PolarsResult<Weight> {
    if !is_real(df) {
        return size(df).map(Weight::Int);
    }
    let z = consolidate(df)?;
    let w = floats(z.column(W)?.as_materialized_series())?;
    // a fold from +0.0: an empty `sum()` of floats is -0.0
    Ok(Weight::Float(w.into_iter().flatten().map(f64::abs).fold(0.0, |t, a| t + a)))
}

/// `t.item()`: the one value of a table with one row and one column. A
/// `w` column is not a value: a Z-set's one row must weigh exactly 1, since
/// a row of weight 2 is two copies and a row of weight 0.5 is half of one.
/// Any other shape is an error that says what the table holds. The value
/// comes back as its column holds it.
/// r[impl mimas.query.item]
pub fn item(df: &DataFrame) -> PolarsResult<AnyValue<'static>> {
    let err = |msg: String| PolarsError::ComputeError(msg.into());
    let names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|n| n.to_string())
        .filter(|n| n != W)
        .collect();
    if names.len() != 1 {
        let list = names.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ");
        return Err(err(format!(
            "item() takes one value out of a table with one column, and this table has {}: {}; \
             `select` the one you want first",
            names.len(),
            if list.is_empty() { "none".to_string() } else { list }
        )));
    }
    if df.height() != 1 {
        return Err(err(format!(
            "item() takes one value out of a table with one row, and this table has {} rows",
            df.height()
        )));
    }
    if let Ok(w) = df.column(W) {
        let w = w.as_materialized_series().cast(&DataType::Float64)?;
        let w = w.f64()?.get(0).unwrap_or(1.0);
        if w != 1.0 {
            return Err(err(format!(
                "item() takes one value, and this table's one row weighs {w}, not 1: \
                 it stands for that many copies of its value"
            )));
        }
    }
    Ok(df.column(&names[0])?.as_materialized_series().get(0)?.into_static())
}

/// The error for `∪` or `∩` on a Z-set with removals in it.
fn bag_negative_message(op: &str) -> String {
    format!(
        "`{op}` is a bag operation, and this Z-set has rows of negative weight: a row taken \
         away has no count to take the larger or smaller of. Use `z⁺` for the rows present, or \
         `+`/`⊎` to apply a change"
    )
}

/// `a ∪ b` (`larger`) or `a ∩ b` (not `larger`): one row for each row of
/// either side, weighing the larger or the smaller of its two weights, a
/// side without the row counting 0. A row whose result is 0 drops out.
fn bag_op(a: &DataFrame, b: &DataFrame, larger: bool, op: &str) -> PolarsResult<DataFrame> {
    check_mix(op, a, b)?;
    // two sets give a set: the larger or smaller of two presences
    let sets = kind_of(a) == Kind::Set && kind_of(b) == Kind::Set;
    let a = consolidate(&weights_of(a, op)?)?;
    let b = consolidate(&weights_of(b, op)?)?;
    for side in [&a, &b] {
        let real = is_real(side);
        let neg = run(side.clone().lazy().filter(col(W).lt(zero(real))))?;
        if neg.height() > 0 {
            return Err(PolarsError::ComputeError(bag_negative_message(op).into()));
        }
    }
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
    let real = is_real(&a) || is_real(&b);
    const SIDE: &str = "__zset_left";
    let keys: Vec<String> = names(&a).into_iter().filter(|n| n != W).collect();
    let cols: Vec<Expr> = a
        .columns()
        .iter()
        .map(|c| {
            if c.name().as_str() == W {
                weight_as(real)
            } else {
                col(c.name().clone()).cast(c.dtype().clone())
            }
        })
        .collect();
    let tag = |df: DataFrame, left: bool| {
        run(df.lazy().select(cols.clone()).with_column(lit(left).alias(SIDE)))
    };
    let both = tag(a, true)?.vstack(&tag(b, false)?)?;
    let part = |left: bool| {
        when(col(SIDE).eq(lit(left)))
            .then(col(W))
            .otherwise(zero(real))
            .sum()
    };
    let (wa, wb) = ("__zset_wa", "__zset_wb");
    let aggs = [part(true).alias(wa), part(false).alias(wb)];
    let grouped = if keys.is_empty() {
        both.lazy().select(aggs)
    } else {
        both.lazy()
            .group_by_stable(keys.iter().map(|k| col(k.as_str())).collect::<Vec<_>>())
            .agg(aggs)
    };
    let pick = if larger {
        when(col(wa).gt_eq(col(wb))).then(col(wa)).otherwise(col(wb))
    } else {
        when(col(wa).lt_eq(col(wb))).then(col(wa)).otherwise(col(wb))
    };
    let out = grouped
        .with_column(pick.alias(W))
        .filter(col(W).neq(zero(real)));
    let out = run(out)?.drop(wa)?.drop(wb)?;
    if sets { to_set(out) } else { Ok(out) }
}

/// `a ∪ b`, `a.union(b)`: the bag union. Each row weighs the larger of its
/// weights in `a` and in `b`, so on two sets (weights 1) it is the set union.
/// It is the multiset union of mathematics; SQL's `UNION` (without `ALL`) is
/// its set case. A row of negative weight on either side is an error.
/// `∪` is not linear, so it has no incremental shortcut: `(a ∪ b)^Δ` needs the
/// states, not only the changes.
/// r[impl mimas.query.zset.bag-ops]
pub fn union(a: &DataFrame, b: &DataFrame) -> PolarsResult<DataFrame> {
    bag_op(a, b, true, "∪")
}

/// `a ∩ b`, `a.intersect(b)`: the bag intersection, SQL's `INTERSECT ALL`.
/// Each row weighs the smaller of its weights in `a` and in `b`, so on two
/// sets it is the set intersection. A row of negative weight on either side
/// is an error. `∩` is not linear either.
/// r[impl mimas.query.zset.bag-ops]
pub fn intersect(a: &DataFrame, b: &DataFrame) -> PolarsResult<DataFrame> {
    bag_op(a, b, false, "∩")
}

/// `a ≈ b` on two weighted tables: the same rows, each weight equal to its
/// partner's within float rounding (the `Approx` rule for `float`: a
/// trillionth of the larger, or 1e-15 near zero). A row only one side has
/// must weigh about 0. Tables whose column names differ are not equal.
/// r[impl mimas.query.zset.approx]
pub fn approx_equal(a: &DataFrame, b: &DataFrame) -> PolarsResult<bool> {
    check_mix("≈", a, b)?;
    let (a, b) = (&weights_of(a, "≈")?, &weights_of(b, "≈")?);
    let names = |df: &DataFrame| {
        let mut n: Vec<String> = df.get_column_names().iter().map(|n| n.to_string()).collect();
        n.retain(|n| n != W);
        n.sort();
        n
    };
    if names(a) != names(b) {
        return Ok(false);
    }
    let a = consolidate(a)?;
    let b = consolidate(b)?;
    let keys: Vec<String> = names(&a);
    let wb = "__zset_wb";
    let right = run(b.lazy().rename([W], [wb], true))?;
    let k: Vec<Expr> = keys.iter().map(|n| col(n.as_str())).collect();
    let mut args = JoinArgs::new(JoinType::Full).with_coalesce(JoinCoalesce::CoalesceColumns);
    args.nulls_equal = true;
    let joined = if keys.is_empty() {
        let mut left = run(a.lazy().select([weight_as(true).sum()]))?;
        let right = run(right.lazy().select([col(wb).cast(DataType::Float64).sum()]))?;
        left.with_column(right.column(wb)?.clone())?;
        left
    } else {
        run(a.lazy().join(right.lazy(), k.clone(), k, args)?)?
    };
    let x = floats(joined.column(W)?.as_materialized_series())?;
    let y = floats(joined.column(wb)?.as_materialized_series())?;
    let close = |p: f64, q: f64| {
        let d = (p - q).abs();
        p == q || d <= 1e-15 || d <= 1e-12 * p.abs().max(q.abs())
    };
    Ok(x.into_iter()
        .zip(y)
        .all(|(p, q)| close(p.unwrap_or(0.0), q.unwrap_or(0.0))))
}

/// `z.unweighted()`: the weights dropped. A plain table of the consolidated
/// rows of positive weight, each once, with no weights. A plain table stays
/// as it is.
/// r[impl mimas.query.zset.unweighted]
pub fn unweighted(df: &DataFrame) -> PolarsResult<DataFrame> {
    if kind_of(df) == Kind::Plain {
        return Ok(df.clone());
    }
    added(df)?.drop(W)
}

/// `z.weighted()`: real weights. A Z-set's counts become floats and a set's
/// presence becomes 1.0, so nothing is lost. A weighted table stays as it
/// is. A plain table has no weights to make real: lift it with `query z`
/// first.
/// r[impl mimas.query.zset.weighted]
pub fn weighted(df: &DataFrame) -> PolarsResult<DataFrame> {
    if kind_of(df) == Kind::Plain {
        return fail(
            "`.weighted()` on a plain table: a plain table has no weights. Lift it with \
             `query z { t }.weighted()`"
                .to_string(),
        );
    }
    let z = consolidate(&weights_of(df, "weighting")?)?;
    run(z.lazy().with_column(weight_as(true)))
}

/// `w.counts()`: a weighted table as a Z-set, each weight made an int. A
/// weight that is not a whole number is an error that names it, since a
/// count cannot hold it. A Z-set stays as it is, and a set counts 1 per row.
/// r[impl mimas.query.zset.counts]
pub fn counts(df: &DataFrame) -> PolarsResult<DataFrame> {
    let z = consolidate(&weights_of(df, "counting")?)?;
    if !is_real(&z) {
        return Ok(z);
    }
    let w = floats(z.column(W)?.as_materialized_series())?;
    let mut out = Vec::with_capacity(w.len());
    for v in w.into_iter().flatten() {
        if v.fract() != 0.0 || !v.is_finite() || v.abs() > 9.0e15 {
            return fail(format!(
                "`.counts()`: a row weighs {v}, which is not a whole number, so it is not a \
                 count. Round the weights first (`query z {{ t derive w = round(w) }}`), or keep \
                 the weighted table"
            ));
        }
        out.push(v as i64);
    }
    let mut z = z;
    z.with_column(Column::new(W.into(), out))?;
    Ok(z)
}

/// `z.expand()`: a plain table with each row repeated as many times as it
/// weighs. A row of negative weight has no copies to repeat, so it is an
/// error. A weighted table must be made counts first. A set gives its rows
/// once each, and a plain table stays as it is.
/// r[impl mimas.query.zset.expand]
pub fn expand(df: &DataFrame) -> PolarsResult<DataFrame> {
    match kind_of(df) {
        Kind::Plain => return Ok(df.clone()),
        Kind::Set => return df.drop(W),
        Kind::Weighted => {
            return fail(
                "`.expand()` repeats each row by its count, and this table's weights are real: \
                 make them counts with `.counts()` first"
                    .to_string(),
            );
        }
        Kind::ZSet => {}
    }
    let z = consolidate(df)?;
    let w = z.column(W)?.as_materialized_series().i64()?.clone();
    let mut idx: Vec<IdxSize> = vec![];
    for i in 0..w.len() {
        let n = w.get(i).unwrap_or(0);
        if n < 0 {
            return fail(format!(
                "`.expand()`: a row weighs {n}, and a row taken away has no copies to repeat. \
                 Take the positive part first (`z⁺.expand()`)"
            ));
        }
        idx.extend(std::iter::repeat_n(i as IdxSize, n as usize));
    }
    let take = IdxCa::from_vec("idx".into(), idx);
    z.drop(W)?.take(&take)
}

/// `z.weights_as(name)`: a plain table of the consolidated rows with the
/// weights in an ordinary column called `name`, last. It is how a page
/// reads the weights as data, such as ranks in a sorted list.
/// r[impl mimas.query.zset.weights-as]
pub fn weights_as(df: &DataFrame, name: &str) -> PolarsResult<DataFrame> {
    if kind_of(df) == Kind::Set {
        let mut out = df.clone();
        out.rename(W, name.into())?;
        return Ok(out);
    }
    let mut z = consolidate(&weights_of(df, "`.weights_as()` on")?)?;
    if z.column(name).is_ok() {
        return fail(format!(
            "`.weights_as(\"{name}\")`: the table has a column named `{name}` already; pick \
             another name"
        ));
    }
    z.rename(W, name.into())?;
    Ok(z)
}

/// `x ∈ z`: whether the row `x` is present, with weight above 0. `x` gives
/// a value for each column of `z` by name. A plain table holds a row when
/// it has it at all. It answers a question about presence, so it needs no
/// conversion on any kind.
/// r[impl mimas.query.zset.member]
pub fn contains(df: &DataFrame, row: &[(String, AnyValue<'static>)]) -> PolarsResult<bool> {
    let names: Vec<String> =
        df.get_column_names().iter().map(|n| n.to_string()).filter(|n| n != W).collect();
    let given: Vec<&String> = row.iter().map(|(n, _)| n).collect();
    if names.len() != row.len() || !names.iter().all(|n| given.contains(&n)) {
        return fail(format!(
            "`x ∈ t` asks whether a row is in the table, so `x` needs a value for each of its \
             columns ({}), and it gives {}",
            names.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", "),
            if given.is_empty() {
                "none".to_string()
            } else {
                given.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")
            }
        ));
    }
    let mut cond: Option<Expr> = None;
    for (n, v) in row {
        let e = col(n.as_str()).eq(lit(Scalar::new(v.dtype(), v.clone())));
        cond = Some(match cond {
            Some(c) => c.and(e),
            None => e,
        });
    }
    let z = match kind_of(df) {
        Kind::Plain => df.clone(),
        Kind::Set => df.clone(),
        _ => added(df)?,
    };
    let hits = match cond {
        Some(c) => run(z.lazy().filter(c))?,
        None => z,
    };
    Ok(hits.height() > 0)
}

/// `x ∈ t` on a table of one column: whether the value `x` is present.
pub fn contains_value(df: &DataFrame, v: AnyValue<'static>) -> PolarsResult<bool> {
    let names: Vec<String> =
        df.get_column_names().iter().map(|n| n.to_string()).filter(|n| n != W).collect();
    let [name] = names.as_slice() else {
        return fail(format!(
            "`x ∈ t` with a single value asks about a table of one column, and this table has \
             {}: give a record with a value for each column",
            names.len()
        ));
    };
    contains(df, &[(name.clone(), v)])
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
            None => consolidate(&weights_of(z, "integrating")?)?,
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
    out.push(consolidate(&weights_of(first, "delaying")?)?.clear());
    for z in &zs[..zs.len() - 1] {
        out.push(consolidate(&weights_of(z, "delaying")?)?);
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
            0 => consolidate(&weights_of(z, "differentiating")?)?,
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
    check_mix(op, a, b)?;
    let a = weights_of(a, op)?;
    let b = weights_of(b, op)?;
    // an int-weighted and a float-weighted table combine as floats
    let real = is_real(&a) || is_real(&b);
    let a = if real && !is_real(&a) {
        run(a.lazy().with_column(weight_as(true)))?
    } else {
        a
    };
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
                (zero(real) - e).alias(W)
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

/// Whether `df` has weights: it is a set, a Z-set or a weighted table.
pub fn is_zset(df: &DataFrame) -> bool {
    kind_of(df) != Kind::Plain
}

/// `a == b` on two frames. Two plain tables give `None`: they compare row
/// by row. Two kinds that do not meet are the mix error. Otherwise the two
/// compare as Z-sets ([`equal`]).
/// r[impl mimas.query.zset.equal]
/// r[impl mimas.query.zset.kind-mix]
pub fn frame_eq(a: &DataFrame, b: &DataFrame) -> Result<Option<bool>, String> {
    let (ka, kb) = (kind_of(a), kind_of(b));
    if ka == Kind::Plain && kb == Kind::Plain {
        return Ok(None);
    }
    if !compatible(ka, kb) {
        return Err(mix_message("==", ka, kb));
    }
    Ok(Some(equal(a, b)))
}

/// `a == b` on two frames with weights: the same Z-set, whatever order the
/// rows come in and however the equal rows are split. It holds exactly when
/// `(a - b).size() == 0`. Frames whose column names differ are not equal.
/// Real weights compare exactly; `≈` ([`approx_equal`]) allows for rounding.
/// r[impl mimas.query.zset.equal]
pub fn equal(a: &DataFrame, b: &DataFrame) -> bool {
    let names = |df: &DataFrame| {
        let mut n: Vec<String> = df.get_column_names().iter().map(|n| n.to_string()).collect();
        n.sort();
        n
    };
    names(a) == names(b) && sub(a, b).is_ok_and(|d| d.height() == 0)
}

/// `a ⋈ b`, `a.natural_join(b)`: the inner join of two Z-sets on every
/// column both have except `w`. Each output row weighs the product of the
/// two rows it pairs. The result is consolidated, `w` last. Two tables with
/// no column in common are an error, since a join on nothing pairs every
/// row with every row.
/// r[impl mimas.query.zset.join-op]
pub fn natural_join(a: &DataFrame, b: &DataFrame) -> PolarsResult<DataFrame> {
    check_mix("⋈", a, b)?;
    let sets = kind_of(a) == Kind::Set && kind_of(b) == Kind::Set;
    let (a, b) = (&weights_of(a, "⋈")?, &weights_of(b, "⋈")?);
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
    let out = consolidate(&join(a, b, &on, JoinType::Inner)?)?;
    if sets { to_set(out) } else { Ok(out) }
}

/// What changed from `old` to `new`, as a Z-set: the rows of `new` at their
/// weights and the rows of `old` at negated weights, consolidated. A row in
/// both cancels; a row only in `new` has a positive weight; a row only in
/// `old` has a negative weight. It is `new - old`.
/// r[impl mimas.query.zset.changes]
pub fn changes(new: &DataFrame, old: &DataFrame) -> PolarsResult<DataFrame> {
    // `changes` names the lift: two plain tables give the Z-set between them
    if kind_of(new) == Kind::Plain && kind_of(old) == Kind::Plain {
        return combine(&weigh(new)?, &weigh(old)?, true, "changes");
    }
    combine(new, old, true, "changes")
}

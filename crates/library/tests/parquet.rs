//! `std::polars`'s parquet I/O -- `from_parquet` and `df.to_parquet` (polars' own
//! `ParquetReader`/`ParquetWriter`). See `crates/library/src/std_lib/parquet.rs`.
//!
//! Like `xlsx.rs`'s tests: paths only exist at test run time (fresh temp files per test),
//! so these are plain `#[test]` fns driving `test_runner::render`/`try_execute` directly.
//! No fixture on disk -- every test writes its parquet with `to_parquet` first, which also
//! exercises the write path on the way to the read path.
#![cfg(feature = "parquet")]

#[macro_use]
mod test_runner;

/// A unique temp path per test (parallel `cargo test` runs share the process's temp dir).
fn tmp(test_name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "mimas-parquet-test-{test_name}-{}.parquet",
        std::process::id()
    ))
}

/// A small frame with the dtypes that prove parquet's schema preservation: ints stay `i64`
/// (xlsx would have widened them to `f64`), floats stay `f64`, strings and bools round-trip
/// natively.
const WRITE: &str = r#"
use std::polars::*;
struct Row { id: int, name: str, price: float, instock: bool }
let df = to_dataframe([
    Row { id = 1, name = "pen", price = 1.5, instock = true },
    Row { id = 2, name = "pad", price = 3.0, instock = false },
    Row { id = 3, name = "cup", price = 8.25, instock = true },
])!;
"#;

#[test]
fn parquet_round_trips_dtypes() {
    let path = tmp("round_trip");
    let preamble = format!(
        "{WRITE} let ok = df.to_parquet({:?})!; let back = from_parquet({:?})!;",
        path.display(),
        path.display()
    );

    pretty_assertions::assert_eq!(test_runner::render(&preamble, "ok"), "true");
    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"back.col_names().join(",")"#),
        r#""id,name,price,instock""#
    );
    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"back.pull("id")!"#),
        "[1, 2, 3]"
    );
    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"back.pull("instock")!"#),
        "[true, false, true]"
    );
    // the dtype row is the readable check: `i64`/`f64`/`bool`/`str` all survive verbatim
    pretty_assertions::assert_eq!(
        test_runner::render_display(&preamble, r#"back.select_names(["id", "price"])!"#),
        "shape: (3, 2)\n┌─────┬───────┐\n│ id  ┆ price │\n│ --- ┆ ---   │\n│ i64 ┆ f64   │\n╞═════╪═══════╡\n│ 1   ┆ 1.5   │\n│ 2   ┆ 3.0   │\n│ 3   ┆ 8.25  │\n└─────┴───────┘"
    );
    // and the result is a full DataFrame -- every verb works on it
    pretty_assertions::assert_eq!(
        test_runner::render(
            &preamble,
            r#"back.filter(col("price") > 2)!.pull("name")!.join(",")"#
        ),
        r#""pad,cup""#
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn to_parquet_pipes_as_a_free_function() {
    let path = tmp("piped");
    // `df |> to_parquet(..)` exercises the free-function registration
    let preamble = format!(
        "{WRITE} let ok = (df |> to_parquet({:?}))!;",
        path.display()
    );
    pretty_assertions::assert_eq!(test_runner::render(&preamble, "ok"), "true");
    std::fs::remove_file(&path).ok();
}

#[test]
fn from_parquet_missing_file_raises() {
    assert!(
        test_runner::try_execute(
            r#"use std::polars::*; let df = from_parquet("/nonexistent/path/nothing.parquet")!;"#
        )
        .is_err(),
        "opening a nonexistent path should raise"
    );
}

#[test]
fn from_parquet_non_parquet_data_raises() {
    let path = tmp("not_parquet");
    std::fs::write(&path, b"this is not parquet data").unwrap();
    assert!(
        test_runner::try_execute(&format!(
            "use std::polars::*; let df = from_parquet({:?})!;",
            path.display()
        ))
        .is_err(),
        "data that isn't parquet should raise"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn to_parquet_unwritable_path_raises() {
    assert!(
        test_runner::try_execute(&format!(
            "{WRITE} let ok = df.to_parquet(\"/nonexistent-dir/out.parquet\")!;"
        ))
        .is_err(),
        "writing somewhere unwritable should raise"
    );
}

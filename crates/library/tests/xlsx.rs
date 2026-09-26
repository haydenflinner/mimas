//! `std::polars`'s xlsx I/O -- `from_xlsx` (calamine) and `df.to_xlsx` (rust_xlsxwriter).
//! See `crates/library/src/std_lib/xlsx.rs`.
//!
//! Like `darkly.rs`'s tests, the paths only exist at test run time (a real fixture on disk
//! plus fresh temp files per test), so these are plain `#[test]` fns driving
//! `test_runner::render`/`render_display`/`try_execute` directly.
#![cfg(feature = "xlsx")]

#[macro_use]
mod test_runner;

/// The Northwind products table: one sheet ("Products"), 77 data rows, six columns
/// (ProductID, ProductName, SupplierID, CategoryID, Unit, Price). From
/// https://github.com/LEARNEREA/Excel_Files (Products.xlsx).
const PRODUCTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/products.xlsx");

/// A unique temp path per test (parallel `cargo test` runs share the process's temp dir).
fn tmp(test_name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "mimas-xlsx-test-{test_name}-{}.xlsx",
        std::process::id()
    ))
}

/// A small workbook exercising the non-text cell kinds `products.xlsx` doesn't have:
/// a bool column, a datetime column, a column mixing numbers and strings, and an
/// empty cell (which reads back as null).
fn write_mixed_fixture(path: &std::path::Path) {
    use rust_xlsxwriter::{ExcelDateTime, Format, Workbook};
    let mut workbook = Workbook::new();
    let ws = workbook.add_worksheet();
    let date = Format::new().set_num_format("yyyy-mm-dd");
    for (c, name) in ["name", "flag", "when", "note"].iter().enumerate() {
        ws.write_string(0, c as u16, *name).unwrap();
    }
    ws.write_string(1, 0, "a").unwrap();
    ws.write_boolean(1, 1, true).unwrap();
    ws.write_datetime_with_format(1, 2, &ExcelDateTime::from_ymd(2024, 3, 15).unwrap(), &date)
        .unwrap();
    ws.write_string(1, 3, "x").unwrap();
    ws.write_string(2, 0, "b").unwrap();
    ws.write_boolean(2, 1, false).unwrap();
    ws.write_datetime_with_format(2, 2, &ExcelDateTime::from_ymd(2024, 3, 16).unwrap(), &date)
        .unwrap();
    ws.write_number(2, 3, 7.0).unwrap();
    ws.write_string(3, 0, "c").unwrap();
    ws.write_boolean(3, 1, true).unwrap();
    ws.write_datetime_with_format(3, 2, &ExcelDateTime::from_ymd(2024, 3, 17).unwrap(), &date)
        .unwrap();
    // note[3] deliberately unwritten -- an Empty cell
    workbook.save(path).unwrap();
}

#[test]
fn from_xlsx_reads_the_products_fixture() {
    let preamble = format!("use std::polars::*; let df = from_xlsx({PRODUCTS:?})!;");

    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"df.col_names().join(",")"#),
        r#""ProductID,ProductName,SupplierID,CategoryID,Unit,Price""#
    );
    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"df.pull("ProductName")!.len()"#),
        "77"
    );
    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"df.pull("ProductName")![0]"#),
        r#""Chais""#
    );
    // ints stay i64; a column mixing whole numbers and decimals widens to f64
    // (asserted through the frame's own Display -- `pull` on a schema-less frame
    // hands back `?mimas<T>` elements, so the dtype row is the readable check)
    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"df.pull("ProductID")![0]"#),
        "1"
    );
    pretty_assertions::assert_eq!(
        test_runner::render_display(&preamble, r#"df.select_names(["ProductID", "Price"])!.head(1)"#),
        "shape: (1, 2)\n┌───────────┬───────┐\n│ ProductID ┆ Price │\n│ ---       ┆ ---   │\n│ i64       ┆ f64   │\n╞═══════════╪═══════╡\n│ 1         ┆ 18.0  │\n└───────────┴───────┘"
    );
    // and the result is a full DataFrame -- every verb works on it
    pretty_assertions::assert_eq!(
        test_runner::render(
            &preamble,
            r#"df.filter(col("Price") > 200)!.pull("ProductName")!.join(",")"#
        ),
        r#""Côte de Blaye""#
    );
}

#[test]
fn from_xlsx_reads_a_named_sheet() {
    let preamble = format!(
        "use std::polars::*; let df = from_xlsx({PRODUCTS:?}, \"Products\")!;"
    );
    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"df.pull("ProductName")!.len()"#),
        "77"
    );
}

#[test]
fn from_xlsx_unknown_sheet_raises() {
    assert!(
        test_runner::try_execute(&format!(
            "use std::polars::*; let df = from_xlsx({PRODUCTS:?}, \"Nope\")!;"
        ))
        .is_err(),
        "a sheet name that isn't in the workbook should raise"
    );
}

#[test]
fn from_xlsx_missing_file_raises() {
    assert!(
        test_runner::try_execute(
            r#"use std::polars::*; let df = from_xlsx("/nonexistent/path/nothing.xlsx")!;"#
        )
        .is_err(),
        "opening a nonexistent path should raise"
    );
}

#[test]
fn from_xlsx_non_xlsx_data_raises() {
    let path = tmp("not_xlsx");
    std::fs::write(&path, b"this is not a zip file").unwrap();
    assert!(
        test_runner::try_execute(&format!(
            "use std::polars::*; let df = from_xlsx({:?})!;",
            path.display()
        ))
        .is_err(),
        "data that isn't a zip/xml container should raise"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn from_xlsx_reads_bools_datetimes_mixed_columns_and_nulls() {
    let path = tmp("mixed");
    write_mixed_fixture(&path);
    let preamble = format!(
        "use std::polars::*; let df = from_xlsx({:?})!;",
        path.display()
    );

    // a bool column stays bool
    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"df.pull("flag")!"#),
        "[true, false, true]"
    );
    // a column mixing a string and a number becomes str cell-by-cell; the unwritten
    // cell comes through as null
    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"df.pull("note")!"#),
        r#"["x", "7", null]"#
    );
    // datetime-formatted cells land in a real datetime[μs] column
    pretty_assertions::assert_eq!(
        test_runner::render_display(&preamble, r#"df.select_names(["when"])!"#),
        "shape: (3, 1)\n┌─────────────────────┐\n│ when                │\n│ ---                 │\n│ datetime[μs]        │\n╞═════════════════════╡\n│ 2024-03-15 00:00:00 │\n│ 2024-03-16 00:00:00 │\n│ 2024-03-17 00:00:00 │\n└─────────────────────┘"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn to_xlsx_round_trips_through_from_xlsx() {
    let path = tmp("round_trip");
    let preamble = format!(
        "use std::polars::*; let df = from_xlsx({PRODUCTS:?})!; let ok = df.to_xlsx({:?})!; let back = from_xlsx({:?})!;",
        path.display(),
        path.display()
    );

    pretty_assertions::assert_eq!(test_runner::render(&preamble, "ok"), "true");
    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"back.col_names().join(",")"#),
        r#""ProductID,ProductName,SupplierID,CategoryID,Unit,Price""#
    );
    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"back.pull("ProductName")!.len()"#),
        "77"
    );
    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"back.pull("ProductName")![0]"#),
        r#""Chais""#
    );
    // values survive the trip (through f64 storage, the only numeric kind xlsx has)
    pretty_assertions::assert_eq!(
        test_runner::render(&preamble, r#"back.pull("SupplierID")![2]"#),
        "1"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn to_xlsx_names_its_sheet_and_pipes() {
    let path = tmp("named_sheet");
    // `df |> to_xlsx(..)` exercises the free-function registration; reading it back
    // by name exercises `set_name`
    let preamble = format!(
        "use std::polars::*; let df = from_xlsx({PRODUCTS:?})!; let ok = (df |> to_xlsx({:?}, \"Inventory\"))!;",
        path.display()
    );
    pretty_assertions::assert_eq!(test_runner::render(&preamble, "ok"), "true");
    pretty_assertions::assert_eq!(
        test_runner::render(
            &preamble,
            &format!(r#"from_xlsx({:?}, "Inventory")!.pull("ProductName")![0]"#, path.display())
        ),
        r#""Chais""#
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn to_xlsx_rejects_an_invalid_sheet_name() {
    let path = tmp("bad_sheet");
    assert!(
        test_runner::try_execute(&format!(
            "use std::polars::*; let df = from_xlsx({PRODUCTS:?})!; let ok = df.to_xlsx({:?}, \"a/very:bad*name?\")!;",
            path.display()
        ))
        .is_err(),
        "xlsx sheet names can't contain / : * ? -- that should raise, not silently sanitize"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn to_xlsx_unwritable_path_raises() {
    assert!(
        test_runner::try_execute(&format!(
            "use std::polars::*; let df = from_xlsx({PRODUCTS:?})!; let ok = df.to_xlsx(\"/nonexistent-dir/out.xlsx\")!;"
        ))
        .is_err(),
        "saving somewhere unwritable should raise"
    );
}

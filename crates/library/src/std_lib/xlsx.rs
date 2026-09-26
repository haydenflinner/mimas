//! `std::polars`'s `.xlsx` file I/O: `from_xlsx(path)` reads a worksheet into a `DataFrame`
//! (via `calamine`, pure-Rust), `df.to_xlsx(path)` exports one (via `rust_xlsxwriter`, a port
//! of the author's own Python `XlsxWriter`). Both live in `std::polars` next to `from_csv` --
//! this module just adds to the same module path (install order doesn't matter; `module()` is
//! a name, not a declaration).
//!
//! Read side: the first row is the header (like `from_csv`); an empty header cell becomes
//! `column_{1-based index}`. Column dtypes are inferred from the cells that actually have
//! one: calamine's xlsx reader hands back every plain number as an `f64` (there's no "the
//! author typed an int" signal in the format), so a column where every numeric cell is a
//! whole, in-range number becomes `i64`; any decimal anywhere widens the column to `f64`.
//! Bools stay `bool`, Excel datetimes become `Datetime(μs)`, and anything else (including
//! *mixed* columns) falls back to `str` text per cell so nothing is silently dropped. Cells
//! with Excel error values (`#N/A` etc.) force their column to `str` and keep the error text.
//! Formulas read as their last cached value -- calamine doesn't evaluate them, and neither
//! do we.
//!
//! Write side: headers plus bare values only, no styling beyond a bold header row (and
//! autofit column widths). Numeric ints round-trip through `f64` because that's all an xlsx
//! cell can store anyway; dates/datetimes and any dtype xlsx doesn't model (structs, lists,
//! categoricals, …) write their `Display` text rather than raising.
//!
//! Round-trip editing of an existing workbook (keeping its formatting/charts) is deliberately
//! out of scope -- that wants `umya-spreadsheet`, the one meaningfully-less-hardened crate in
//! this space; reach for it only if a real need shows up.

use macros::native;
use vm::{Ctx, api::Api, conversion::Raisable};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    let mut m = api.module("std::polars");
    m.add(from_xlsx);
    // also a free function so `df |> to_xlsx("out.xlsx")` pipes resolve, same as the verbs
    m.add(to_xlsx);
    api.add_method(to_xlsx);
}

/// `from_xlsx("products.xlsx")` reads the first sheet; `from_xlsx(path, "Sheet2")` picks one
/// by name. Sheet names, missing files, and non-xlsx data all raise.
#[native]
fn from_xlsx<'gc>(
    ctx: Ctx<'gc>,
    path: &str,
    sheet: Option<&str>,
) -> Raisable<vm::DataFrame<'gc>> {
    read_xlsx(path, sheet)
        .map(|d| ctx.new_dataframe(d))
        .into()
}

fn read_xlsx(path: &str, sheet: Option<&str>) -> Result<polars::frame::DataFrame, String> {
    use calamine::Reader;
    let mut workbook: calamine::Xlsx<_> = calamine::open_workbook(path)
        .map_err(|e| format!("from_xlsx: {e}"))?;
    let names = workbook.sheet_names();
    let sheet = match sheet {
        Some(want) if names.iter().any(|n| n == want) => want.to_string(),
        Some(want) => {
            return Err(format!(
                "from_xlsx: no sheet named {want:?} in {path:?} (sheets: {})",
                names.join(", ")
            ));
        }
        None => match names.first() {
            Some(n) => n.clone(),
            None => return Err(format!("from_xlsx: {path:?} has no sheets")),
        },
    };
    let range = workbook
        .worksheet_range(&sheet)
        .map_err(|e| format!("from_xlsx: sheet {sheet:?}: {e}"))?;
    range_to_dataframe(&range)
}

fn range_to_dataframe(
    range: &calamine::Range<calamine::Data>,
) -> Result<polars::frame::DataFrame, String> {
    use calamine::Data;
    use polars::prelude::PlSmallStr;

    if range.is_empty() {
        return Ok(polars::frame::DataFrame::empty());
    }
    let width = range.width();
    let mut rows = range.rows();
    let header_row = rows.next().unwrap_or(&[]);

    let mut names: Vec<PlSmallStr> = Vec::with_capacity(width);
    {
        let mut seen = std::collections::HashSet::new();
        for (i, cell) in header_row.iter().enumerate() {
            let text = match cell {
                Data::Empty => format!("column_{}", i + 1),
                other => cell_text(other),
            };
            if !seen.insert(text.clone()) {
                return Err(format!(
                    "from_xlsx: duplicate column name {text:?} in the header row"
                ));
            }
            names.push(PlSmallStr::from(text));
        }
    }

    let data: Vec<&[Data]> = rows.collect();
    let mut columns = Vec::with_capacity(width);
    for (c, name) in names.iter().enumerate() {
        let cells: Vec<&Data> = data.iter().map(|row| &row[c]).collect();
        columns.push(build_column(name, &cells));
    }
    polars::frame::DataFrame::new_infer_height(columns).map_err(|e| format!("from_xlsx: {e}"))
}

/// The single dtype a column infers to. Folded over the column's non-empty cells: `Int` +
/// `Float` widen to `Float`, same-kind stays, and *any* other mixture (or `Str` meeting
/// anything) becomes `Str` -- each cell then converts to text, so a "number stored as text"
/// in a numeric column (or vice versa) is visible in the data instead of erroring the whole
/// read.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Int,
    Float,
    Bool,
    Str,
    DateTime,
}

fn cell_kind(d: &calamine::Data) -> Option<Kind> {
    use calamine::Data;
    Some(match d {
        Data::Empty => return None,
        Data::Int(_) => Kind::Int,
        // a whole, finite, i64-range float is indistinguishable in xlsx from an int the
        // author typed -- calling it one keeps id/count columns as `i64` the way
        // fastexcel/polars' excel reader does
        Data::Float(x) if x.is_finite() && x.fract() == 0.0 && *x >= i64::MIN as f64 && *x <= i64::MAX as f64 => {
            Kind::Int
        }
        Data::Float(_) => Kind::Float,
        Data::Bool(_) => Kind::Bool,
        // an Excel *duration* (`[hh]:mm:ss`-formatted) isn't a point in time -- the
        // fraction-of-day serial it stores is the honest value to hand over
        Data::DateTime(dt) if dt.is_duration() => Kind::Float,
        Data::DateTime(_) => Kind::DateTime,
        // iso dates/durations and `#N/A`-style cell errors all read as their text
        Data::String(_) | Data::DateTimeIso(_) | Data::DurationIso(_) | Data::Error(_) => {
            Kind::Str
        }
    })
}

fn merge_kind(a: Option<Kind>, b: Kind) -> Option<Kind> {
    Some(match (a, b) {
        (None, k) => k,
        (Some(Kind::Int), Kind::Float) | (Some(Kind::Float), Kind::Int) => Kind::Float,
        (Some(x), y) if x == y => x,
        _ => Kind::Str,
    })
}

fn build_column(
    name: &polars::prelude::PlSmallStr,
    cells: &[&calamine::Data],
) -> polars::prelude::Column {
    use calamine::Data;
    use polars::prelude::Column;

    let kind = cells
        .iter()
        .copied()
        .filter_map(cell_kind)
        .fold(None, merge_kind);
    match kind {
        None => Column::full_null(name.clone(), cells.len(), &polars::prelude::DataType::Null),
        Some(Kind::Int) => Column::new(
            name.clone(),
            cells
                .iter()
                .copied()
                .map(|d| match d {
                    Data::Int(x) => Some(*x),
                    // whole floats take the `Kind::Int` path above -- truncate toward the
                    // exact value they hold (they're all integral here by construction)
                    Data::Float(x) => Some(*x as i64),
                    _ => None,
                })
                .collect::<Vec<Option<i64>>>(),
        ),
        Some(Kind::Float) => Column::new(
            name.clone(),
            cells
                .iter()
                .copied()
                .map(|d| match d {
                    Data::Int(x) => Some(*x as f64),
                    Data::Float(x) => Some(*x),
                    Data::DateTime(dt) => Some(dt.as_f64()),
                    _ => None,
                })
                .collect::<Vec<Option<f64>>>(),
        ),
        Some(Kind::Bool) => Column::new(
            name.clone(),
            cells
                .iter()
                .copied()
                .map(|d| match d {
                    Data::Bool(b) => Some(*b),
                    _ => None,
                })
                .collect::<Vec<Option<bool>>>(),
        ),
        Some(Kind::Str) => Column::new(
            name.clone(),
            cells
                .iter()
                .copied()
                .map(|d| match d {
                    Data::Empty => None,
                    other => Some(cell_text(other)),
                })
                .collect::<Vec<Option<String>>>(),
        ),
        Some(Kind::DateTime) => {
            let micros = cells
                .iter()
                .copied()
                .map(|d| match d {
                    // out-of-range serials (past year 9999) just read null -- a
                    // datetime cell has no better fallback than its opaque serial
                    Data::DateTime(dt) => dt
                        .as_datetime()
                        .map(|ndt| ndt.and_utc().timestamp_micros()),
                    _ => None,
                })
                .collect::<Vec<Option<i64>>>();
            Column::new(name.clone(), micros)
                .as_materialized_series()
                .clone()
                .into_datetime(polars::prelude::TimeUnit::Microseconds, None)
                .into()
        }
    }
}

/// A cell as the text a mixed/string column shows for it. Numbers render the way Excel does
/// (no trailing `.0` -- `f64::to_string` already gives that).
fn cell_text(d: &calamine::Data) -> String {
    use calamine::Data;
    match d {
        Data::Empty => String::new(),
        Data::Int(x) => x.to_string(),
        Data::Float(x) => x.to_string(),
        Data::Bool(b) => b.to_string(),
        Data::String(s) | Data::DateTimeIso(s) | Data::DurationIso(s) => s.clone(),
        Data::DateTime(dt) => match dt.as_datetime() {
            Some(ndt) => ndt.format("%Y-%m-%d %H:%M:%S").to_string(),
            None => dt.as_f64().to_string(),
        },
        Data::Error(e) => e.to_string(),
    }
}

/// `df.to_xlsx("out.xlsx")` writes the frame to a new workbook (one sheet, named `Sheet1`
/// unless `sheet` says otherwise); `df.to_xlsx(path, "Data")` names it. Returns `true` on
/// success like `std::fs::write`. Sheet-name violations and I/O errors raise.
#[native]
fn to_xlsx<'gc>(
    _ctx: Ctx<'gc>,
    df: vm::DataFrame<'gc>,
    path: &str,
    sheet: Option<&str>,
) -> Raisable<bool> {
    use polars::prelude::AnyValue;
    use rust_xlsxwriter::{Format, Workbook};

    let frame = df.0.borrow();
    let result: Result<bool, rust_xlsxwriter::XlsxError> = (|| {
        let mut workbook = Workbook::new();
        let worksheet = workbook.add_worksheet();
        if let Some(name) = sheet {
            worksheet.set_name(name)?;
        }
        let header = Format::new().set_bold();
        for (c, column) in frame.0.columns().iter().enumerate() {
            let c = c as u16;
            worksheet.write_string_with_format(0, c, column.name().as_str(), &header)?;
            for (r, value) in column.as_materialized_series().iter().enumerate() {
                let r = r as u32 + 1;
                // every xlsx number is an f64 anyway; the other dtypes write their own cell
                // kind, and anything xlsx can't model falls back to display text
                let number = match value {
                    AnyValue::Int8(x) => Some(x as f64),
                    AnyValue::Int16(x) => Some(x as f64),
                    AnyValue::Int32(x) => Some(x as f64),
                    AnyValue::Int64(x) => Some(x as f64),
                    AnyValue::UInt8(x) => Some(x as f64),
                    AnyValue::UInt16(x) => Some(x as f64),
                    AnyValue::UInt32(x) => Some(x as f64),
                    AnyValue::UInt64(x) => Some(x as f64),
                    AnyValue::Float32(x) => Some(x as f64),
                    AnyValue::Float64(x) => Some(x),
                    _ => None,
                };
                match number {
                    Some(x) => worksheet.write_number(r, c, x)?,
                    None => match value {
                        AnyValue::Null => continue,
                        AnyValue::Boolean(b) => worksheet.write_boolean(r, c, b)?,
                        AnyValue::String(s) => worksheet.write_string(r, c, s)?,
                        AnyValue::StringOwned(s) => {
                            worksheet.write_string(r, c, s.as_str())?
                        }
                        other => worksheet.write_string(r, c, other.to_string())?,
                    },
                };
            }
        }
        worksheet.autofit();
        workbook.save(path)?;
        Ok(true)
    })();
    result.into()
}

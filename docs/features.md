# Features

Memorable additions to mimas beyond what the changelog tracks — newest first.

## xlsx spreadsheet I/O — `std::polars`

`from_xlsx(path, sheet?)` reads a worksheet into a `DataFrame` via calamine;
`df.to_xlsx(path, sheet?)` exports one via rust_xlsxwriter (new files only —
no editing existing workbooks). Behind the default-on `xlsx` feature on
`library`/`mimas`, same opt-out story as `dataframe`/`darkly`.

- Type inference on read: whole numbers `i64`, any decimal widens `f64`,
  bools `bool`, Excel datetimes `datetime[μs]`, mixed columns fall back to
  per-cell `str`, Excel errors preserved as text, empty cells null. Formulas
  read cached values, never evaluated.
- `rust_xlsxwriter`'s `polars` feature stays off — it would resolve a second,
  crates.io `polars` that doesn't unify with the workspace's path-dep clone.
- Deferred: in-place workbook editing → `umya-spreadsheet` if the need ever
  shows up. SheetJS and the C `xlsxwriter` bindings were ruled out.

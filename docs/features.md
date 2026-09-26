# Features

Memorable additions to mimas beyond what the changelog tracks — newest first.

## DataFrame rows as records — `df.row(i)` / `df.rows()`

`df.row(i)!` hands back one row as an instance of the declared struct whose
fields are exactly the frame's columns, filled by name; `df.rows()!` gives them
all in order. The point is `cleared_1k(shuttle().row(2)!)` — a real table row
going straight into a function typed on the record.

- The struct resolves by field-name *shape*, not by name: zero matching structs
  and several matching structs both raise (`row: no declared struct has fields
  […]`, `… match more than one declared struct (…)`). Out-of-bounds raises.
  Missing cells materialize as `null`.
- Two paths to the record type: the declared return is `anon` (`Anon`/`ArrayOf`
  slot 0) and unifies wherever the row is used — that's the only path for an
  opaque frame (`from_csv`). For a closed-schema frame (`table {}` literal,
  `.schema("…")`, or a schema-propagating op) the solver's `frame_method_call`
  finds the adt up front via `record_adt_for`, so `df.row(0)!.riders` type-checks
  directly without annotation.
- `to_dataframe` is the exact inverse: array-of-structs in, struct-rows out.
- Both are free fns and `DataFrame` methods in `std::polars`; `row` is also how
  `query`-derived frames read back — any frame whose columns match a declared
  struct's fields decomposes, so `derive`d extra columns need a wider struct or
  a `select` first.

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

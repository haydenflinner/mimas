# Features

Memorable additions to mimas beyond what the changelog tracks — newest first.

## Units across the native boundary — `param_dims` / `return_dim` on native signatures

Natives used to be `Any` at the boundary: `music::play_for(t, n, amp, 0.5s, 1200Hz)`
in the wrong order compiled fine because no argument dimension ever reached the
checker. Now a native parameter declared as a unit-carrying Rust type —
`Secs`, `Hz`, `Beats`, `St`, `Tempo`, `Bits` (thin `f64` newtypes in
`mimas::vm::units`) — lands its `Dim` on `ApiFunction`/`ApiMethod`
(`param_dims`, `return_dim`), parallel to the `Vec<Option<Ty>>` slots that
`MimasType::mimas_ty` already produced. `MimasType` gains `mimas_dim`,
`IntoNativeResult` gains `return_dim`; `Option<T>`/`Raisable<T>` forward the
inner dim.

- The dims pass resolves the call's `DecId` through `node_decs` (all three
  call shapes — ident, `x.m()`, `mod::f()` — record it) into `dec_to_native`,
  then checks each arg positionally against `param_dims` and returns
  `D::Q(return_dim)`. Undeclared slots stay `None` = unchecked, and a native
  declaring *no* dims falls through to `builtin`'s special cases (`pull`,
  `to`, `pow`, …) untouched.
- The check rejects *known* mismatches — a plain `0.4` or `1200Hz` into a
  `secs` slot errors — while `Any` (other natives' returns, dynamic values)
  still flows. So adopting a unit on one slot never breaks callers that
  produce the value dynamically.
- `music::synth`/`adsr` take `Secs`/`Hz`, `play_for`/`play_with` take
  `Hz`/`Secs`, `transpose`/`rate` take `St`, `hz` returns `Hz`. zzfx tables
  stay plain by contract — feed a measured value into a slot list with
  `.to("Hz")`, which is exactly the conversion the checker asks for.
- `Dim`'s arithmetic (`mul`/`div`/`powi`/`sqrt`) is `const` now, so a wrapper
  can spell its dimension as a `const` (`Tempo::DIM = BEAT.div(TIME)`).

## `warn` — a non-fatal, source-located diagnostic channel

`Ctx::warn(msg)` records a line on `Prints` like `print` does, but kinded
`Warn` and carrying the call site's `Location` via `DebugInfo::top_loc()`; a
`warn(...)` builtin gives userland the same channel. Entries dedupe on
`(loc, text)` so a warn inside a per-frame loop doesn't flood. Eval reports
them as `warn: true` lines and the web editor renders them amber with the
span marked, a game session accumulates them across frames and clears on
start/stop/replay.

- This is the livecoding answer to Raisable: a bad note in a pattern warns
  and plays silence; the run continues. Music natives (`midi`, `pattern`,
  `pat`, `mask`, …) warn through it; genuinely unrecoverable failures still
  raise.
- `Prints` entries are `(Location, PrintKind, String)` — every consumer of
  `take()` handles the kind.

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

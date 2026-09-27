# Features

Memorable additions to mimas beyond what the changelog tracks — newest first.

## `|>` — why the pipe operator is worth keeping

The TODO asked the honest question: nobody who doesn't write pipelines uses it, so what
does `|>` offer over `.`? Here is what it actually does, where it earns its keep, and
where `.` is the right tool instead.

**What it lowers to.** `|>` is pure sugar at parse time — there is no `ExprKind` for it.
`x |> f(a, b)` becomes the plain call `f(x, a, b)`; `x |> f` becomes `f(x)`. The piped
value is inserted as the *first argument of a free call*, which is the entire semantics.
It is also the loosest operator in the grammar (`a + b |> f` is `f(a + b)`), and it is
the one infix exempt from the newline cutoff, because a line can't start with `|>` —
that is what makes the vertical, one-verb-per-line style parse:

```mimas
samples
|> arrange(["rider", "clock"], [false, false])!
|> mutate([when(col("rider") == col("rider").shift(1),
               col("clock").diff(1),
               col("clock") - lit(START)).alias("laptime")])!
|> select_names(["rider", "laptime"])!
```

Two subtleties worth knowing: `x |> f(1)!.g(2)` puts `x` into the *first* call of the
postfix chain (`f(x, 1)!.g(2)`, not `(f(1)!.g)(x, 2)`), so `!` unwraps per-step inside a
pipeline. And `x |> n.f(1)` pipes into `n.f(x, 1)` — the callee can be any expression
that calls, not just a bare ident.

**What it buys over `.`.** `.` is member access: `x.f(a)` only resolves when `f` is a
method on `x`'s type (or a field). `|>` doesn't dispatch at all — `x |> f(a)` calls the
*free function* `f(x, a)`. So `|>` is the only way to put a value through functions that
aren't, and can't be, methods:

- functions on types you don't own — you can't `impl` methods onto builtin `[T]`, `str`,
  or a `DataFrame` type declared by a library;
- ordinary `fn`s written without thinking about receivers — `x |> clean(2)` for any
  `fn clean(x, k)`;
- the `std::polars` verbs, which are deliberately registered *both* as methods and as
  module functions (see `dataframe.rs::install`) precisely so `df |> filter(..)!` and
  `df.filter(..)!` both resolve — the pipeline style for data transforms was the original
  motivation (tidy/PRQL, commit `8767a67`).

The secondary benefit is visual: a pipeline lines its steps up vertically and reads as a
recipe, and `|>` being loosest means `expr |> f` grabs the whole expression to its left —
usually what you want.

**When `.` is better.** Almost everywhere else. `.` is how you read fields, index, and
call *methods* — including the mutating ones: `xs.push(0)` takes `&mut self`, and
`xs |> push(0)` desugars to a *call* `push(xs, 0)`, which doesn't resolve because `push`
isn't a free function. `|>` passes the value by first-argument position; it cannot reach
`&mut self` receivers. If a name is only a method, it won't resolve under `|>` — that is
the main footgun, and the reason polars verbs are dual-registered.

**Verdict: keep.** The `.` operator covers methods; `|>` covers the one thing `.` can't —
threading a value through free functions in reading order — and it costs almost nothing:
a desugar in `Parser::binary`/`pipe` (~50 lines), no new AST, no runtime, no type-system
surface. Nothing about it is mandatory; code that never pipelines never sees it. The cost
it does impose is a second spelling for call chains (`x.f()` vs `x |> f()` when `f` is
dual-registered), and the resolution surprise when a method-only name doesn't exist as a
free function. Both are documented above; the upside is that the tidy surface reads like
tidy, which was the point.

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

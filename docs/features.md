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

## Cranelift JIT tier — `mimas-jit` (branch `jit-cranelift`)

Second Futamura projection: `jit::compile(&program)` emits one native function
per bytecode chunk with Cranelift (`FunctionBuilder`, dense op dispatch through
`br_table` on a byte-offset→op-index map) and returns a `Vec<Option<BodyFn>>`
for `Vm::install_bc` — the same slot bcgen bodies use, so interpreter/JIT
bodies interleave freely.

- Scalar int/float registers live in SSA vars ("shadows" with ok-flags) inside
  a body; they flush to the register window before any call that observes it.
  Unsupported ops delegate to `bc::jit::step_at` (= the interpreter's own
  `step_one`), so semantics can't drift — only speed can.
- `vm::bc::jit` is the `extern "C"` helper layer bound via `JITBuilder::symbol`
  (`mj_*` names in `crates/jit/src/lib.rs::SPECS`, `H` indexes it 1:1).
  **ABI gotcha:** `Ctx` is a repr(C) 2-pointer struct; under AAPCS64 a composite
  that doesn't fully fit in remaining arg registers goes *wholly* to the stack,
  while a flat Cranelift signature would split it — so `call_native` (ctx at
  arg slot 7) takes `mc`/`st` as two pointers and rebuilds via
  `Ctx::from_parts`. Any new helper with `Ctx` past arg 6 must do the same.
- Calls: `CallDirect`/dynamic `Call` run inline below `INLINE_CALL_DEPTH` via
  `jit::enter` (=`enter_call`) + a direct/indirect call into the callee body
  (fn-ptr table in JIT module data), `pop_return` on `Flow::Return`; deeper
  frames return `Flow::Call` to the driver as before.
- Parity: `mimas-jit-test` runs each fixture under both lanes and diffs
  `TEST_VALUE` + `keep` log + error — including op-budget *counts*, fuel
  windows, pause/resume (`run_frame`), snapshot/restore, and GC pressure.
- Bench (bc-bench `jit` lane, M-series arm64): fib_iter jit 172ms vs bc 254ms,
  mandelbrot 519 vs 710 — beats pre-tuning bcgen on scalar loops; fib_rec
  2313 vs 1089 and physics 2897 vs 704 lose (call-shim + flush overhead is the
  known gap vs the batched-quota/loop-form bcgen on `bcgen-tuning`).

## bcgen second specialization pass — branch `bcgen-perf`

On top of `bcgen-tuning`'s scalar shadows and batched quota gates:

- **Sparse multi-block loop wrapping** (`emit_body` + `grow_region` in
  `crates/bcgen/src/lib.rs`): a bounded DFS grows a loop region over
  continuation edges — jump targets first, then conditional fallthrough —
  backtracking on dead ends. Regions stay ascending and validate iff every
  interior edge lands on the head (`continue`), the next emitted op, or
  outside the region (`code.ip` + `break`). Handles `while A && B`
  short-circuit re-entry (mandelbrot's inner loop) and physics' 13-block
  loop; `Switch` regions are rejected outright.
- **Bool shadows**: `W::Bool` joins int/float in `analyze` — comparisons,
  `JumpIf` conditions and `Constant::Bool` keep a native `bool` + ok flag
  instead of boxing `Val::Bool` per write. `Unary::Not` emits a direct
  `Val::Bool(b) => !b` arm, generic `unary` otherwise.
- **Typed (Array, Int) indexing**: `GetIndex`/`SetIndex` inline the
  bounds-checked `borrow`/`borrow_mut(&ctx)` path; every other
  collection/index combo keeps the `get_index`/`set_index` helpers, so the
  `IndexOutOfBounds`/`Option`-Null error contract is identical.

Measured (`bc-bench` bc lane vs the pre-pass tree, best-of ≥20): mandelbrot
190→86–104ms (bool shadows alone ≈ −17%, the rest mostly the loop form),
prime_numbers 245→196–210ms (typed indexing ≈ −10%), physics ~unchanged,
fib_iter/fib_rec unchanged. Rejected along the way: contiguous gap-fill of
sparse regions (regressed hot inner loops) and `_vN` write-forwarding locals
(the extra copies cost more than the `rd` slot loads they saved — +6–7%).

## JIT round-2 perf — branch `jit-perf` (PR haydenflinner/mimas#7)

Four layered optimizations over the tier above, all semantics-preserving
(helpers/`estep` still provide the ground truth):

- **Quota batching (M1):** CLIF `bcn`/`bcn0` vars mirror bcgen's
  `settle!`/`gateq!`/`gatep!`: each op decrements a batched counter; pause,
  fuel-out and ops_left exhaustion escape through cold gate trampolines;
  `settle` runs at every observable boundary (calls, helper ops, exits).
- **Bulk shadow flush (M2):** one `mj_flush` FFI call drains a packed
  `FlushEnt` list — replaces per-register `wr_i`/`wr_f` calls.
- **Megacalls (M3):** `mj_call_body`/`mj_call_dyn` fold callee resolution,
  `enter_call_regs` frame push, callee `BodyFn` invocation, and the
  `Flow::Return` pop/truncate/dst writeback into one FFI hop, returning the
  rebuilt caller-window pointer for re-pinning.
- **Direct `Val` access (M4):** `bc::jit::layout()` probes the live layout —
  discriminant position/width, `Int`/`Float`/`Bool`/`Fn` payload offsets,
  `ThreadState`/`Frame`/`Decoder` fields, `Vec` header order — by semantic
  round-trip (mutate bytes of a known `Val`, `ptr::read` it back and compare
  `discriminant()`; invalid patterns decode as "other", never abort).
  Emitted code then does tag compares + payload loads/stores inline —
  `ri`/`rf`/`rval`/`wr_*` FFI calls vanish from hot paths.
  **Gotcha:** `size_of::<Discriminant<Val>>` (8) is the *token* width, not
  the field width — rustc leaves stale bytes in discriminant padding, so
  the probe tries widths smallest-first and emits 1-byte tag compares.
- Result (release): mandelbrot jit 117ms vs bc 199ms (**jit wins x1.7**);
  fib_iter 89 vs 60; prime_numbers 334 vs 255; fib_rec 1006 vs 723 and
  physics 1065 vs 273 — call-heavy still trails; next lever is inlining /
  direct body dispatch to skip the megacall FFI hop.

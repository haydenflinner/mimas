# Features

Memorable additions to mimas beyond what the changelog tracks. Newest first.

## `query z { }`: queries over Z-sets (`library/src/zset.rs`, spec `mimas.query.zset`)

`query z { … }` takes the verbs of `query { … }`, but each frame is a
Z-set: rows carry an integer weight in a `w` column, and identical rows
merge by adding their weights. The parser lowers the block to the same
`__q_*` natives plus five `__z_*` ones, at the verbs that must see the
weights:

- The source table and each joined table go through `__z_in`: weight 1 per
  row unless the frame has an integer `w` already, so a `query z` result
  feeds the next `query z` with its weights, and so does any table with an
  int `w` column. A `w` of another type is an error: the frame-schema pass
  reports it when it knows the column's type (`ZsetWeightNotInt`), and
  `zset::weigh` faults on it at run time. Both messages name the fix,
  `.rename("w", "width")!`.
- `filter`, `derive` and `rename` lower as in plain mode. `select` adds
  `w` to its name list.
- `join` is `__z_join`: the polars join, then `w = w * w_right`, and
  `w_right` goes. A
  null weight (an unmatched side of a left, right or full join) counts as 1.
- `group … { aggregate { … } }` consolidates first (`__z_cons`), and the
  parser rewrites each aggregate to read `w` (`Parser::z_weigh_agg`):
  `count()` is `sum(w)`, `sum(x)` is `sum(x * w)`, `count(x)` sums `w`
  where `x` is not null, and `mean(x)` is `sum(x * w)` over `sum(w)`. A
  negative weight counts against the total, so a delta Z-set aggregates
  to the change. `min`, `max`, `median`, `first`, `last` and `n_unique`
  read their column through `zset::positive`, the rows of positive
  weight (the `__z_positive` apply). Each output row weighs 1.
- `distinct` is `__z_distinct`: consolidate, keep positive rows, weight 1.
- `sort` and `take` see `__z_cons(t)`, the consolidated frame, and so does
  the block's value. Consolidation is a stable group-by over every other
  column, so a row stays where its first copy was and a sorted frame stays
  sorted. `w` ends up as the last column.

`changes(new, old)` in `std::polars` is the delta: `new`'s rows at their
weights, `old`'s at negated weights, consolidated. `z1 + z2` and
`z1 - z2` on two frames are `zset::add` and `zset::sub`, and `changes`
is `sub`. The VM reaches them through `Api::add_frame_bin_op`, a hook in
`instance_bin` for `DataFrame` operands. `pl.rs` implements `Add`/`Sub`
for `DataFrame`. The frame-schema pass types them
(`Solver::frame_evaluation`): the left side's columns and `w` last, and
`ZsetColumnsDiffer` when the two sides' column names differ. The frame-schema pass
knows each `__z_*` native, so `.pull("w")!` checks as `[int]`.

The algebra lives in `library::zset` as plain polars transforms, and
`host/eval`'s `pl.rs` calls the same functions for generated Rust. This is
phase 1: each block recomputes from its inputs. Incremental upkeep across
live passes is not here yet (see the parent repo's
`docs/devlog/zset-query-2026-11-21.md`).

Tests: `library/tests/dataframe.rs` (`zset_*`), one per verb rule plus
`changes`, the int `w` rule (`zset_int_w_column_is_the_weights`,
`zset_float_w_column_*`), the weighted aggregates
(`zset_aggregate_sums_by_weight`, `zset_min_max_*`) and `+`/`-`
(`zset_tables_add_and_subtract`, `zset_add_with_different_columns_*`); `cli/src/fmt.rs` `query_cells_keep_adjacency`.

## Canonical formatter `mimas fmt` (`cli/src/fmt.rs`, spec `mimas.fmt`)

A token-stream respacer, not an AST reprint: comments are tokens, so every
comment survives verbatim and each literal keeps its written form (`1e-6`,
`#1.10`, `1_000`, `25kW`). Newlines are semantic under semicolon insertion,
so the formatter only rewrites whitespace — gaps, indentation, blank runs,
trailing whitespace — plus canonical operator spellings (`and`→`&&`,
`≠`→`!=`, `~=`→`≈`). The semantic adjacencies stay: `x!`/`x?`/`x%` postfix
glue, `check (` vs `check(`, `[0 -1]` vs `[0 - 1]`, glued generics, `px/s`
unit types.

- `mimas fmt <paths>` rewrites `.mim`/`.mimas` in place (dirs walk
  recursively), `--check` lists without writing (exit 1 on diffs, 2 on
  operational errors), no paths or `-` reads stdin to stdout.
- `// fmt: off` … `// fmt: on` emits a region verbatim; `// fmt: skip`
  exempts the next statement or item, `{…}` body included.
- Columnar alignment: consecutive same-shape, same-indent lines pad a
  shared pivot — `=` of `let`/`const`/assignments and `=` record fields,
  values after `:` field colons, `=>` match arms, `->` signature lines,
  comma-separated row cells. Groups break at blank lines, comment-only
  lines, items, and shape changes; a group drops whole if its column
  passes 80 or a member needs more than 32 spaces. Emission is two
  passes — measure columns under canonical single-space gaps, then write
  pads — so `fmt(fmt(x)) == fmt(x)` holds by construction.
- Guard: the output must relex to the same non-comment token stream, or
  the file is left alone. `table {}` rows align to columns with a
  4-space gutter; `query {}` keeps written adjacency; `unit` specs glue
  `m/s` operands; `macro` decls glue `$x:expr` fragments and `),*`
  repeats.

## IR optimizer: inlining and scalar replacement (`compile/src/codegen/opt.rs`)

Two passes run on the IR before `clean`, for every lane (interpreter,
jit, wasmgen) since they rewrite the shared `compile::Ir`. `MIMAS_OPT=0`
disables them; `MIMAS_DUMP_OPT=1` prints the IR after them.

- **Inlining.** A direct call to a small body (≤ 64 insts) is replaced by
  a copy: the call's block binds the callee's params as fresh locals and
  jumps into the copy; every copied `return` stores into one fresh local
  and jumps to a continuation block holding the rest of the caller's
  block, with the call instruction itself turned into a read of that
  local -- so nothing that referenced the call's id changes, and phis
  that named the split block are re-pointed at the continuation.
  Callees are processed bottom-up (post-order over the direct-call
  graph), a caller stops at 4096 insts. Never inlined: the entry body
  (into or out of), closures (captures), anything that reaches itself,
  a body containing `raise` (`Op::Raise` returns from the *frame*, which
  a copy no longer has), and a body that calls a frame-reading native --
  the VM hands the compiler every `std::debug::*` id via
  `Compiler::with_frame_natives`, so `debug::caller_line` keeps reporting
  the truth. Call-edge *tracing* (`Vm::trace_take`) sees no edge for an
  inlined call; `vm/tests/trace.rs` uses a recursive callee for that
  reason.
- **Scalar replacement.** A struct local that is only ever built from a
  `make_instance` and read or written field by field becomes one local
  per field, and the struct is never allocated. The unit is an *object
  class*: a root local whose single store is a `make_instance` nothing
  else uses, plus every local that is a plain copy of a member
  (`set l2: get l1`, which is what inlining leaves for returns and
  params); members share the field locals, which is what aliasing
  means, and the copies are dropped. Soundness rule: every member is
  written exactly once, that write dominates all of its reads (block
  dominators over the body, position within a block), and every read is
  a `get_field`/`set_field` on the object or the copy into another
  member. Params, captures, entry slots, anything stored, compared,
  passed or returned keep their object and identity. Up to four rounds
  per body (a field that held a struct becomes a local holding one).
  The entry body is skipped entirely: its locals are the page's
  top-level bindings, read off the heap by the debugger, `#m` prose and
  `envDraw`.

Measured (`wasmgen/examples/wgbench`, release, interpreter vs the wasm
lane under wasmtime; `float::*` natives bound for real -- a zero stub
made the earlier comparison a lie):

| program | interpreter | wasm lane | wasm arena per run |
| --- | --- | --- | --- |
| closest-point loop, 200k × 4 segments (`V2` math) | 0.956 s → 0.312 s (3.1×) | 56.5 → 29.9 ms (1.9×) | 121 MB → 1.1 MB |
| walk-on-stars solver, 2000 walks (the `wost-kit`) | 0.154 s → 0.085 s (1.8×) | 9.4 → 6.8 ms (1.4×) | 19.9 → 6.3 MB |

Results identical with the passes off and on. The arena column is the
point for the wasm lane: it has no GC, so an allocation-heavy program
runs out of memory before it finishes -- the solver at 20k walks
trapped in `__alloc` before, and runs now.

Tests: `compile/src/tests/opt_tests.rs` (what the passes leave:
inlined arithmetic, no field ops on a scalarized local, params and
escaping or rebound structs untouched, entry body untouched, recursion
not inlined), `vm/tests/opt_semantics.rs` (mutation through an inlined
param, copy-then-rebind, struct in an array, returned structs,
structural equality, a struct per loop iteration, branch-chosen struct,
nested fields). `wgbench` moved here from `host/eval/examples` and binds
the float natives by name (floats cross the import boundary as f64 or
as their bit pattern in a `Word`; result types come from the import's
signature), instantiates fresh per run (the arena persists across calls
of one instance), and reports the arena high-water mark.

**SSA for locals and phis of structs (round 2).** `promote_locals`
turns a local into SSA values: every read becomes the definition that
reaches it, phis go at the iterated dominance frontier of the local's
stores (block dominators, immediate dominators and frontiers computed
per body), and the slot disappears. Before that, `split_critical_edges`
puts a forwarder on every `jump_if_false`/`for_next`/`switch` edge into
a block with two or more predecessors, because codegen hosts phi copies
only before an unconditional `jump` (`clean::thread_jumps` leaves a
forwarder whose target opens with a phi). Promoted: unnamed locals
(what inlining and scalar replacement make) and named locals whose
every store is a struct -- built there, copied from another such local,
or chosen by an `if`. Not promoted: params and captures, entry slots, a
`for_next` loop variable (the op bumps the slot's register in place), a
local with a read no store dominates, and *named scalar locals*: a read
of a slot is already free (codegen aliases it to the slot's register),
SSA would only add a phi copy per merge on top of the store the step
debugger's frame view (`artifacts`) needs. `split_struct_phis` then
dissolves a struct chosen by control flow -- `x = if c { V2 {…} } else
{ V2 {…} }`, a loop-carried `v = V2 { x = v.x * 2.0, … }`: the class is
a phi, its inputs, every phi a member feeds (the struct built before a
loop feeds the loop-head *and* the loop-exit phi), and it qualifies
when every use of every member is a `get_field` on it or membership in
another member phi -- a field read off the phi becomes a phi of the
field reads, a field read off a `make_instance` its operand, and no
object is ever allocated. Codegen now emits a block's phi copies as one
parallel assignment (ordered so no source is overwritten before it is
read, a cycle broken through a scratch register), which a loop that
swaps two values needs once several phis share a block.
`MIMAS_SSA=0` keeps the slot form, as an A/B.

A/B within one run (`wgbench`, release, machine under other builds --
only same-run ratios mean anything): the `V2` loop's `run` body goes
from 138 ops and 51 moves to 114 ops and 27 moves; interpreter 0.918 s
→ 0.868 s, wasm lane unchanged (0.042 s); the solver 0.182 s → 0.174 s.
Results identical.

**Trace edges survive inlining.** An inlined call leaves an
`Inst::TraceEdge { callee }` → `Op::TraceEdge` where the call was: the
hot dispatcher bumps the trace fixture's `caller → callee` counter with
the current frame's chunk, so `Vm::trace_take` -- the sonification
substrate -- shows the same call graph it did before (`vm/tests/trace.rs`
runs its original inlinable program again). bcgen and JIT bodies drop
it (they bypass call tracing already); the wasm lane has no trace.
Per-op hits were never lost: copied instructions keep the callee's
source spans in the loc table.

**Value types, inferred (round 3).** `value_adts` finds the adts no
instruction in the program mutates: no `set_field` on one, no
`set_index` into one (a tuple struct's `t.0 = v`), no native that
mutates its receiver (`mutates_recv`, handed over by the VM as
`Compiler::with_mutating_natives`). Since `==` is structural, an
instance of such an adt has no identity anyone can observe, so copying
one is always sound -- it is a value. Receivers are typed from the IR
(`ty_of`: construction, typed locals and params, a field of a typed
struct, an element of a typed array, an unwrapped option, a phi of
agreeing inputs); a receiver whose type cannot be recovered and could
be an instance fails closed and turns value types off for the program
(`MIMAS_DUMP_OPT=1` says where). What a value type buys:
`forward_value_fields` reads a field off a freshly built value as its
operand whatever else the struct is used for (`let d = a.sub(b);
keep(d); d.x`), and `split_struct_phis` lets a value-type class
*escape* -- be returned, passed, stored, compared, join a phi outside
the class -- by building a fresh struct from the field values right
before the use (for a phi outside the class, at the end of that edge's
block). A class may also have *leaves*: a value-type struct from
outside the class (a param, a call result, an element) joining its
phi, whose fields are read once right where it is defined -- so a
loop seeded from a param runs unboxed too. The walk's `x` is such a
class: unboxed through the loop, one `V2` built for `g(x)` and one for
`Walk { x_end = x }` (solver wasm arena 6.3 -> 4.5 MB per run, 6.6 ->
5.6 ms; interpreter 0.144 -> 0.095 s). A reference
type (`Rng`, anything with an `inc`) keeps its objects and identity.

**Self tail calls (round 3).** `self_tail_calls` turns a call of a body
to itself whose value is returned straight away -- `fn count(n, acc)
-> int { if n == 0 { acc } else { count(n - 1, acc + n) } }`, the
lowering's forwarder chains between the call, the phi block and the
return followed -- into a jump: the arguments are assigned to the params
as one parallel assignment (an argument that is itself a param reads
the slot, so a cycle like `swap(b, a, n - 1)` goes through a scratch
local), a `trace_edge` keeps the self edge in the call graph, and
control jumps to a header block that took over the entry block's
instructions, so the entry block itself stays phi-free for SSA.
`count(100000, 0)` runs where the frame limit used to say "recursion is
too deep", and a walk can be written as the recurrence it is.

**The alias rule.** Codegen aliases a `get l` to the slot's register
rather than copying it, so such a read is not a snapshot: forwarding
another value's reads to a `get` of a slot that is written later would
see the later write (`let t = a; a = b; b = t`, or a param rebound by a
tail call). Every forwarder refuses: `promote_locals` keeps a local
whose store reads a written, unpromoted slot, `forward_value_fields`
leaves such a field read alone, `split_struct_phis` leaves such a
class alone (`volatile_reads`). `vm/tests/opt_semantics.rs` pins it
(`a_read_of_a_slot_written_later_is_a_snapshot`).

**The debugger's shadows.** A struct local the optimizer dissolves
keeps its name for the frame view through `Body::shadows`, a tree per
named local over the registers that now hold it (`compile::Shadow`:
`Slot` a register, `Struct` a struct rebuilt from its parts).
`scalar_replace` points the tree at the field slots it made, which are
live under every later store, mutation included. `promote_locals`
gives a promoted local (named, or a leaf of a tree) a shadow local and
an `Inst::Shadow` copy right after each store -- the debugger's move,
`Op::Shadow`, which the interpreter runs and bcgen, the JIT and wasm
drop. The copies cost the interpreter (the solver 0.095 -> 0.109 s,
the V2 loop 0.56 -> 0.60 s), so they are emitted only for a VM a
debugger will read: `Compiler::with_shadows`, `Vm::compile_debug` /
`compile_files_opts(.., debug)`; a program that only runs is compiled
without, and a slot-SROA tree whose leaf then leaves its slot is
dropped so the view shows null rather than a stale part. `split_struct_phis` treats a shadow of a class member as no
escape and splits it with the class: the leaf becomes a struct of
fresh leaves, every shadow into that leaf -- of a member, into the
field values; of anything else, into reads of its fields -- so the
tree is the same at every store, and a nested value splits again in
its own turn. The chunk carries the tree by name (`Chunk::shadows`),
and `Vm::frames`, `local_inspect` and `locals_graph` rebuild the
struct from the registers (`frame_locals`), a fresh instance per
read; a struct whose every leaf is still null is the local before its
first store. The program never reads a shadow slot, so the
optimizer's results are unchanged. Pinned by
`a_dissolved_struct_local_still_shows_in_the_frame`
(`vm/tests/debug_events.rs`) and
`a_dissolved_struct_local_keeps_a_shadow_per_field`.

**Value types for hosts.** `Program::value_adts` is the inferred set,
one flag per adt, and `HeapGraph` carries it per instance
(`Obj::Instance::value`, from `GraphNames::value_adts`) so a host can
draw a value inline in its holder like a scalar instead of as a box
an arrow points at -- a value has no identity to point at, and one
held twice honestly reads twice. The host's `autodiagram` does that
now.

## Tagged literals, spanned literal faults, prelude-first solving, `SeqSide`/`SortOrder`

- **Tagged literals.** A name glued to a string: `pat"bd sd"`,
  `keys"right@5"`. Lexes as `TokKind::Tagged` and lowers to a call of
  the prelude native `__lit_<tag>("…")`, so a host registers one native
  per tag and the literal is typed by that native's return and checked
  by its validator at compile time. A tag no native backs errors by the
  literal's own spelling (`mimas.str.tagged`).
- **`api::LitFault`.** A literal validator reports which argument and
  which char span inside it the fault is about; the solver places the
  error there (`frames::fault_span`). Module-qualified calls
  (`music::pat("…")`) now run validators, only plain and method callees
  did (`mimas.lit.validate`).
- **Prelude first.** `load_files` solves the built-in source and the
  library's own sources to completion before any user file, so a user
  type that takes a prelude name shadows it for user code (with a
  `shadowed_builtin` warning, `ShadowedBuiltinType`) instead of breaking
  the prelude's own impls mid-hoist (`mimas.prelude.shadow`).
- **Struct update.** `S { f = v, ..base }`, the listed fields change,
  the rest are read from `base` (same struct type, evaluated once);
  an enum variant literal refuses a base (`UpdateNeedsStruct`), since
  it might be another variant. `mimas.struct.update`.
- **`notation`.** `notation x(s: str) -> T { … }` is `fn __lit_x`, so a
  program's own tag `x"…"` has a reader; the word stays a name elsewhere
  (`mimas.notation.user`). The `ApplyMerge` intrinsic is gone: a host
  that wants a closure-taking combinator writes it in Mimas source
  (`Library::add_source`) and calls the closure itself.
- **Enum glyphs; bool glyphs.** `enum Suit { Spades "♠", … }` synthesizes
  an inherent `fn glyph(self) -> str` (every plain variant or none);
  `impl Opp for bool` reads ✗/✓.
- **Glyphs and symbol names.** `opp Sky { Sun "☀", Moon "☾" }` gives
  each end a glyph (`pact Opp`'s `fn glyph(self) -> str` defaults to
  `""`); the prelude ships ➕➖ ☀☾ 🌞🌙 ↗↘ ⬤● ⇤⇥. Names may be symbols
  or emoji (`is_symbol_ident`: arrows, shapes, misc symbols, dingbats,
  emoji), so `opp Sky { ☀, ☾ }` is legal; `Vm::resolve_chain` calls a
  zero-arg method chain on a top-level local for hosts that draw ends.
- **Opps named by their ends.** The prelude is `PosNeg`, `LightDark`,
  `DayNight`, `AscDesc`, `BigSmall`, `StartEnd`: the two ends glued,
  first end first, no alternative-spelling consts. The name says what
  the ends are and never squats on a plain word (`Side`, `Size`,
  `Order`); user opps can follow the rule mechanically (`LeftRight`,
  `SunMoon`). `AscDesc::Desc.rank/sort/argsort/arrange`,
  `StartEnd::End.pad`, `StartEnd::Start.justify`.
- **`RelativeSize` replaces `Size` and `Compare`; `Fill` and `Flow` are
  gone.** `Size { Big, Small }` took the name a width-and-height type
  wants and `Compare { Less, Greater }` was a second spelling of it;
  the one opp is `RelativeSize { Bigger, Smaller }` with `Greater`/`Less`
  consts. `Fill`/`Flow` had no users beyond their involution checks. The
  prelude: `Sign`, `Shade`, `Daytime`, `SortOrder`, `RelativeSize`,
  `SeqSide`, every name a qualifier and a noun.
- **Descriptive built-in opps.** `Side { Start, End }` is `SeqSide` and
  `Order { Asc, Desc }` is `SortOrder` (`SortOrder::Increasing`/
  `Decreasing` stay as consts), so the everyday `Side` and `Order` are
  free for user types, and `LrSide`/`TbSide` have room beside `SeqSide`.
  `SortOrder::Desc.rank`, `.sort`, `.argsort`, `.arrange`;
  `SeqSide::End.pad`, `SeqSide::Start.justify`.

## Axes and directions — which way a quantity points

A quantity can be *along* an axis, and a direction after it says which
way: `50.0px↑` claims 50 px measured up-positive along y, `x↓` and
`3.0px→` sign their ends, `5usd@in` is along an axis the program
declared. The builtin axes are `x y z t` with `left/right`, `down/up`,
`outward/inward`, `earlier/later` and the glyphs `← → ↑ ↓ ⊙ ⊗ ⏪ ⏩`; `axis
flow (out, in);` declares one (negative end first, four per
program) and `axis y (south, north);` renames a builtin's words. A word
is a direction only after its operand — `up 50` is a call to a name —
so no existing
program changes meaning. In a type the axis follows the unit: `px↕`,
`px/s↔`, `usd@flow` (`@name` for any axis); a literal takes the same
suffix, `40.0px↕`; `#[mimas_dim("px↕")]` carries it across a native field.

The algebra is parity, not exponents. `Dim` grew eight axis slots after the
unit slots (`units::AXIS0`); `*` and `/` xor them where they add the
exponents, so along-x times along-x is along neither (a square, a ratio),
along-x times along-y is along `x·y` (an area, a slope) and `slope * dx`
lands back on y — a quantity along x is one that flips sign when x is
mirrored, and that is the whole invariant. `+ - < ==` and slots need
*agreeing* axes (`Dim::agree`): equal, or one side untagged, which then
adopts the other's (`Dim::absorb`) the way a plain number meets an `index` —
`b.x + BALL` stays along x without tagging every size twice. Diagnostics
name both: `cannot add pixels (px) along y and pixels (px) along x`. The
direction word is parsed as a signed unary whose node the parser records in
`Ast::directions`, so at run time `up 50.0px` is the float `-50.0` and
nothing else moves. A direction is `Sign` on an axis, the opp made
spatial; `space` fields that are quantities flip the same way.
Tests: `library/tests/axes.rs`. Book: Units › Axes and directions.

## Math spellings, the big reductions, `⋅`, masks and gather

`≠ ≤ ≥ ¬ ∧ ∨ ∈ ∉` lex to the tokens their keyboard forms do, so the AST and
every diagnostic still print ASCII. `∑ ∏ ⨅ ⨆` are prefix reductions — the
big form of an operator is its fold — desugaring in `Parser::unary` to
`.sum()` `.product()` `.inf()` `.sup()`: meet and join, all-of/min and
any-of/max, the lattice glyphs rather than `⋀ ⋁` so they don't read as
boolean logic. They bind like a unary minus (`∑ xs.map(f)`, `⨆⨅ m`,
`(⨅ xs)!`), a list of lists reduces its innermost axis (one answer per
row), and `∑→ m` / `∑↓ m` sum a tensor along a direction (`sum_dim(1)` /
`sum_dim(0)`). `a ⋅ b` is `a.dot(b)` — contraction, APL's `+.×` — and tensor
`dot` falls through to matmul above rank 1. A list against a scalar
compares pervasively to a mask, nested as deep as the list, the way a tensor
does (`seq_cmp_overload_ty`; list-vs-list and anything-vs-null stay
structural), and `xs[idxs]` gathers. Tic-tac-toe's win check is
`⨆⨅ (b[LINES] == p)`, gather–compare–meet–join, and a test.
Tests: `library/tests/symbols.rs`. Book: Operators › Math spellings, Arrays ›
Gather and pervasive comparison.

## `raise` carries any value

`raise` used to take only `str`: `Val::Raised` stored the note directly,
and the solver rejected `raise 42`. Now the payload is any `Val`, boxed
as `Raised(Gc<Val>)` so a `Val` never holds a `Val` inline: `raise 42`,
`raise [1, 2]`, `raise {code = 7}` all work, and a raised value can even
carry another raised value (`raise inner()` nests to `<raised 7>`).

The point is *when* the text appears: the payload stays structured
through `Flow::Return`, `is_raised`, `?` demotion, binding, and the
snapshot/inspect/capture surfaces, and only stringifies at the cold
boundaries that need a `str`:

- `x!` and `()`-unwraps still fault `UnwrappedRaised(String)`, the
  display render happens on the fault path, where it's already cold.
- `absolve`'s handler is `(any) -> T`: `e` binds the payload itself, not
  a stringified note: `f"{e}"` formats on use, and `match` narrows it
  with runtime probes.
- `Raisable<T>` keeps its `String` variant. Native authors see no API
  change; a raised payload crossing `from_value` renders (or reads
  `<unprintable raise payload>`), and `into_value` wraps the message
  `Str` in the structured payload.
- `Captured::Raised`/`Inspect::Raised`/`SnapVal::Raised` all hold the
  payload recursively (`Box<…>`), so the web host's `inspect_json` hands
  JS a nested `{"$raised": …}` document. Richer raise/resume context
  without a flattening pass.

Cycle protection rides along: the payload `Gc` joins the render path's
pointer guard, capture/inspect `seen` sets, and snapshot node walk, so
`raise` on a cyclic structure errors deep instead of recursing forever.

The wasm backend needed nothing: `Val::Raised` there is already
`tag|word` with the payload as a raw word. The JIT/bcgen helpers grew a
`Ctx` arg only where cold-stringify lives (`unwrap`, `unwrap_unit`,
`raise`); `unwrap_raised` still just moves the payload.

Tests: `optionals.rs`. Structured payloads (`int`/`[int]`/`dict`),
`absolve`/`?`/`!` boundaries, nested raise, snapshot roundtrip;
`solve_results.rs`: `raise` on non-`str` solves to `T!`.

## `absolve`'s `e` is `any`: `match` narrows the payload

`raise` carries any value, so `absolve`'s handler parameter is the
dynamic type `any` rather than a pre-stringified `str`. `T` fulfills
`any`; `any` fulfills only `any`, it stays quarantined until narrowed,
which is what `match` is for:

```mimas
enum Flow { Push, Pull }
fn retry() -> int! { raise Flow::Push }

retry() absolve |e| {
    match e {
        Flow::Push => -1,
        Flow::Pull => 1,
        _ => 0,      // required — the payload could be anything
    }
}
```

- **Variant/struct/tuple-variant arms are runtime probes**: the pattern
  resolves the variant as usual and emits `is_instance`, but skips the
  scrutinee unify, the probe proves the variant, never claims `e` is
  one. Struct-variant fields bind with the variant's *declared* member
  types (`Err::Fatal { code }` gives `code: int`); a generic variant's
  params stay `any` (`Wrap::Full(x)` gives `x: any`, the probe proves
  `Full`, not `T`).
- **Literal arms are structural-eq probes** (`42`, `"x"`, `null`); `x?`
  is a non-null probe binding `x: any`. A `match` over `any` requires a
  `_` arm (the payload could be anything), and `match … !` still
  panics on a miss.
- **Quarantine**: `e + 1`, `e.field`, `|e| e` into a concrete return
  type, and `let n: int = e` all fail. Format (`f"{e}"`), match, or
  pass it somewhere expecting `any`.
- `==`/`!=` on `any` is allowed (structural eq is total at runtime), so
  `e == Order::Desc` works without a `match`.
- Any-scrutinee matches always take the linear probe chain, never the
  `switch` jump table: `switch` indexes `Val::Int` payloads and
  instance struct ids into the same table, and on `any` the two tag
  spaces can collide.
- The emit is the cold-path design carried through: `absolve` no longer
  emits a `Format`, the handler gets the raw payload and picks what to
  do with it. Handler closures are single-use call-site literals, so
  inlining one is a syntactic fact, not an analysis problem.

## Z-set `+`/`-` on sequences + the glued-postfix rule

`xs + x` appends one `x`, `xs - x` drops the first `==` match, `xs + ys`
unions (concat), `xs - ys` removes each `y` in turn, and `xs += x` /
`xs -= x` / `xs += ys` / `xs -= ys` come with them. This is the
weight-±1 surface of the algebra `Change` already carries weighted:
`xs + add(3)` and `xs + 3` now agree, and it replaces nothing anyone
could write: `[T] ± T`/`[T] ± [T]` were type errors before (the VM's
sequence broadcast arms were unreachable from solved code; `*`/`/`
keep theirs).

Two pieces worth knowing:

- The point/extent choice is *type-directed*, resolved the same way an
  `impl` would: `add`/`sub` are element-shaped natives, `union`/`minus`
  are list-shaped ones, and `xs + ys` falls through `add`'s element
  slot to `union`, so `[[1]] + [2]` appends `[2]` (it *is* a valid
  element), while `[[1]] + [[2]]` concats. `op_dispatch` now treats an
  arg that doesn't fit a candidate as "keep looking" instead of an
  error, which is also what lets `xs + c` still reach `c.radd(xs)`.
- `xs - x` is a quiet no-op when `x` isn't there. Arrays can't hold
  the negative multiplicities a `{int}` Z-set can (`xs - xs` empties).
  `[T; n]` places refuse `±=` like `push` does.

Postfix `!` now hugs its operand like `x%`: `x!` unwraps, but a *spaced*
`!` starts the next thing: `[!a !b]` is `[!a, !b]` (the parse everyone
meant), and `f !x` errors at the statement boundary rather than
silently meaning `(f!)(x)`. Same rule Zig calls partial precedence:
adjacency with two readings gets neither. `?` demote follows the same
rule: `x?` needs the hug too, so `f ?x` errors rather than reading
`(f?)(x)` (a spaced `?` opens nothing, so there's no second element
to win; the win is purely that the silent misread is gone).

Tests: `zset.rs`. Arithmetic, `±=` forms, element/list-type rejection,
the `[T; n]` guard, and the spaced-`!`/`?` parses both ways.

## `nand`/`nor`: the gate operators, word or glyph

`a nand b` and `a ⍲ b` spell the same op; `a nor b`/`a ⍱ b` likewise
(the APL glyphs are canonical, the AST prints `⍲`/`⍱` as it does `⊕`).
They sit with `&`/`|`/`⊕` in the bit/gate family: on bools they're the
*eager* forms: `!(a && b)`/`!(a || b)` with both operands always
evaluated, the parallel read where `&&`/`||` are the serial walk that
may skip the tail, and on ints they run per bit-lane, `~(a & b)`/
`~(a | b)`, a 64-wide gate array just as `&` is a 64-wide read.

What makes them load-bearing rather than trivia: `nand`/`nor` are each
functionally complete bases, and the choice between them is real:
`nand` is the existential gate (`¬∀`, any low input settles it, the
parallel-deciding shape, and in silicon literally a *parallel* pull-up
pair), `nor` the universal (`¬∃`, every input must report, the serial
one). `host/web/games/logic-kit.mimas` writes the whole story as `check`
lines: the lazy/eager witness (a `log` counting operand reads),
`nand = ¬all`/`nor = ¬any` over `[bool]`, `dual(f) = λ¬f(¬a,¬b)` as a
higher-order fn (`dual(nand) = nor`, `dual(xor) = xnor`), sheffer
completeness built from both bases, a Kleene three-valued `Tri`
overloading the whole family, and the `span`/`points` fencepost split
(half-open `a..b` holds `span(a,b)` points, dijkstra's rule, the +1
only appears when the extent is closed).

Dispatch: `nand`/`nor` are the first operator words that also name
methods: `mod`/`xor`/`and`/`or`/`not` are keywords or reserved infix
spellings, but `fn nand` stays an ordinary declaration (contextual like
`xor`), so `a.nand(b)` and `a nand b` are the same call. Overload names
are `nand`/`rnand` and `nor`/`rnor`.

Tests: `operators.rs` `nand_nor_builtin_gates` /
`nand_nor_dispatch_to_impl_methods`; the kit's 13 `check` lines run in
`logic_kit::logic_kit_check_lines_all_pass`.

## Operator overloading: operators resolve to ordinary methods

`a + b` is a method call: the left operand's type is asked for `add`, and
if it doesn't provide one the right operand's is asked for `radd`, so
`s * v` finds `v.rmul(s)` with `self` always the operand that owns the
implementation. The same naming covers the whole infix surface:
`add sub mul div idiv rem band bor bxor shl shr` for `+ - * / ~/ mod &
| ⊕ << >>`, `lt le gt ge` for the comparisons, `neg pos bnot inv` for
prefix `- + ~ !`. Plus `r`-prefixed reverse forms for every binary
name. Names dodge the keywords (`rem` not `mod`, `bxor` not `xor`),
and `!`'s is `inv`, an ordinary name, so overloading it never shadows
the `not` operator word; types without `inv` keep `!` bool-only, and
`not x` spells the same `x.inv()` call. `==`/`!=` stay structural and
`&&`/`||`/`??` stay non-overloadable.

Resolution is static and recorded: the solver writes an `OpDispatch`
(`dec` + `swap`) into `Resolutions::op_decs` keyed by the operator node,
and codegen lowers to an ordinary direct/native call. Receiver first,
operand second. Rather than a generic `Bin`. An operator-shaped method
must take exactly one operand (zero for unary); a `fn add(self, o, k)`
is just a method and `a + b` keeps its plain invalid-op error. Compound
assignments ride the same dispatch: `x += y` calls `x.add(y)` and stores
the result: `??=` keeps its option semantics.

Two consequences fell out of making it ordinary methods:

- **Vectors, tensors and deltas speak infix.** `std::math`'s `Vec2`/`Vec3`
  ship `mul`/`rmul`/`div`/`neg`, so `v * 2.0`, `2.0 * v`, `v / 2.0` and
  `-v` all work (`add`/`sub` were already methods: `v + w` arrived
  free). In game-kit, `Change<T>` gets `radd`/`rsub`, so `xs += add 3`
  applies a delta and `xs -= remove 2` applies its undo.
- **CUE-style unification is a library.** `host/web/games/cue-kit.mimas`
  defines a `CVal` ADT (top/bottom/atoms/lists/structs/disjunctions/
  bounds/defaults) and implements `band`/`bor` as `meet`/`join`, so a
  config schema reads `{port = ge(1024) & le(65535)}` and `schema &
  record` unifies fieldwise with `⊥` propagating. The lattice laws are
  `check` lines run by `cue_kit::cue_kit_check_lines_all_pass`.

`crates/library/tests/operators.rs` pins the contract: left dispatch,
reverse dispatch, non-bool comparisons, compound assign, structural
equality untouched, and arity/type rejections.

## `and`/`or`/`not`: the logical operators in word form

C++-style alternative tokens: `and` and `or` lex straight to
`DoubleAmpersand`/`DoublePipe`, and `not` lexes to a new `NotWord` token
that parses as prefix `!` (never a postfix unwrap: `x!` stays `Bang`
territory). Same precedence, same short-circuit; `t or f and f` is
`(t || f) && f`. Source canonically spells `&&`/`||`/`!`, the words
exist so the editor can show and accept them as a per-person reading
option, and so `not x` stops being a foreign-spelling teaching error.

**A source can take a word back.** A file that declares `fn not` (or
`fn and`/`fn or`, `pub` or not) gets that word as a plain name: the
lexer's pre-scan (`lex::shadowed_words`) sees the declaration and lexes
the word as an `Ident` in that page, so `not(a)`, `map1(a, not)` and
`not a` all call the fn, and the operator is spelled with its symbol.
The reach is directional: a declaration covers its own page and the
pages that `use` it, never the pages it uses (`lex::shadow_sets`), so
n2t-lib's `fn not(a: int)` reaches n2t-gates but not game-kit's
`not UP`. Per file in `load_files`; in one spliced source the host's
splice writes a `//# page: name` line (`lex::PAGE_MARK`) at each page's
start and `lex::shadow_table` bounds pages there. The
grades pass warns at the declaration: `shadowed_builtin`, a stable
miette code a host can filter on (Nina hides it per page with the
`allow` page property). nand2tetris's gate lib is the motivating page.
Tests: `parse_validity::a_declared_fn_takes_back_an_operator_word`,
`library/tests/juxta.rs` (`a_declared_fn_shadows_its_operator_word`,
`shadowing_an_operator_word_warns_with_a_code`,
`a_kits_shadowing_reaches_the_page_that_uses_it`),
`parse_validity::a_spliced_source_shadows_per_page`.

## `space`: axes taken together

`space Politics { econ: Hand, power: Grip }` declares a struct whose
fields are all axes, plus a synthesized `impl Opp` that flips every field
(`Politics { econ = self.econ.inv(), … }`), so `!p` is the opposite
point, a space is an axis itself, and spaces nest. An axis is any type
implementing `Opp`: an `opp`, `bool`, a number (`impl Opp for int` /
`float` in the built-in source: the flip is negation, so `px` and other
quantities count), a `Dial`, another space. A field without a flip gets a
targeted error at the field ("`name: str` isn't an axis, so it can't be
part of `space S`"). Numbers serve static flips only, a space calls its
fields' `inv` directly, and don't travel as `Opp`-typed values, since a
pact-typed call tests the value's kind at runtime and a number has no
test (`involutive(3.0)` is a type error). Tests:
`space_flips_every_axis`, `space_fields_must_be_axes`.

## `datalog { … }`: Ascent's language, run in mimas

A `datalog` block declares relations, facts and rules, evaluates them to a
fixpoint, and yields the relations in declaration order (a tuple of arrays
of tuples):

```
let (edge_, path_) = datalog {
    relation edge(int, int) from pairs          // or `rel edge(a: int, b: int)`
    relation path(int, int)
    path(x, y) <-- edge(x, y)                    // or `if`
    path(x, z) <-- edge(x, y), path(y, z)
};
```

The language follows Ascent (the Rust Datalog crate) but runs in mimas,
so rules can call the page's own functions and it works in the browser.
A body holds atoms, conditions (`if cond` or bare), `let x = e`, `for x in
e`, negation `!rel(args)` / `not rel(args)` (every variable bound to its
left), and aggregates `agg x = f(e) in rel(args)`: `count()`, `sum`,
`min`, `max`, `mean` (over no tuples only `count` derives). Rules are
stratified the way Ascent's are: a head sits at least as high as every
relation it reads and strictly higher than one it negates or aggregates,
so that relation is complete when asked; depending on your own negation or
aggregate is an error. Each stratum runs semi-naively to its own
fixpoint.

The parser lowers a block to plain mimas (`parse::parser::datalog`): one
array per relation, a hash-bucketed dedup index, a `loop` per stratum
whose rounds only revisit combinations touching last round's delta. No new
runtime. Not yet: lattices, `if let`, several heads per rule. Tests:
`library/tests/datalog.rs` (closure, seeds, filters, the carp-kit and
learned lessons, Ascent spelling, negation, aggregates, errors).

## `opp`: a dichotomy, as basic as an enum

`opp Sky { Sun, Moon }` declares a type with exactly two ends, each the
other's opposite: the enum of its two ends plus a synthesized `impl Opp
for Sky` (built by the parser, every node located at the declaration)
whose `inv` swaps them. So `!Sky::Sun == Sky::Moon`, `not` spells the
same flip, and `fn involutive(x: Opp) -> bool { !(!x) == x }` takes any
opp: `!` dispatches through the pact at runtime. Anything but two plain
names is an error ("an `opp` has exactly two ends"); `opp` stays an
ordinary name wherever a type name and `{` don't follow it.

Every program has a small built-in source (`solve::BUILTIN_SRC`,
appended last by `load_files` so user files keep their ids): the `Opp`
pact itself and the pairs pages kept spelling as bools or ±1:
`Sign { Pos, Neg }`, `Size { Big, Small }`, `Shade { Light, Dark }`,
`Daytime { Day, Night }`, `Flow { Push, Pull }`, `Fill { Full, Empty }`, `Order { Asc, Desc }` (a
sort's direction. Std and pages take it instead of an `asc: bool`),
`Compare { Less, Greater }`, `Side { Start, End }`. Alternative spellings
are associated consts: `Order::Increasing`/`Decreasing`,
`Compare::Smaller`/`Bigger`, so `!Order::Increasing ==
Order::Decreasing`. The core source (`solve/src/builtin.mim`) is
declarations only; std's operations on the opps ship with std as mimas
source (`library/src/opp.mim`, via the new `Library::add_source`), on
the opps themselves so they add no global names: `Order::Desc.rank(keys)`
(indices, Desc the exact reverse of a stable Asc: `xs.reorder(...)`
sorts by key), `Side::End.pad(xs, n, fill)` (extend any list, the
zero-extend case) and `Side::Start.justify(s, width)` (align text). A host asking a variant's value for a
method (the debugger's `draw`, Nina's switch calling `inv`) finds the
enum's impl: the host-facing method table registers an enum's methods
under each variant's runtime id too. Each
mirrors: `(!o).rank(k) == rev(o.rank(k))`, `(!Side::End).pad(xs, n, f) ==
rev(Side::End.pad(rev(xs), n, f))` (`builtin_opp_operations_mirror`).
Built-in and library sources solve first (their impls' consts are typed
before user code reads them) while keeping file ids after the user's.

A field-less variant is a constant: `const ME = Team::Blue;` and
`const BOTH = [Team::Blue, Team::Red]` (built at each use, like struct
consts); a tuple variant's constructor is still not a value
(`unit_variant_consts`, `solve_variables::const_tuple_variant_ctor`). `bool` is the first opp
in spirit; `impl Opp for bool` waits on pact impls for primitive types.
A unit variant now prints as its bare name (`Sky::Moon`, not
`Sky::Moon {  }`). Tests: `parse_validity::opp_is_an_enum_and_its_flip`,
`library/tests/structs.rs` (`opp_declares_a_dichotomy`,
`unary_operator_through_a_pact`).

**`bool` is an opp.** The built-in source has `impl Opp for bool` (`inv` is
`!self`), so a bool goes wherever an `Opp` is expected: `involutive(true)`.
Through the same runtime dispatch as an enum (a bool arm tests the value
with `x == true || x == false`; there's no instance id to test). A pact
may now be implemented for a primitive (`impl P for bool`; `bool` only for
now), `Self` inside is the primitive, and a primitive satisfies a pact its
library adt implements. A primitive's operators stay built in: `!b` is
never dispatched to `inv` (which is `!self`, so it couldn't recurse),
and the same holds for every operator on `int`/`float`/`str`/`bool`
(`bool_is_an_opp`).

## `check (expr)` opens a check

Juxtaposition made `check (a * b).same(c)` the call `check(a * b)…`. A
gap after the word now opens a check before `(`, `[`, `{` and `-`, as
it always did before a name or literal; `check(x)` and `check[i]`
touching the word stay a call and an index
(`parse_validity::check_opens_before_a_spaced_paren_bracket_or_minus`).

## Const arrays may hold structs

`const DIRS4 = [RIGHT DOWN LEFT UP]` over struct consts used to be
"non-constant" (and `[P { x = 1 }]` an ICE): only scalar arrays have a
bytecode `Constant` form. A const array/tuple that holds a struct or
dict (directly, nested, or through a const ref), now allocates at each
use site, the way a struct const already did (`Constant::needs_alloc`
in `emit_const_dec` and the module-constant export), so a write into
one use never reaches the next. Tests:
`solve_variables::{const_struct_in_array, const_array_of_struct_consts}`,
`library/tests/structs.rs::const_array_of_struct_consts`.

## Typed arrays: `IntArray`/`FloatArray` backing stores: branch `typed-arrays`

Homogeneous `[int]`/`[float]` sequences stop paying `Vec<Val>`'s 16-byte
tagged cells (and a GC trace per element): they get a structure-of-arrays
backing instead: `Gc<RefLock<ArrayStore>>` where `ArrayStore` is
`Empty | Ints(Vec<i64>) | Floats(Vec<f64>) | Vals(Array)` (`val.rs`).

**Distinct `Val` tags on purpose.** `Val::IntArray`/`Val::FloatArray` are
separate discriminants, so every existing `Val::Array` pattern. Including
the bcgen-generated and JIT-inlined fast paths that borrow the `Vec<Val>`
directly. *misses* typed arrays and falls through to the helpers that box
elements back out. That's the safety property; no inline path can misread a
`Vec<i64>` as `Vec<Val>`. The JIT needed zero changes: `GetIndex`/`SetIndex`/
`GetField`/`SetField` tag-compare against `t_array` before touching the vec
layout, and `Push`/`NewArray`/`Len` were already helper-backed (`bc.rs`'s
`jit::*` shims now share the same `seq_push`/`new_seq` bodies).

**Promotion/demotion policy.** `[]` (`Ctx::new_seq`) starts `Empty`; the
first `Push` of an int/float picks `Ints`/`Floats`. `Ctx::array_val` maps an
all-`Int`/`Float` `Vec<Val>` to the typed tags (constants, `collect`
results, native `Vec<T>` returns. Tuples like `(1,2,3)` included).
Promotion happens *only* at construction, an existing `Val::Array` never
upgrades. Any write that doesn't fit demotes in place: `set`/`push` box the
store into `Vals`, which holds a real `Array` handle so a native that took
`as_untyped_array` keeps mutating the very contents the typed `Val` still
observes. The tag is a birth hint; the store is the truth (an
`IntArray`-tagged value can hold `Floats`).

**Everywhere else it's just an array.** Reads box to `Val::Int`/`Val::Float`;
`==`, `display`, `capture`/`inspect`, iteration, `in`, `len`, broadcast
`bin`/`unary`, deep-clone, snapshots (`SnapNode::IntArray/FloatArray` ride
unboxed) and rebind all go through `ArrayStore`'s boxed accessors, so
semantics. Including cross-representation equality between a typed array
and a `Vec<Val>` array. Are identical. `crates/vm/tests/typed_arrays.rs`
pins both halves of that contract.

## `|>`: the pipe operator is gone

`|>` used to desugar `x |> f(a)` to `f(x, a)` — the loosest infix, and the only
one that could lead a line. It is removed: a second spelling for call chains
(`x |> f()` next to `x.f()`) kept asking which free function a method-only name
resolved to, and method chains already cover every real pipeline. The corpus
now writes the verbs as methods:

```mimas
samples
    .arrange(["rider", "clock"], [false, false])!
    .mutate([when(col("rider") == col("rider").shift(1),
                 col("clock").diff(1),
                 col("clock") - lit(START)).alias("laptime")])!
```

The `std::polars` verbs stay dual-registered (method and module function), so
`df.filter(..)` and `filter(df, ..)` both work — only the `|>` infix is gone.
A `|>` left in old source lexes as `|` then `>` and fails to parse.

## Units across the native boundary: `param_dims` / `return_dim` on native signatures

Natives used to be `Any` at the boundary: `music::play_for(t, n, amp, 0.5s, 1200Hz)`
in the wrong order compiled fine because no argument dimension ever reached the
checker. Now a native parameter declared as a unit-carrying Rust type:
`Secs`, `Hz`, `Beats`, `St`, `Tempo`, `Bits` (thin `f64` newtypes in
`mimas::vm::units`). Lands its `Dim` on `ApiFunction`/`ApiMethod`
(`param_dims`, `return_dim`), parallel to the `Vec<Option<Ty>>` slots that
`MimasType::mimas_ty` already produced. `MimasType` gains `mimas_dim`,
`IntoNativeResult` gains `return_dim`; `Option<T>`/`Raisable<T>` forward the
inner dim.

- The dims pass resolves the call's `DecId` through `node_decs` (all three
  call shapes. Ident, `x.m()`, `mod::f()`. Record it) into `dec_to_native`,
  then checks each arg positionally against `param_dims` and returns
  `D::Q(return_dim)`. Undeclared slots stay `None` = unchecked, and a native
  declaring *no* dims falls through to `builtin`'s special cases (`pull`,
  `to`, `pow`, …) untouched.
- The check rejects *known* mismatches, a plain `0.4` or `1200Hz` into a
  `secs` slot errors, while `Any` (other natives' returns, dynamic values)
  still flows. So adopting a unit on one slot never breaks callers that
  produce the value dynamically.
- `music::synth`/`adsr` take `Secs`/`Hz`, `play_for`/`play_with` take
  `Hz`/`Secs`, `transpose`/`rate` take `St`, `hz` returns `Hz`. zzfx tables
  stay plain by contract. Feed a measured value into a slot list with
  `.to("Hz")`, which is exactly the conversion the checker asks for.
- `Dim`'s arithmetic (`mul`/`div`/`powi`/`sqrt`) is `const` now, so a wrapper
  can spell its dimension as a `const` (`Tempo::DIM = BEAT.div(TIME)`).

## `warn`: a non-fatal, source-located diagnostic channel

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
- `Prints` entries are `(Location, PrintKind, String)`, every consumer of
  `take()` handles the kind.

## DataFrame rows as records: `df.row(i)` / `df.rows()`

`df.row(i)!` hands back one row as an instance of the declared struct whose
fields are exactly the frame's columns, filled by name; `df.rows()!` gives them
all in order. The point is `cleared_1k(shuttle().row(2)!)`, a real table row
going straight into a function typed on the record.

- The struct resolves by field-name *shape*, not by name: zero matching structs
  and several matching structs both raise (`row: no declared struct has fields
  […]`, `… match more than one declared struct (…)`). Out-of-bounds raises.
  Missing cells materialize as `null`.
- Two paths to the record type: the declared return is `anon` (`Anon`/`ArrayOf`
  slot 0) and unifies wherever the row is used, that's the only path for an
  opaque frame (`from_csv`). For a closed-schema frame (`table {}` literal,
  `.schema("…")`, or a schema-propagating op) the solver's `frame_method_call`
  finds the adt up front via `record_adt_for`, so `df.row(0)!.riders` type-checks
  directly without annotation.
- `to_dataframe` is the exact inverse: array-of-structs in, struct-rows out.
- Both are free fns and `DataFrame` methods in `std::polars`; `row` is also how
  `query`-derived frames read back, any frame whose columns match a declared
  struct's fields decomposes, so `derive`d extra columns need a wider struct or
  a `select` first.

## xlsx spreadsheet I/O: `std::polars`

`from_xlsx(path, sheet?)` reads a worksheet into a `DataFrame` via calamine;
`df.to_xlsx(path, sheet?)` exports one via rust_xlsxwriter (new files only,
no editing existing workbooks). Behind the default-on `xlsx` feature on
`library`/`mimas`, same opt-out story as `dataframe`/`darkly`.

- Type inference on read: whole numbers `i64`, any decimal widens `f64`,
  bools `bool`, Excel datetimes `datetime[μs]`, mixed columns fall back to
  per-cell `str`, Excel errors preserved as text, empty cells null. Formulas
  read cached values, never evaluated.
- `rust_xlsxwriter`'s `polars` feature stays off, it would resolve a second,
  crates.io `polars` that doesn't unify with the workspace's path-dep clone.
- Deferred: in-place workbook editing → `umya-spreadsheet` if the need ever
  shows up. SheetJS and the C `xlsxwriter` bindings were ruled out.

## Cranelift JIT tier: `mimas-jit`

Restored in-tree (2026-10-31) alongside `mimas-bcgen`/`mimas-bcgen-test`, and
ported to the `*const BodyEnv` `BodyFn` ABI: emitted bodies take the single
env pointer and load `thread`/`code`/`mutation`/`state`/`strs`/`fuel`/`op_ip`/
`out` fields via `offset_of!` at entry; `call_body`/`call_dyn` and inline
`CallDirect` calls forward `env` verbatim instead of ten flat params.
(`mimas-llvm`'s `llvm-jit` MCJIT lane is intentionally not merged, LLVM's
compile latency made it the wrong tier; pure-Rust lanes only.)

Second Futamura projection: `jit::compile(&program)` emits one native function
per bytecode chunk with Cranelift (`FunctionBuilder`, dense op dispatch through
`br_table` on a byte-offset→op-index map) and returns a `Vec<Option<BodyFn>>`
for `Vm::install_bc`, the same slot bcgen bodies use, so interpreter/JIT
bodies interleave freely.

- Scalar int/float registers live in SSA vars ("shadows" with ok-flags) inside
  a body; they flush to the register window before any call that observes it.
  Unsupported ops delegate to `bc::jit::step_at` (= the interpreter's own
  `step_one`), so semantics can't drift, only speed can.
- `vm::bc::jit` is the `extern "C"` helper layer bound via `JITBuilder::symbol`
  (`mj_*` names in `crates/jit/src/lib.rs::SPECS`, `H` indexes it 1:1).
  **ABI gotcha:** `Ctx` is a repr(C) 2-pointer struct; under AAPCS64 a composite
  that doesn't fully fit in remaining arg registers goes *wholly* to the stack,
  while a flat Cranelift signature would split it, so `call_native` (ctx at
  arg slot 7) takes `mc`/`st` as two pointers and rebuilds via
  `Ctx::from_parts`. Any new helper with `Ctx` past arg 6 must do the same.
- Calls: `CallDirect`/dynamic `Call` run inline below `INLINE_CALL_DEPTH` via
  `jit::enter` (=`enter_call`) + a direct/indirect call into the callee body
  (fn-ptr table in JIT module data), `pop_return` on `Flow::Return`; deeper
  frames return `Flow::Call` to the driver as before.
- Parity: `mimas-jit-test` runs each fixture under both lanes and diffs
  `TEST_VALUE` + `keep` log + error. Including op-budget *counts*, fuel
  windows, pause/resume (`run_frame`), snapshot/restore, and GC pressure.
- Bench (bc-bench `jit` lane, M-series arm64): fib_iter jit 172ms vs bc 254ms,
  mandelbrot 519 vs 710. Beats pre-tuning bcgen on scalar loops; fib_rec
  2313 vs 1089 and physics 2897 vs 704 lose (call-shim + flush overhead is the
  known gap vs the batched-quota/loop-form bcgen on `bcgen-tuning`).

## bcgen second specialization pass: branch `bcgen-perf`

On top of `bcgen-tuning`'s scalar shadows and batched quota gates:

- **Sparse multi-block loop wrapping** (`emit_body` + `grow_region` in
  `crates/bcgen/src/lib.rs`): a bounded DFS grows a loop region over
  continuation edges: jump targets first, then conditional fallthrough.
  Backtracking on dead ends. Regions stay ascending and validate iff every
  interior edge lands on the head (`continue`), the next emitted op, or
  outside the region (`code.ip` + `break`). Handles `while A && B`
  short-circuit re-entry (mandelbrot's inner loop) and physics' 13-block
  loop; `Switch` regions are rejected outright.
- **Bool shadows**: `W::Bool` joins int/float in `analyze`. Comparisons,
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
(the extra copies cost more than the `rd` slot loads they saved. +6–7%).

## JIT round-2 perf: branch `jit-perf` (PR haydenflinner/mimas#7)

Four layered optimizations over the tier above, all semantics-preserving
(helpers/`estep` still provide the ground truth):

- **Quota batching (M1):** CLIF `bcn`/`bcn0` vars mirror bcgen's
  `settle!`/`gateq!`/`gatep!`: each op decrements a batched counter; pause,
  fuel-out and ops_left exhaustion escape through cold gate trampolines;
  `settle` runs at every observable boundary (calls, helper ops, exits).
- **Bulk shadow flush (M2):** one `mj_flush` FFI call drains a packed
  `FlushEnt` list. Replaces per-register `wr_i`/`wr_f` calls.
- **Megacalls (M3):** `mj_call_body`/`mj_call_dyn` fold callee resolution,
  `enter_call_regs` frame push, callee `BodyFn` invocation, and the
  `Flow::Return` pop/truncate/dst writeback into one FFI hop, returning the
  rebuilt caller-window pointer for re-pinning.
- **Direct `Val` access (M4):** `bc::jit::layout()` probes the live layout.
  Discriminant position/width, `Int`/`Float`/`Bool`/`Fn` payload offsets,
  `ThreadState`/`Frame`/`Decoder` fields, `Vec` header order, by semantic
  round-trip (mutate bytes of a known `Val`, `ptr::read` it back and compare
  `discriminant()`; invalid patterns decode as "other", never abort).
  Emitted code then does tag compares + payload loads/stores inline:
  `ri`/`rf`/`rval`/`wr_*` FFI calls vanish from hot paths.
  **Gotcha:** `size_of::<Discriminant<Val>>` (8) is the *token* width, not
  the field width. Rustc leaves stale bytes in discriminant padding, so
  the probe tries widths smallest-first and emits 1-byte tag compares.
- Result (release): mandelbrot jit 117ms vs bc 199ms (**jit wins x1.7**);
  fib_iter 89 vs 60; prime_numbers 334 vs 255; fib_rec 1006 vs 723 and
  physics 1065 vs 273. Call-heavy still trails; next lever is inlining /
  direct body dispatch to skip the megacall FFI hop.

## bcgen generated call path: branch `bcgen-calls2`

Recursive `CallDirect` was ~330x rustgen on `fib_rec`. Profiling the
generated call path found the costs in order: (1) `Val`'s derived
`PartialEq` never inlines, so `rd(regs, r) == Val::Bool(x)` on `JumpIf`
conditions without a bool shadow compiled to a *call* per branch, two per
mandelbrot inner iteration and one per `fib` call frame; (2) full
`Vec::resize(..., Val::Null)` + `enter_call_regs` bookkeeping on every
generated-to-generated call; (3) `RtResult<Flow>` construction on every
return and quota settle/re-arm at each boundary.

- **Const-folded `enter_call_regs`** (`emit_call_fast`): snapshots args,
  writes the callee window into spare capacity directly, Null-fills only
  uncovered slots, emits a literal `Frame`; generic helper stays as the
  capacity/dynamic-call fallback.
- **`u8` return tag**: successful inline returns run `gen_return_pop`
  (frame pop, `regs.truncate`, IP restore, dst writeback) and yield 0;
  tag 1 means `io.out` holds the callee's `RtResult<Flow>` verbatim.
- **`GenIo` boundary bundle**: `out`/`fuel`/`op_ip`/quota-pair/`inl` move
  into one struct, the inner ABI drops from 11 params to 7 (no AArch64
  stack spill) and `[bcn0, bcn]` rides the same `io` across inline calls,
  so quota settles once per propagating exit instead of once per call.
- **`gbail` cold exit**: all tag-1 exits funnel the `io.out` write through
  a `#[cold]` helper. Keeps big call-free bodies small enough that LLVM
  stops spilling loop-carried shadows.
- **`matches!` discriminant checks**: `JumpIf` and `GetField` option-null
  checks emit `matches!(v, Val::Bool(..)/Val::Null)`, same predicate,
  no `Val::eq` call.
- Debug `INLINE_CALL_DEPTH` is 48 (release 256): generated debug frames
  are ~7.7KB, so ~96 nested inline calls overflowed the parity thread's
  stack.

Measured (`bc-bench` bc lane, min-of-5, vs `e5b6214` binary): fib_rec
714→534ms (−25%), mandelbrot 90→54ms (−40%), prime_numbers ~202ms flat,
fib_iter ~56ms flat, physics ~268→280ms (+4.5%, regalloc noise). rustgen
`fib_rec` is ~2.2ms, still ~245x ahead.

## JIT round-4 runtime specialization: branch `jit-spec`

Observe the live VM, recompile with what was seen, reinstall, no VM
surgery: `jit::JitSession::observe(&mut vm, &program)` installs one
profiling `BodyFn` shim per body through the ordinary `install_bc` table
(each dispatch runs the interpreter's own `step_at`, so observation costs
interpreter speed, not correctness); `session.facts()` snapshots a `Facts`
struct; `session.specialize` recompiles via `jit::compile_with` and swaps
the bodies in mid-run.

- **`Facts`/`BodyFacts`** (`crates/jit/src/lib.rs`): per-body `entry[r]`
  merged `Obs` (tag + stable GC payload ptr), `calls` (call-site resume ip
  → the one callee `BodyId` ever seen), and `sites[ip]` (the merged
  *receiver operand* of each `GetIndex`/`GetField`. Sees mid-body values
  `entry` can't, e.g. a global `LoadEntry`'d config dict). `Obs` merges
  decay to `Mixed`/`ptr=0` on disagreement, so polymorphism degrades
  gracefully.
- **Forced entry shadows**: an `entry` tag of `Int`/`Float` puts the reg
  in the scalar-shadow set unconditionally; the entry probe's ok-flag is
  the runtime guard and every write site refreshes or clears the shadow,
  so a wrong observation only ever deopts through `estep`. Never
  miscomputes (`spec.rs::wrong_entry_facts_deopt_not_break` feeds a lying
  `Float` claim at an `Int` param and still gets 42).
- **Monomorphic call ICs** (`emit_call_ic`): `Op::Call` sites with a single
  observed callee emit two guard arms: `Val::Fn` tag + body-id compare,
  `Val::Closure` tag + guarded `cl_func` deref + `captures.len` match,
  each routing through `call_gates`/`inline_enter_run` (the refactored
  `emit_call_direct_inline` core). Any miss runs `mj_call_dyn`, which
  reproduces `not_callable`/`WrongArity`/`Flow::Call` verbatim.
- **Frozen-constant baking**: `Facts::frozen` maps a `Gc::as_ptr` payload
  address to `{kind, unguarded}`, a host promise of immutability for the
  bodies' install lifetime (break it and you get stale reads, not
  crashes). `Dict` bakes entries into a key-pointer compare chain (cap 32,
  `Str` equality is pointer equality); `Array`/`Seq`/`Instance` fold to a
  compile-time `(base, len)` view; `GetField` bakes `fields[slot]` to an
  immediate `Val` when it's a scalar/primitive. Every site still
  tag+pointer-guards `regs[r]` unless `unguarded` (honored only for regs
  never re-seated in the body).
- Bench (`jit-test --example specbench`, release, M-series arm64:
  `run(500_000)` where `tick` reads `CFG["rate"]`/`["scale"]`/`["bias"]`
  per call): interp 127.5ms → jit generic 95.7ms (1.33x) → jit + facts
  **32.9ms (2.91x vs generic, 3.87x vs interp)**: frozen-dict reads turn
  ~50% of the loop body's runtime constant traffic into immediates.
- Tests: `jit-test/tests/spec.rs`. IC hit/reseat-miss, polymorphic site
  staying generic, frozen config bake from a `LoadEntry`'d global, wrong
  facts deopting, `specialize_now` parity.

Host-facing quirk to know: `Vm::call` uses entry reg 0 as its return
scratch and restores it afterwards, so a callee's `StoreEntry` to global
slot 0 inside an injected call is silently restored away. Keep mutable
globals out of slot 0 (see `call_ic_hit_and_miss`'s `pad`).

## Incremental operators: `std::incremental`

`incremental(q)` (`q^Δ`) in `group.mim` is `∇ ∘ q ∘ ∫`: correct, but each
step rebuilds the whole state and runs `q` on all of it. `incremental.mim`
gives the same changes from the changes alone. Each operator is a value
with a `step` method (one change in, one change out) and keeps the state it
needs between steps; `run` steps a whole stream.

- `integrator(z)`, `differentiator(z)`, `delayer(z)`: `∫`, `∇` and `z⁻¹`
  with one value of state, for any `AbelianGroup`.
- `linear(q)`, `zmap_op`, `zfilter_op`, `zflat_map_op`: linear, so a step is
  `q` on the change, with no state.
- `zjoin_op(ka, kb, both)`: bilinear, `Δa ⋈ B + A ⋈ Δb + Δa ⋈ Δb`, with both
  inputs indexed by key.
- `zdistinct_op()`: an item's output changes only when its weight crosses 0.
- `zgroup_sum_op(key, val)`: takes back the old row and adds the new row of
  each group the change touches.

Each has a batch form (`zmap`, `zfilter`, `zflat_map`, `zjoin`, `zdistinct`,
`zgroup_sum`). `tests/incremental.rs` checks every operator against
`incremental(q)` or against `∇` of the batch query on the states, on seeded
random change streams with removals and re-adds. A one-row change to a
join over 4,000 rows a side costs 214 VM ops; recomputing costs 760,219.
`examples/incremental-build.mim` writes a build as a circuit: items `(name,
body, iface)` and edges in, the change of `{ (name, body, Σ iface of deps)
}` out, so a body edit rebuilds one item and an interface edit also
rebuilds its direct users.

Recursion is `zfixpoint_op(make)`, a nested circuit for `R = f(I, R)` (batch
form `zfixpoint`). `make()` builds a copy of the body: a closure over fresh
operators that turns `(ΔI, ΔRᵢ)` into `ΔRᵢ₊₁`. There is one copy per inner
iteration, each fed the outer-time changes of its inputs. A step runs at
least as deep as any step before, and stops when `ΔRᵢ₊₁ == ΔRᵢ`: past the old
depth the previous state had converged, so equal changes mean the new state
has converged too. A step that goes deeper primes each new copy with what
the deepest copy was fed before (`∫ ΔI` and its summed `ΔR` inputs), which
is exact because every operator's state is the sum of its inputs.
Transitive closure is `distinct(ΔE + join(ΔRᵢ, ΔE))` in the body.
`examples/incremental-build.mim` has a second circuit, `DeepBuild`, without
interface hashes: an artifact depends on every body it reaches through the
closure, so a body edit rebuilds everything that reaches it.

`Tally.put_all(o)` adds `o`'s weights in place, written on `put`.

The join and group-by indexes key on a key's printed form, as `Tally` does,
until value-keyed dicts land.

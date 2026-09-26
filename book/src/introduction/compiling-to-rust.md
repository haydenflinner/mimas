# Compiling to Rust

mimas normally runs on a bytecode VM, which is what the numbers on the
[benchmarks](./benchmarks.md) page measure. But the same `parse → solve`
frontend produces a fully-typed AST, and nothing stops that AST from going
somewhere other than the bytecode emitter — so `rustgen`
(`host/eval/src/rustgen.rs`) transpiles mimas source directly to Rust.

Same `.mim` file in, `gen.rs` out. The generated code links against a small
runtime shim instead of the VM:

- **`mrt.rs`** — mimas's collection and stdlib surface re-implemented over
  persistent structures: `imbl::Vector` for `[T]` arrays,
  `indexmap::IndexMap` for `~{T}` dicts (insertion-ordered, exactly like
  the VM's `DictMap`), plus JSON and scalar helpers.
- **`pl.rs`** — `std::polars`'s `__q_*` surface over real polars, so
  `query { }` blocks work natively too.
- **`game::*`** — headless stubs. Native code never draws; it exists to run
  rules, search, and servers fast.

## How much faster?

Same source files as the interpreter benchmarks, transpiled and run in
release mode, best-of-5 wall clock on the same machine:

| workload | mimas VM | rustgen→native | speedup |
|---|---:|---:|---:|
| [fib_rec](https://github.com/haydenflinner/dsa/blob/master/mimas/benchmarks/fib_rec/fib_rec.mim) — fib(30) recursively, 20× ([generated](https://github.com/haydenflinner/dsa/blob/master/host/eval/src/gen_fib_rec.rs)) | ~1490 ms | **~2.6 ms** | **~570×** |
| [fib_iter](https://github.com/haydenflinner/dsa/blob/master/mimas/benchmarks/fib_iter/fib_iter.mim) — 20M modular additions ([generated](https://github.com/haydenflinner/dsa/blob/master/host/eval/src/gen_fib_iter.rs)) | ~360 ms | **~67 ms** | ~5× |
| [trips_gen](https://github.com/haydenflinner/dsa/blob/master/mimas/benchmarks/trips_query/trips_gen.mim) — generate 200k taxi rows ([generated](https://github.com/haydenflinner/dsa/blob/master/host/eval/src/gen_trips_gen.rs)) | ~180 ms | **~104 ms** | ~1.7× |
| [trips_query](https://github.com/haydenflinner/dsa/blob/master/mimas/benchmarks/trips_query/trips_query.mim) — join/group/filter/sort over 200k rows ([generated](https://github.com/haydenflinner/dsa/blob/master/host/eval/src/gen_trips_query.rs)) | ~190 ms | **~101 ms** | ~1.9× |
| [meridian](https://github.com/haydenflinner/dsa/blob/master/host/web/games/meridian.mimas) — a real 1900-line game engine: rules + MCTS ([generated](https://github.com/haydenflinner/dsa/blob/master/host/eval/src/gen.rs)) | ~250 MCTS iters/s | **~4,200 iters/s** | ~17× |

Read the table as a spectrum:

- **`fib_rec` (~570×)** is the pure win — recursion is where bytecode
  dispatch hurts most (a call frame and boxed `Val`s per call), and it's
  what native code is best at. The generated code matches a hand-written
  Rust `fib` for speed.
- **`fib_iter` (~5×)** is the VM's best case already — a flat loop over
  specialized integer ops — so there's less overhead left to remove.
- **`trips_*` (~2×)** both sides bottom out in the same polars kernels and
  string formatting. The transpiler removes the mimas-side cost; it can't
  make polars itself faster.
- **meridian (~17×)** is the honest middle: real game code with
  allocation, collections, and control flow in roughly the proportions a
  program actually has.

## The interesting part: semantics

mimas has reference semantics — `let tn = arena[cur]` *aliases* into the
arena, and writes through `tn` land on the original. Rust doesn't allow
that shape, so the emitter resolves each binding by use:

- **Read-only bindings** detach with an O(1) structural clone
  (persistent collections make snapshots nearly free).
- **Write-through bindings** emit no Rust variable at all — each use site
  re-emits the place expression with its index keys snapshotted at bind
  time, so `tn.w += 1` compiles to `arena[cur].w += 1` with zero held
  borrows.
- **Mutating dict loops** iterate a snapshotted key list and write rows
  back, avoiding `iter_mut`'s borrow lasting the whole body.

## How do we know it's correct?

Differentially, against the VM. `meridian.mimas` contains a
`fuzz_run(seed, steps)` function — a deterministic LCG plays random legal
moves and returns a transcript of state signatures — so *the same source*
runs under both engines and `host/eval/examples/fuzz.rs` diffs the
transcripts. Last run: **128 seeds × 120 steps (a full game), byte-identical.**

It caught a real bug on the first run: an earlier sorted-map dict
implementation reordered a `for kv in dict` loop, which reordered a float
summation in territory scoring and leaked a 1-ulp difference into the
result. That's exactly the class of bug differential fuzzing exists for —
the fix (insertion-ordered `IndexMap`) is in `mrt.rs`.

`where` checks in the source also emit as `#[test]` functions, so the
whole rule suite runs natively under `cargo test` — all 15 pass.

## Caveats

- This is an experimental backend in the `dsa` workspace, not part of the
  published mimas compiler.
- The emitter covers the AST subset these programs exercise; exotic
  corners may not emit yet.
- `game::*` calls compile to headless stubs, so native output can't host
  the interactive canvas — it exists for servers, tests, and search.
- Regeneration is currently manual (`cargo run -p literate-eval --example
  rustgen -- in.mim out.rs`); `build.rs` integration is on the TODO list.

## Reproduce

```bash
cd host
cargo run -p literate-eval --features gen --release --example bench   # timings
cargo run -p literate-eval --features gen --release --example fuzz -- 128 120
cargo test -p literate-eval --features gen                            # where checks
```

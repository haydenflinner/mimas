<div align="center">

<img src="book/src/brand/icon-color.svg" alt="mimas" width="140" />

# mimas

**A flexible, statically typed scripting language for Rust.**

[![docs](https://img.shields.io/badge/docs-mim.as-3d8ef7?style=flat-square)](https://mim.as)
[![version](https://img.shields.io/badge/version-0.2.0-66E8FF?style=flat-square)](https://crates.io/crates/mimas)
[![built with Rust](https://img.shields.io/badge/built_with-Rust-dea584?style=flat-square&logo=rust&logoColor=white)](https://www.rust-lang.org)
[![license](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-8E9BFF?style=flat-square)](#license)
[![ci](https://img.shields.io/github/actions/workflow/status/imlazyeye/mimas/test.yml?branch=main&style=flat-square&label=ci)](https://github.com/imlazyeye/mimas/actions/workflows/test.yml)
[![tests](https://img.shields.io/badge/tests-1%2C900%2B_passing-5BD6B0?style=flat-square)](https://mim.as)

[**Get started**](https://mim.as/introduction/getting-started.html) ·
[Tour the language](https://mim.as/introduction/tour.html) ·
[Why mimas?](https://mim.as/introduction/why-mimas.html) ·
[Benchmarks](https://mim.as/introduction/benchmarks.html) ·
[Reference](https://mim.as/reference.html)

</div>

---

mimas is a statically typed, embeddable scripting language for Rust. It carries over much of Rust's syntax and ergonomics, reshaping the rest to deliver what a scripting layer is good for: fast iteration, quick compile times, and runtime flexibility -- without trading away the safety that keeps you out of the debugger.

```rust
enum Shape {
    Circle(float),
    Rect(float, float),
}

fn area(shape: Shape) -> float {
    match shape {                          // exhaustive -- miss a variant and it won't compile
        Shape::Circle(r) => r * r * std::math::PI,
        Shape::Rect(w, h) => w * h,
    }
}

let shapes = [Shape::Circle(1.0), Shape::Rect(2.0, 3.0)];
let total = 0.0;
for s in shapes { total += area(s); }
print(f"area of {shapes.len()} shapes: {total}");
```

## This fork

This tree is a permanently separate fork of upstream [mimas](https://github.com/imlazyeye/mimas) (the `origin` remote; this repo's `fork` remote is [haydenflinner/mimas](https://github.com/haydenflinner/mimas)). Upstream fixes are cherry-picked in (e.g. `00681ea`), but the changes below are deliberate, ours, and not bound for upstream. Everything listed is verified against the code; paths are under `crates/`.

### Language syntax & semantics

| Area | Ours | Upstream |
| --- | --- | --- |
| Units of measure | Quantity literals `25kW`, `0.14usd/kWh`, `90deg` + compile-time dimensional analysis (`shared/src/units.rs`, `solve/src/dims.rs`); unit names work as annotations (`fn cost(e: kWh) -> usd`, `Interval<kW>`) and `pct`/`bp`/`prob`/`logit` are dimensionless intent scales | numbers are plain `int`/`float` |
| `±` intervals | `x ± d` / `x ± d%` desugar to `Interval::within`/`Interval::pm` (`parse/src/parser.rs`, `library/src/std_lib/interval.rs`) | not present |
| `%` / `mod` | `%` is postfix percent (`50%` = `0.5`, hugs its operand); modulo is the contextual word `a mod b`; `x %=` still means `x = x mod 2` | `%` is infix modulo |
| `|>` | Pipeline infix, loosest precedence: `x |> f(a)` ≡ `f(x, a)` (`parse/src/lex/tok.rs` `PipeGreater`) | not present |
| `xor` / names | xor spells `⊕`/`⊻`/`xor`; `^` and `-` join identifiers, so `x^2` and `foo-bar` are single names (`parse/src/lex/lexer.rs`) | `^` is an operator; `-` is always minus |
| `table {}` / `query {}` | Literal dataframes and PRQL-style query blocks with `$expr` splices (`parse/src/parser.rs`, `solve/src/traits/query.rs`); the checker tracks column→type schemas (`solve/src/frames.rs`) | not present |
| `use` | `use "page-name";` is a host-resolved include (`Use::Host`, `parse/src/item/item_kinds/use.rs`) | `use` takes `a::b::{…}` paths only |
| `name_(…)` patterns | `FIELD_(X, Y)` binds `FIELD_X`, `FIELD_Y` in every pattern position (`parse/src/parser.rs`) | not present |
| `const` | Takes irrefutable patterns: `const (A, B) = (1, 2)` (`parse/src/item/item_kinds/const.rs`) | `const` binds a single name |
| Generics | `fn map<T, U>(…)`, `struct Pair<A, B>`, generic enums/impls; `Ty::Param` (`shared/src/ty.rs`) | monomorphic declarations |
| `expr?` demote | `T!` → `T?`: a raised error reads back as `null` (`parse/src/expr/expr_kinds/demote.rs`) | `?` applies to options only |
| Tests | `#[test]` / `tests`/`check` items are syntax (`#[attr]` tokens, `parse/src/item/item_kinds/tests.rs`) | not present |
| Globals | Top-level `let`s are globals any `fn` can read/write via the entry frame (`LoadEntry`/`StoreEntry`) | top-level `let` is invisible inside `fn`s |
| Docstrings | `"""…"""` in statement position is captured for tooling (`parse/src/lex/lexer.rs`) | not present |
| Invertible fns | `iso`/`lens`/`un`/`under`/`at` (Uiua-style `un`/`under`); `under`/`at` lower in codegen (`library/src/std_lib/iso.rs`, `api/src/lib.rs` `Intrinsic`) | not present |

### Host integration & embedding

| Area | Ours | Upstream |
| --- | --- | --- |
| Native metadata | `#[native]`/`#[mimas]` submit `NativeMeta` via `inventory`: doc + `param_names` + `param_dims` on `ApiFunction`/`ApiMethod` (`api/src/records.rs`, `macros/src/lib.rs`) | `doc` only |
| Units at the boundary | `#[mimas_dim("s")]` field attr and `Px`/`Secs`/`Hz` wrapper types put dims on native signatures (`macros/src/derive.rs`, `vm/src/units.rs`) | not present |
| Cooperative frames | Natives can `yield_frame`-pause the VM; host pumps via `Vm::debug_step`, caps work via `set_op_budget` (`vm/src/vm.rs`) | `Vm::run` to completion |
| Snapshots | `Vm::snapshot`/`restore` clone the whole paused heap to host memory (`vm/src/snapshot.rs`) | not present |
| Hot reload | `Vm::rebind` carries a snapshot into an edited program, matching state by name (`vm/src/rebind.rs`) | not present |
| Host → script calls | `call_fn`, `call_method_on_first_instance[_inspect]` invoke fns/methods from outside (`vm/src/vm.rs`) | `run` only |
| Inspection | `Inspect` views incl. `Inspect::Table` (DataFrame flattened for hosts) (`vm/src/val.rs`) | not present |
| Output plumbing | `print`/`warn` route through `Out`/`Prints`/`DebugInfo` fixtures to host sinks, not stdout (`library/src/prelude.rs`, `vm/src/fixtures.rs`) | `print` is `println!` |
| Test runner | `Vm::run_tests` executes `#[test]`/`check` items for the host (`vm/src/vm.rs`) | not present |
| Tooling maps | `Resolutions::node_dims`/`want_dims` expose each expr's computed/demanded dim to editors (`solve/src/resolutions.rs`) | not present |
| Data values | `DataFrame`/`PlExpr` are first-class `Val`s behind the `dataframe` feature (`vm/`, `library/src/std_lib/dataframe.rs`) | not present |

### Other divergences

| Area | Ours | Upstream |
| --- | --- | --- |
| `hash` crate | Unison-style content-addressed code: scoped alpha-renamed hashing, SCC groups, link-by-hash loader, cross-page `use "…"` edges (`crates/hash/`) | no such crate |
| Stdlib surface | Adds `std::polars` (dataframes), `std::interval`, `std::iso`, `std::darkly`, `std::debug`, parquet/xlsx I/O (`library/src/std_lib/`) | `std::{fs, math, parse, process, sys}` |
| Deps | `polars` pinned to a git fork (`haydenflinner/polars` rev `c90a33e`); `dataframe`/`darkly`/`xlsx`/`parquet` features are default-on (`library/Cargo.toml`) | no polars dep |
| Check-time literals | Literal args validated at compile time — `5kg.to("parsecs")`, bad regex in `str.find` fail the check (`solve/src/errors.rs` `BadLiteralArg`) | runtime errors |
| RNG | One seeded `Rng` stream behind every random native (`library/src/methods/rng.rs`) | unseeded per-call |
| Diagnostics | Dim mismatches name the written unit: `2px + 5ft` says `length (ft)`, not `length (m)` (`shared/src/units.rs` `describe_unit`) | n/a (no dims) |

## What you get

| | |
| --- | --- |
| 🪶 **Flexible** | Inference writes your types, compiles stay fast, and errors point toward the fix instead of just turning you away. Garbage collected -- no borrow checker, no lifetimes. |
| 🛡️ **Typed** | Static typing with inference, user-defined types, exhaustive [pattern matching](https://mim.as/reference/control-flow/match.html), and `T?` option safety so an unexpected `null` can't reach you. |
| 🧩 **Extendable** | Share Rust types and functions with the `#[mimas]` macro -- they're type-checked just like native ones. |
| ✅ **Robust** | Every panic is treated as a bug, top to bottom. Over **1,900 tests** (the tests are tested, via [cargo mutants](https://mutants.rs)), with clear diagnostics powered by [miette](https://github.com/zkat/miette). |

Read the [full tour](https://mim.as/introduction/tour.html) for a quick pass over the whole language.

## Quick start

**At the command line** -- mirroring cargo, with `check`, `build`, and `run`:

```sh
cargo install mimas-cli

mimas check my_script.mim   # parse + type-check
mimas run my_script.mim     # execute (or just `mimas my_script.mim`)
mimas run my_project        # runs the project's main.mim
mimas my_project            # running with no subcommand defaults to `run`
```

**Embedded in a Rust project** -- add `mimas` and compile a script in two lines:

```rust
const SOURCE: &str = include_str!("my_script.mim");

let mut vm = mimas::compile_source(SOURCE).unwrap();
let _ = vm.run();
```

Sharing your own Rust types is one attribute away:

```rust
#[mimas]
struct User(String);

#[mimas]
impl User {
    fn greet(self) { println!("Hello, {}!", self.0); }
}
```

```mimas
let user = User("mimas");
user.greet(); // -> Hello, mimas!
```

The full embedding guide lives at [Extension with Rust](https://mim.as/extension-with-rust.html).

## Performance

mimas compiles `.mim` source to bytecode for a register-based VM with a lifetime-safe ([gc-arena](https://github.com/kyren/gc-arena)) heap. It outpaces the other pure Rust languages in most tests and is within shooting range of Luau, a mature C++ runtime. See the [full benchmarks](https://mim.as/introduction/benchmarks.html) for the methodology and numbers.

The compiler is quick too: the whole pipeline (parse, type-check, lower, emit bytecode) runs at roughly **500,000 lines per second**, so for scripts there's effectively no compile step you'd notice.

## Examples

Runnable projects live in [`examples/`](examples):

- [`extension`](examples/extension) -- sharing Rust structs, enums, methods, and fallible functions with a script via `#[mimas]`.
- [`game-loop`](examples/game-loop) -- driving a script from a host game loop, using fixtures and `FreezeCell` to safely hand mimas a `&mut` to host state.
- [`bevy`](examples/bevy) — a breakout game on the `bevy` feature, which runs `.mim` scripts as hot-reloaded Bevy assets with typed access to reflected components, resources, and messages. See the [Bevy guide](https://mim.as/extension/bevy.html), and run it with `cargo run --manifest-path examples/bevy/Cargo.toml`.

## Editor support

A language server, `mimas-lsp`, gives any LSP-capable editor diagnostics, hover, go to definition, references, rename, outlines, and inlay hints. It only knows the standard library for now, so it doesn't yet work for scripts that use types or functions from a Rust host. See the [Language Server](https://mim.as/introduction/lsp.html) page for setup and limitations.

A VS Code extension lives in [`tools/vscode`](tools/vscode) -- syntax highlighting, snippets, and language configuration for `.mim` files, plus the language server when `mimas-lsp` is installed. Build it with `vsce package` and install the `.vsix`. The TextMate grammar it uses ([`tools/highlighter`](tools/highlighter)) is written in mimas, and is the same one that colors the docs.

## Status

mimas is early in development. It compiles, type-checks, and runs end-to-end, but nothing is promised to be stable yet. Expect sharp edges, expect things to move, and feel most welcome to [contribute](./CONTRIBUTING.md).

## How it's built

A single pipeline turns source into a running program, split across a handful of crates in [`crates/`](crates):

```
.mim -> parse -> solve (type check) -> compile (lower + bytecode) -> vm (register VM)
```

`shared` carries the common vocabulary, `api` and `macros` back the `#[mimas]` embedding surface, `library` is the standard library, and `cli` is the `mimas` binary.

## License

Dual licensed under your choice of [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE).

Built on the shoulders of [gc-arena](https://github.com/kyren/gc-arena), [miette](https://github.com/zkat/miette), and [chompy](https://github.com/imlazyeye/chompy); the `gc-arena` singleton and freeze patterns are adapted from [fabricator](https://github.com/kyren/fabricator).
</content>
</invoke>

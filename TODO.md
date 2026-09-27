# TODO

- **rustgen: widen the native-wasm engine beyond meridian** — `host/meridian-wasm`
  + `host/web/src/nativehost.ts` run games on transpiled wasm answering the
  same `gameFrameAt` contract (docs/features.md). meridian, tictactoe,
  brickbreaker, and pong (dsa 4129b3b — `use "img"` splicing needed nothing
  new, `dir_resolver` already covers seeded pages) are differential-green.
  Remaining: **asteroids** (needs real `mrt::sprite`/`mrt::tile` — currently
  no-op stubs in `host/eval/src/mrt.rs` — plus an RNG-parity check between
  `mrt::random`'s xorshift64 and the VM's `float::random`), `host::*` natives
  returning real values (stubs are inert; `choices` now emits its GameOut),
  and heap snapshots for replay seeks (native sessions refeed instead).

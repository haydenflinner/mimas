# TODO

- **rustgen: widen the native-wasm engine beyond meridian** — `host/meridian-wasm`
  + `host/web/src/nativehost.ts` now run meridian on transpiled wasm answering
  the same `gameFrameAt` contract (docs/features.md). Remaining: register more
  pages (tictactoe/brickbreaker are differential-green already; the NATIVE map
  is one line each once each has a `-gen` crate), `use "img"`-style seeded-page
  splicing for pong/asteroids, `host::*` natives, and heap snapshots for
  replay seeks (native sessions refeed instead).

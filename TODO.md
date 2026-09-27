# TODO

- **rustgen: real `game::*`/`host::*` bindings for generated code** — today the
  emitted Rust links `game::*` calls to headless stubs (`mrt`), so native
  output can't host the interactive canvas. If the eval host can ship
  generated Rust to the browser (wasm target calling the same drawing
  primitives `literate-eval::game` registers for the VM), pages could run at
  transpiled speed (~17× on meridian-class code) with no VM in the loop.
  Scope `host/web/src/evalworker.ts` first — the win is real but the plumbing
  is unknown.

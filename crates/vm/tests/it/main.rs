//! The integration tests of `mimas-vm`, as one test binary. Every test
//! binary links polars, so one binary per file cost about 210 MB each.
//! Add a new test file here as a `mod`, not as a new top-level
//! `tests/*.rs`. Run one file with `cargo test -p mimas-vm --test it --
//! trace::`.

#[macro_use]
mod vm_test_utils;

mod asi;
mod bodycache;
mod budget;
mod call_method;
mod codegen;
mod collections;
mod control_flow;
mod debug_events;
mod expressions;
mod function_dataflow;
mod functions;
mod globals;
mod graph;
mod img_inspect;
mod matching;
mod modules;
mod natives;
mod opt_semantics;
mod optionals;
mod pacts;
mod scene;
mod trace;
mod typed_arrays;
mod types;

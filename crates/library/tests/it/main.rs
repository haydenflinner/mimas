//! The integration tests of `mimas-library`, as one test binary. Every
//! test binary links polars, so one binary per file cost about 480 MB
//! each. Add a new test file here as a `mod`, not as a new top-level
//! `tests/*.rs`. Run one file with `cargo test -p mimas-library --test it
//! -- dict::`.

#[macro_use]
mod test_runner;

mod approx;
mod array;
mod assert;
mod block_check;
mod closure_expected;
mod atoms;
mod axes;
mod colrow;
mod computed;
mod const_read;
mod contracts;
mod darkly;
mod dataframe;
mod datalog;
mod debug;
mod dec;
mod derived_writes;
mod dict;
mod display;
mod doc_examples;
mod effects;
mod group;
mod impl_methods;
mod incremental;
mod inout;
mod inplace_for;
mod interval;
mod iso;
mod json;
mod juxta;
mod laws;
mod let_else;
mod lists;
mod literals;
mod macros;
mod math;
mod mem_budget;
mod multi_dispatch;
mod numbers;
mod operators;
mod pact_ops;
mod parquet;
mod place_generators;
mod positional;
mod prelude;
mod prepared;
mod propagate;
mod records;
mod reflect;
mod regions;
mod rng;
mod runtime_errors;
mod scripts;
mod short_circuit;
mod std_fs;
mod str;
mod structs;
mod subscript_fuzz;
mod subscripts;
mod symbols;
mod sys;
mod tensor;
mod units;
mod with;
mod xlsx;
mod zset;

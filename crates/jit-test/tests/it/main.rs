//! The integration tests of `mimas-jit-test`, as one test binary (each
//! binary links polars, about 234 MB). Add a new test file here as a
//! `mod`. Run one file with `cargo test -p mimas-jit-test --test it --
//! parity::`.

mod parity;
mod repro;
mod speed;

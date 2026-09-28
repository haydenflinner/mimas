//! `std::polars`'s `.parquet` file I/O: `from_parquet(path)` reads a file into a `DataFrame`,
//! `df.to_parquet(path)` writes one. Both are thin wrappers over polars' own
//! `ParquetReader`/`ParquetWriter`, so dtypes come back exactly as stored -- unlike `to_xlsx`,
//! an `i64` column stays `i64` and strings/bools/datetimes/nulls all round-trip natively.
//!
//! File paths only, like the `xlsx` module: these natives exist on native builds (the `mimas`
//! CLI's `parquet` feature), not in the wasm host, which compiles `mimas` with
//! `default-features = false`.
//!
//! There's deliberately no `compression` knob yet -- `ParquetWriter::new` defaults to zstd,
//! which is the right default for interchange; a per-call option can join the signature the
//! day a caller actually needs one.

use macros::native;
use vm::{Ctx, api::Api, conversion::Raisable};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    let mut m = api.module("std::polars");
    m.add(from_parquet);
    // also a free function so `df |> to_parquet("out.parquet")` pipes resolve, same as the verbs
    m.add(to_parquet);
    api.add_method(to_parquet);
}

/// `from_parquet("products.parquet")` reads a parquet file into a `DataFrame`. Missing files
/// and non-parquet data both raise.
#[native]
fn from_parquet<'gc>(ctx: Ctx<'gc>, path: &str) -> Raisable<vm::DataFrame<'gc>> {
    use polars_io::parquet::read::ParquetReader;
    use polars_io::prelude::SerReader;
    (|| {
        let f = std::fs::File::open(path).map_err(|e| format!("from_parquet: {path:?}: {e}"))?;
        ParquetReader::new(f)
            .finish()
            .map_err(|e| format!("from_parquet: {path:?}: {e}"))
    })()
    .map(|d| ctx.new_dataframe(d))
    .into()
}

/// `df.to_parquet("out.parquet")` writes the frame. Unwritable paths and dtypes parquet
/// can't model raise.
#[native]
fn to_parquet<'gc>(ctx: Ctx<'gc>, df: vm::DataFrame<'gc>, path: &str) -> Raisable<bool> {
    use polars_io::parquet::write::ParquetWriter;
    (|| {
        let f = std::fs::File::create(path).map_err(|e| format!("to_parquet: {path:?}: {e}"))?;
        let mut frame = df.0.borrow_mut(ctx.mutation());
        ParquetWriter::new(f)
            .finish(&mut frame.0)
            .map_err(|e| format!("to_parquet: {path:?}: {e}"))?;
        Ok::<_, String>(true)
    })()
    .into()
}

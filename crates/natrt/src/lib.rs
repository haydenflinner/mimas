//! The companion runtime a separately-compiled mimas object links against:
//! every `mj_*` helper is `vm::bc::jit`'s extern-C shim re-exported under
//! `#[unsafe(no_mangle)]` (`vm::bc::nat`). Build as `staticlib` for a
//! self-contained artifact link, or `cdylib` for a shared runtime.
pub use vm::bc::nat::*;

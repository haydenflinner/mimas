mod utils;
mod vm;

pub use utils::*;
pub use vm::*;

mod errors;
mod val;
pub use errors::*;
pub use val::*;
pub mod adt;
pub mod anon;
pub mod api;
pub mod conversion;
pub mod freeze;
mod heap;
pub use heap::*;
mod native;
pub use native::*;
pub mod fixtures;
pub use fixtures::*;
mod math;
pub use ::glam;

// re-exports the MimasEnum / MimasStruct derives resolve against -- saves user crates from
// depending on mimas-api / mimas-shared directly.
pub use ::api::{AdtBinding, ApiAdtKind, ApiVariantFields, Registry};
pub use ::compile::{Constant, Function, Module, Type, Vis};
pub use ::macros::{MimasEnum, MimasStruct, mimas, native};
pub use ::shared::{BodyId, FnHeader, Literal, Ty};
pub use conversion::{Arg, Args};

// the `#[mimas]` attribute macro expands to `vm::inventory::submit!{ ... }`, so the inventory
// crate has to be reachable through `vm`. re-exporting it here means downstream crates that
// author natives only need a `vm` dependency, never a direct `inventory` one.
pub use ::inventory;

// `Vm::function_dataflow`'s return type -- re-exported so a `vm`-only dependent (the inspector,
// say) can name `vm::DataflowGraph` etc. without also depending on `mimas-compile` directly.
// `BinOp`/`UnaryOp` join them: `Api::add_bin_op`/`add_unary_op` take them as args.
pub use ::compile::{
    BinOp, DataflowEdge, DataflowError, DataflowGraph, DataflowNode, NodeKind, UnaryOp,
};

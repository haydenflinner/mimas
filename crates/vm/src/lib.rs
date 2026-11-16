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
pub mod bc;
pub mod conversion;
pub mod freeze;
mod heap;
pub use heap::*;
pub mod mem;
mod native;
pub use native::*;
pub mod fixtures;
pub use fixtures::*;
mod snapshot;
pub use snapshot::*;
mod rebind;
pub use rebind::*;
mod graph;
pub use graph::*;
mod math;
#[cfg(feature = "tensor")]
pub mod tensor;
pub mod units;
pub use ::glam;
pub use units::{
    Beats, Bits, Hz, Len, Px, PxDown, PxRight, PxS, PxS2, PxUp, PxX, PxY, Rad, Secs, Spx, SpxDown,
    SpxRight, SpxUp, SpxX, SpxY, St, Tempo, Wpx, WpxDown, WpxRight, WpxUp, WpxX, WpxY,
};

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
    BinOp, DataflowEdge, DataflowError, DataflowGraph, DataflowNode, NodeKind, Program, UnaryOp,
};

/// `is_instance`: an instance of the adt, or — for the primitive sentinels
/// (`AdtId::PRIM_INT` and friends) — a value of that primitive kind, which
/// is how `impl Num for float` takes its arm in a pact dispatch chain.
/// r[impl mimas.pact.primitive]
pub fn val_is_instance(v: &Val<'_>, adt: u32) -> bool {
    use shared::{AdtId, Id};
    let id = AdtId::new(adt);
    match v {
        _ if id == AdtId::PRIM_INT => matches!(v, Val::Int(_)),
        _ if id == AdtId::PRIM_FLOAT => matches!(v, Val::Float(_)),
        _ if id == AdtId::PRIM_STR => matches!(v, Val::Str(_)),
        _ if id == AdtId::PRIM_BOOL => matches!(v, Val::Bool(_)),
        Val::Instance(i) => i.0.borrow().struct_id == adt,
        _ => false,
    }
}

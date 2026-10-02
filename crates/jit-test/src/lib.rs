//! Support crate for the mimas-jit parity tests. Unlike bcgen-test there is
//! no build-time emission — `mimas_jit::compile` runs at test time on the
//! same `Program` the VM will execute.

pub mod natives {
    include!("../natives.rs");
}

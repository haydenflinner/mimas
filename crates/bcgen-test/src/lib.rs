//! Support crate for the bcgen parity tests. `generated::<fixture>` modules are
//! emitted by `build.rs` from `fixtures/*.mimas`; `natives` is the shared
//! native table (see `natives.rs`).

pub mod natives {
    include!("../natives.rs");
}

pub mod generated {
    pub mod basic {
        include!(concat!(env!("OUT_DIR"), "/basic.rs"));
    }
    pub mod deep {
        include!(concat!(env!("OUT_DIR"), "/deep.rs"));
    }
    pub mod cold {
        include!(concat!(env!("OUT_DIR"), "/cold.rs"));
    }
    pub mod mixed {
        include!(concat!(env!("OUT_DIR"), "/mixed.rs"));
    }
}

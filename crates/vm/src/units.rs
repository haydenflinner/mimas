//! Unit-carrying scalar wrappers for native signatures.
//!
//! A native whose Rust parameter is one of these types still receives an ordinary `float` at
//! runtime -- units are compile-time only. What changes is that the API record carries the
//! slot's [`Dim`], so the solver's dimension pass can check call sites: `play_for(.., 1200Hz,
//! 0.5s)` type-checks, `play_for(.., 0.5s, 1200Hz)` is a compile error. Declare a wrapper where
//! the unit is part of the contract; keep `f64` where it isn't.

use api::Registry;
use shared::units::{self, Dim};

use crate::{
    Ctx, Ty, Val,
    conversion::{MimasType, TypeError, ty_error},
};

macro_rules! unit_ty {
    ($(#[$m:meta])* $name:ident, $unit:literal, $dim:expr) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
        pub struct $name(pub f64);

        impl $name {
            pub const DIM: Dim = $dim;
        }

        impl<'gc> MimasType<'gc> for $name {
            fn mimas_ty(_: &Registry) -> Option<Ty> {
                Some(Ty::Float)
            }
            fn mimas_dim(_: &Registry) -> Option<Dim> {
                Some(Self::DIM)
            }
            fn from_value(_ctx: Ctx<'gc>, v: Val<'gc>) -> Result<Self, TypeError> {
                v.as_float()
                    .map(Self)
                    .ok_or_else(|| ty_error($unit, v))
            }
            fn into_value(self, _ctx: Ctx<'gc>) -> Val<'gc> {
                Val::Float(self.0)
            }
        }

        impl From<f64> for $name {
            fn from(v: f64) -> Self {
                Self(v)
            }
        }
        impl From<$name> for f64 {
            fn from(v: $name) -> f64 {
                v.0
            }
        }
    };
}

unit_ty!(
    /// Seconds.
    Secs, "s", units::TIME
);
unit_ty!(
    /// Hertz.
    Hz, "Hz", units::FREQ
);
unit_ty!(
    /// Beats (`meas`/`phrase` are the same dimension, scaled).
    Beats, "beat", units::BEAT
);
unit_ty!(
    /// Semitones (`oct` too).
    St, "st", units::PITCH
);
unit_ty!(
    /// Beats per second -- what `bpm`/`cpm` literals reduce to.
    Tempo, "bpm", units::BEAT.div(units::TIME)
);
unit_ty!(
    /// Bits of data.
    Bits, "b", units::DATA
);

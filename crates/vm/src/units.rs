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

/// Resolve a unit name to its dimension — backs `#[mimas_dim("s")]` on `MimasStruct`/
/// `MimasEnum` fields, so a plain `f64` field can declare its unit for the checker.
/// `#[mimas_dim("px")]`, `#[mimas_dim("px/s")]`, `#[mimas_dim("px↕")]` -- any unit
/// expression, axis included, so a host field can say which way it points.
pub fn dim_of(name: &str) -> Option<Dim> {
    units::parse(name).map(|(d, _)| d)
}

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
unit_ty!(
    /// Pixels -- screen-space positions and extents.
    Px, "px", units::PIXELS
);
// r[impl mimas.unit.nested]
unit_ty!(
    /// Pixels along x (`px↔`) -- a horizontal screen coordinate. Untagged `px`
    /// arguments adopt the axis at the call site; a `px↕` argument does not.
    PxX, "px↔", units::PIXELS.mul(Dim::axis(0))
);
unit_ty!(
    /// Pixels along y (`px↕`) -- a vertical screen coordinate, down-positive
    /// like the axis it rides.
    PxY, "px↕", units::PIXELS.mul(Dim::axis(1))
);
unit_ty!(
    /// Screen pixels -- coordinates in a space the host owns (browser
    /// points, viewport extents, sheet-plane positions), as delivered on
    /// input fixtures. `Spx` values can't meet `Px` ones in arithmetic or
    /// typed slots: the page crosses with `game::to_world`/`to_screen`.
    Spx, "spx", units::SCREEN_PX
);
unit_ty!(
    /// Screen pixels along x (`spx↔`).
    SpxX, "spx↔", units::SCREEN_PX.mul(Dim::axis(0))
);
unit_ty!(
    /// Screen pixels along y (`spx↕`).
    SpxY, "spx↕", units::SCREEN_PX.mul(Dim::axis(1))
);
unit_ty!(
    /// World pixels -- coordinates in the page's own world (the field,
    /// board or chart plane the camera looks at), as `game::to_world`
    /// returns them. `Wpx` values meet `Px` ones — `px` is the pixel
    /// family's unspecified member — but never `Spx` ones: the seam
    /// natives own that crossing.
    Wpx, "wpx", units::WORLD_PX
);
unit_ty!(
    /// World pixels along x (`wpx↔`).
    WpxX, "wpx↔", units::WORLD_PX.mul(Dim::axis(0))
);
unit_ty!(
    /// World pixels along y (`wpx↕`).
    WpxY, "wpx↕", units::WORLD_PX.mul(Dim::axis(1))
);
unit_ty!(
    /// Screen pixels, right-positive (`spx→`) — a horizontal position
    /// or delta in the host's space.
    SpxRight, "spx→", units::SCREEN_PX.mul(Dim::axis_dir(0, false))
);
unit_ty!(
    /// Screen pixels, up-positive (`spx↑`) — the y the host hands page
    /// code today (input arrives already flipped into game space).
    SpxUp, "spx↑", units::SCREEN_PX.mul(Dim::axis_dir(1, false))
);
unit_ty!(
    /// Screen pixels, down-positive (`spx↓`) — raw CSS-style y.
    SpxDown, "spx↓", units::SCREEN_PX.mul(Dim::axis_dir(1, true))
);
unit_ty!(
    /// World pixels, right-positive (`wpx→`).
    WpxRight, "wpx→", units::WORLD_PX.mul(Dim::axis_dir(0, false))
);
unit_ty!(
    /// World pixels, up-positive (`wpx↑`).
    WpxUp, "wpx↑", units::WORLD_PX.mul(Dim::axis_dir(1, false))
);
unit_ty!(
    /// World pixels, down-positive (`wpx↓`) — a world laid out the
    /// sheet's way.
    WpxDown, "wpx↓", units::WORLD_PX.mul(Dim::axis_dir(1, true))
);
// r[impl mimas.unit.angle]
unit_ty!(
    /// Radians -- `deg`/`turn`/`grad`/`arcmin`/`arcsec`/`mrad` are the same
    /// dimension, scaled. An angle-taking native declares this and the
    /// checker refuses plain numbers at the call site.
    Rad, "rad", units::ANGLE
);
// r[impl mimas.unit.length]
unit_ty!(
    /// Meters -- `mm`/`cm`/`inch`/`ft` are the same dimension, scaled.
    /// The `use cad` kernel's lengths: `cad::box(40mm, 30mm, 6mm)` type-checks,
    /// a bare `40` does not.
    Len, "m", units::LENGTH
);
unit_ty!(
    /// Pixels per second -- screen-space velocities.
    PxS, "px/s", units::PIXELS.div(units::TIME)
);
unit_ty!(
    /// Pixels per second squared -- screen-space accelerations.
    PxS2, "px/s^2", units::PIXELS.div(units::TIME.powi(2))
);

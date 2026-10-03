//! Effect sets: what a mimas function may do beyond returning a value.
//!
//! `Fx` is a set over `{doc, net, rng, yield, io, time}` — the side-effect vocabulary a
//! host can gate. The solver infers a function's set as the union of its callees; a
//! `#[effects(...)]` attribute on a native declares its footprint. The [`Fx::UNAUDITED`]
//! flag is orthogonal to the six: it means "a call whose effects nobody declared lives in
//! this set," so the named effects are a floor, not a ceiling — a function can carry every
//! flag *and* still be unaudited.
//!
//! Gating reads straightforwardly: `page_fn.fits(Fx::RNG)` is true only when the function's
//! whole footprint is randomness or less — an unaudited call inside it fails closed.

use bitflags::bitflags;

bitflags! {
    /// A function's side-effect footprint — see the module docs.
    pub struct Fx: u8 {
        /// Reads or writes the host document (cells, page state).
        const DOC = 1 << 0;
        /// Network access of any kind (fetch, sync, peer calls).
        const NET = 1 << 1;
        /// Draws from a randomness source (`random`, `shuffle`, ...).
        const RNG = 1 << 2;
        /// Can suspend the frame (`yield_frame` and anything reaching it).
        const YIELD = 1 << 3;
        /// Talks to the outside world that isn't the document or the network
        /// (`print`, files, wall-clock-independent output).
        const IO = 1 << 4;
        /// Reads clocks or timers (`now`, `elapsed`).
        const TIME = 1 << 5;
        /// Contains a call whose effects were never declared — the rest of the set is a
        /// lower bound, not a guarantee. Always set on top of [`Fx::ANY`] for an
        /// unannotated native.
        const UNAUDITED = 1 << 7;
    }
}

impl Fx {
    /// Every effect a host could name — no `UNAUDITED`, that's a separate question.
    pub const ANY: Fx = Fx::from_bits_truncate(
        Fx::DOC.bits()
            | Fx::NET.bits()
            | Fx::RNG.bits()
            | Fx::YIELD.bits()
            | Fx::IO.bits()
            | Fx::TIME.bits(),
    );

    /// The set for a call nobody annotated: it might do anything, and the audit is
    /// incomplete. Fails every gate except `fits(Fx::ANY | Fx::UNAUDITED)`.
    pub const fn unknown() -> Fx {
        Fx::ANY.union(Fx::UNAUDITED)
    }

    /// Provably free of every tracked effect.
    pub const fn pure() -> Fx {
        Fx::empty()
    }

    /// `is_empty` reads awkwardly at call sites — `fx.is_pure()` is the question being asked.
    pub const fn is_pure(&self) -> bool {
        self.is_empty()
    }

    /// Every effect in `self` is covered by `allowed`, and nothing unaudited slipped in.
    /// `UNAUDITED` bits only fit when the gate itself allows unaudited code.
    pub const fn fits(&self, allowed: Fx) -> bool {
        self.difference(allowed).is_empty()
    }

    /// The declared subset with the audit flag dropped — for display and comparisons that
    /// only care about the named footprint.
    pub const fn declared(&self) -> Fx {
        self.intersection(Fx::ANY)
    }
}

/// `{doc, rng}` for a declared set; a trailing `…` marks `UNAUDITED` ("and possibly more").
/// The pure set prints as `{}`.
impl std::fmt::Display for Fx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names = [
            (Fx::DOC, "doc"),
            (Fx::NET, "net"),
            (Fx::RNG, "rng"),
            (Fx::YIELD, "yield"),
            (Fx::IO, "io"),
            (Fx::TIME, "time"),
        ]
        .into_iter()
        .filter(|(bit, _)| self.contains(*bit))
        .map(|(_, name)| name)
        .collect::<Vec<_>>();
        let mut out = format!("{{{}}}", names.join(", "));
        if self.contains(Fx::UNAUDITED) {
            out.insert_str(out.len() - 1, ", …");
        }
        f.pad(&out)
    }
}

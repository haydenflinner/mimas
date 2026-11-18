//! The parity-harness native table — `host_add`, `keep`, `pause`,
//! `print` on top of `library::std` — shared by every lane that compiles
//! or runs generated bodies (`tools/pliron-mimas`'s driver, the
//! `mimas-plwasm` shim). Native ids are registration-order, so the
//! compile probe and the run Vm must see identical installs; keeping the
//! list in one crate is what guarantees that.

/// The fixture install — mirrored lanes must call this, not a local
/// copy, or every `NativeId` baked into generated code indexes a
/// shifted table (silent wrong-native dispatch).
pub fn install(api: &mut vm::api::Api<'_, '_>) {
    // `std` (superset of `codegen`, the lowering subset): the workload
    // sources use real prelude/method natives (`print`, `to_float`,
    // `str.len`, ...), and `std` additionally registers the `use std::*`
    // source modules.
    library::std(api);
    api.add_named("host_add", host_add);
    api.add_named("keep", keep);
    api.add_named("pause", pause);
    api.add_named("print", print);
}

#[vm::native]
fn host_add<'gc>(_ctx: vm::Ctx<'gc>, a: i64, b: i64) -> i64 {
    a + b
}

fn keep<'gc>(ctx: vm::Ctx<'gc>, v: vm::Val<'gc>) {
    ctx.fixture::<Kept>().0.borrow_mut().push(v.capture());
}

/// `yield_frame`-style native: latches the cooperative-pause flag.
fn pause<'gc>(ctx: vm::Ctx<'gc>) {
    ctx.state().paused.set(true);
}

/// The workload sources (`vm_*.mim`, `page_*.mimas`) print instead of
/// `keep`ing — capture the line into the same log the parity check diffs,
/// so a native-side and interp-side print compare equal only if the text
/// does too.
fn print<'gc>(ctx: vm::Ctx<'gc>, v: vm::Val<'gc>) -> Result<(), vm::RtErr> {
    let line = ctx.to_string(v)?;
    ctx.fixture::<Kept>()
        .0
        .borrow_mut()
        .push(vm::Captured::Str(line));
    Ok(())
}

/// Values `keep` was called with, captured at native-call time so runs can
/// compare across separate Vms.
#[derive(Default)]
pub struct Kept(pub std::cell::RefCell<Vec<vm::Captured>>);

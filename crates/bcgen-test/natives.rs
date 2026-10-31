// Included by both `build.rs` (where it supplies the native table
// `Vm::compile_parts` resolves `CallNative` ids against) and `src/lib.rs`
// (where tests install the same natives into the run Vm). Native ids are
// registration-order, so both sites must run identical installs — keeping the
// definitions in one `include!`d file is what guarantees that.

pub fn install(api: &mut vm::api::Api<'_, '_>) {
    api.add_named("host_add", host_add);
    api.add_named("keep", keep);
}

#[vm::native]
fn host_add<'gc>(_ctx: vm::Ctx<'gc>, a: i64, b: i64) -> i64 {
    a + b
}

fn keep<'gc>(ctx: vm::Ctx<'gc>, v: vm::Val<'gc>) {
    ctx.fixture::<Kept>().0.borrow_mut().push(v.capture());
}

/// Values `keep` was called with, captured to owned snapshots at native-call
/// time so tests can compare across two separate VM runs.
#[derive(Default)]
pub struct Kept(pub std::cell::RefCell<Vec<vm::Captured>>);

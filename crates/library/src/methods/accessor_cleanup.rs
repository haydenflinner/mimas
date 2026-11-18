use macros::native;
use vm::{Ctx, RtErr, anon, api::Api};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add(__cleanup_register);
    api.add(__cleanup_mark);
    api.add(__cleanup_take);
    api.add(__place_live);
    api.add(__seq_fresh);
    api.add(__cleanup_promote);
}

// r[impl mimas.sub.end]
#[native]
#[effects()]
fn __cleanup_register<'gc>(ctx: Ctx<'gc>, action: anon::Fn0<'gc, ()>) {
    ctx.defer(action.0);
}

#[native]
#[effects()]
fn __cleanup_mark(ctx: Ctx<'_>) -> i64 {
    ctx.cleanup_mark() as i64
}

#[native]
#[effects()]
fn __cleanup_take<'gc>(ctx: Ctx<'gc>, mark: i64) -> Option<anon::Fn0<'gc, ()>> {
    ctx.take_cleanup(mark.max(0) as usize)
        .map(|v| anon::Fn0(v, std::marker::PhantomData))
}

#[native]
#[effects()]
fn __place_live(lifetime: &[bool]) -> Result<(), RtErr> {
    if lifetime.first() == Some(&true) {
        Ok(())
    } else {
        Err(RtErr::Custom(
            "the borrowed accessor place has left its scope".into(),
        ))
    }
}

/// A kept `&xs[i]` place's staleness stamp: `cap` is the container the
/// descriptor's closures captured at the lend, `cur` is the container
/// re-read at the use, `gen` the length at the lend. Different object
/// (`s.items = packed` under `&s.items[i]`) or a different length
/// (`xs.push` since the lend) faults the use.
// r[impl mimas.place.stale]
#[native]
#[effects()]
fn __seq_fresh<'gc>(cap: vm::Val<'gc>, gen0: vm::Val<'gc>) -> Result<(), RtErr> {
    use vm::Val;
    // a Null stamp means the captured length never landed -- nothing to
    // check against, so the use is fresh by default
    let lent_len = match gen0 {
        Val::Int(n) => n.max(0) as usize,
        _ => return Ok(()),
    };
    // only the *captured* container's shape matters: a rebound binding
    // detaches the place by spec (mimas.place.let), so a swapped-in array
    // is not staleness -- a grown or shrunk one is.
    let len = match &cap {
        Val::Array(a) => a.0.borrow().len(),
        Val::IntArray(a) | Val::FloatArray(a) => a.0.borrow().len(),
        _ => return Ok(()),
    };
    let ok = len == lent_len;
    if ok {
        Ok(())
    } else {
        Err(RtErr::Custom(
            "this place's collection changed since it was lent -- the slot it names moved or went away"
                .into(),
        ))
    }
}

#[native]
#[effects()]
fn __cleanup_promote(ctx: Ctx<'_>) {
    ctx.promote_cleanups();
}

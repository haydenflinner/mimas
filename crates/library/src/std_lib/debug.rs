//! `std::debug` -- reflection natives for trace/debug tooling. The interesting one is
//! `caller_line`: it maps the live call stack back to source, so a script-side helper
//! (`fn emit(...) { dbg::caller_line(1) }`) can tag a structured event with the line that
//! produced it -- no manual line numbers, no duplicated source constants.

use macros::native;
use shared::FileId;
use vm::{Ctx, api::Api, fixtures::DebugInfo};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    let mut m = api.module("std::debug");
    m.add(caller_line);
    m.add(srcfile);
}

/// The `(file, byte offset)` of the call site `skip` frames above the frame that invoked the
/// native. The dispatch loop holds `thread` mutably borrowed, so this reads the `DebugInfo`
/// stack mirror the `CallNative` arm refreshes instead: the top entry's ip sits just past
/// this call op, and each suspended parent's ip is its frozen return-site save, so one byte
/// back lands inside the call op itself and the loc table resolves to that op's source span.
fn caller_loc(ctx: Ctx<'_>, skip: i64) -> Option<(FileId, usize)> {
    let skip = usize::try_from(skip).ok()?;
    let dbg = ctx.fixture::<DebugInfo>();
    let stack = dbg.stack.borrow();
    let (chunk, ip) = *stack.get(stack.len().checked_sub(1 + skip)?)?;
    let rel = ip.checked_sub(*dbg.offsets.borrow().get(chunk)?)?;
    let table = dbg.locs.borrow();
    let locs = table.get(chunk)?;
    let i = locs.partition_point(|(off, _)| *off < rel);
    if i == 0 {
        return None;
    }
    let loc = locs[i - 1].1;
    Some((loc.file_id, loc.span.start as usize))
}

/// The 1-based source line of the call site `skip` frames up: `dbg::caller_line(0)` inside a
/// plain fn is that fn's own call of the native; a wrapper like `emit` passes 1 to report the
/// line that called *it*. Null when nothing is loaded, the frame doesn't exist, or the Vm has
/// no source text to count lines against.
#[native]
fn caller_line<'gc>(ctx: Ctx<'gc>, skip: i64) -> Option<i64> {
    let (file_id, offset) = caller_loc(ctx, skip)?;
    let sources = ctx.fixture::<DebugInfo>().sources.borrow();
    let text = sources.get(&file_id)?;
    let upto = offset.min(text.len());
    Some(text[..upto].bytes().filter(|&b| b == b'\n').count() as i64 + 1)
}

/// The whole source text of the file the call site `skip` frames up lives in -- a debugger
/// page grabs it once at init and slices its own context window around `caller_line`'s hits.
#[native]
fn srcfile<'gc>(ctx: Ctx<'gc>, skip: i64) -> Option<String> {
    let (file_id, _) = caller_loc(ctx, skip)?;
    ctx.fixture::<DebugInfo>()
        .sources
        .borrow()
        .get(&file_id)
        .map(|text| text.to_string())
}

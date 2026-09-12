use crate::{Ctx, RtResult, Val};

pub trait VmDisplay<'gc> {
    fn vm_display(&self, ctx: Ctx<'gc>) -> RtResult<String>;
}

impl<'gc> VmDisplay<'gc> for Val<'gc> {
    fn vm_display(&self, ctx: Ctx<'gc>) -> RtResult<String> {
        ctx.display(*self)
    }
}

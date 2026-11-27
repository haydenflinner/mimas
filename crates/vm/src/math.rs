use std::any::TypeId;

use glam::{DQuat, DVec2, DVec3};

use crate::{
    ApiAdtKind, ApiVariantFields, Ctx, Fields, Registry, Ty, Val,
    adt::{ApiAdtDescriptor, ApiVariantShape, MimasAdt},
    conversion::{MimasType, TypeError},
};

// r[impl mimas.math.vec2-f64] the std vectors hold `f64`, the width of a
// Mimas `float`, so a value round-trips through them exactly
macro_rules! glam_struct {
    ($ty:ident as $name:literal, $new:path, $($field:ident),+) => {
        impl MimasAdt for $ty {
            fn descriptor(_: &Registry) -> ApiAdtDescriptor {
                ApiAdtDescriptor {
                    name: $name,
                    module: &[],
                    kind: ApiAdtKind::Struct,
                    doc: "",
                    variants: vec![ApiVariantShape {
                        name: "@".to_string(),
                        doc: "",
                        fields: ApiVariantFields::Named(vec![
                            $((stringify!($field).to_string(), Ty::Float, None)),+
                        ]),
                    }],
                }
            }
        }

        impl<'gc> MimasType<'gc> for $ty {
            fn mimas_ty(reg: &Registry) -> Option<Ty> {
                Some(
                    reg.ty_of_id(TypeId::of::<Self>())
                        .expect(concat!("`", stringify!($ty), "` isn't registered with mimas")),
                )
            }

            fn from_value(ctx: Ctx<'gc>, value: Val<'gc>) -> Result<Self, TypeError> {
                let mismatch = || TypeError {
                    expected: $name.into(),
                    got: format!("{value:?}"),
                };
                let Val::Instance(instance) = value else {
                    return Err(mismatch());
                };
                let id = ctx.binding(TypeId::of::<Self>()).map(|b| b.adt_id);
                let fields = {
                    let instance = instance.0.borrow();
                    if id.is_none_or(|id| id.index() as u32 != instance.struct_id) {
                        return Err(mismatch());
                    }
                    instance.fields.clone()
                };
                let mut fields = fields.into_iter();
                Ok($new($(
                    <f64 as MimasType>::from_value(ctx, fields.next().ok_or_else(|| {
                        TypeError { expected: stringify!($field).into(), got: "<empty>".into() }
                    })?)?
                ),+))
            }

            fn into_value(self, ctx: Ctx<'gc>) -> Val<'gc> {
                let id = ctx
                    .binding(TypeId::of::<Self>())
                    .expect(concat!("`", stringify!($ty), "` isn't registered with mimas"))
                    .adt_id;
                let fields = vec![$(Val::Float(self.$field)),+];
                Val::Instance(ctx.new_instance(id.index() as u32, Fields::new(fields)))
            }
        }
    };
}

glam_struct!(DVec2 as "Vec2", DVec2::new, x, y);
glam_struct!(DVec3 as "Vec3", DVec3::new, x, y, z);
glam_struct!(DQuat as "Quat", DQuat::from_xyzw, x, y, z, w);

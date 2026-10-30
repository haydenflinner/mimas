//! Confirms the exact `Inspect` shape a scene renderer will see for an `img`-style enum: which
//! field carries the variant name, and how a struct-style vs. tuple-style variant's fields come
//! back labeled. This is load-bearing for `dataflow_view`-style rendering code that will match
//! on `Inspect::Instance { type_name, .. }` by string -- if this shape ever changes, that
//! matching code silently stops firing, so it's worth pinning down explicitly.

use vm::{Inspect, Vm};

const IMG_SOURCE: &str = r#"
enum Color {
    Named(str),
}

enum Image {
    Circle { radius: float, color: Color },
    Overlay { top: Image, bottom: Image },
}

pact Draw {
    fn draw(self) -> Image;
}

struct Scene { }

impl Draw for Scene {
    fn draw(self) -> Image {
        Image::Overlay {
            top = Image::Circle { radius = 5.0, color = Color::Named("red") },
            bottom = Image::Circle { radius = 3.0, color = Color::Named("blue") },
        }
    }
}

let s = Scene { };
"#;

#[test]
fn struct_variant_reports_variant_name_and_field_names() {
    let mut vm = Vm::compile(IMG_SOURCE, |_| {}).expect("should compile");
    vm.run().expect("should run to completion");

    let result = vm
        .call_method_on_first_instance_inspect("draw")
        .expect("should find the Scene instance and call draw()");

    let Inspect::Instance { type_name, fields } = result else {
        panic!("expected an Instance, got {result:?}");
    };
    // variant names come back qualified as "EnumName::Variant", not just "Variant" -- matches
    // the qualification `Image::Circle { .. }` construction/pattern syntax requires.
    assert_eq!(type_name, "Image::Overlay");
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].0, "top");
    assert_eq!(fields[1].0, "bottom");

    let Inspect::Instance {
        type_name: top_name,
        fields: top_fields,
    } = &fields[0].1
    else {
        panic!("expected `top` to be an Instance, got {:?}", fields[0].1);
    };
    assert_eq!(top_name, "Image::Circle");
    assert_eq!(top_fields[0].0, "radius");
    assert!(matches!(top_fields[0].1, Inspect::Float(f) if f == 5.0));
    assert_eq!(top_fields[1].0, "color");

    // a tuple-style variant (`Named(str)`, not `Named { .. }`) still reports its variant name,
    // with its single positional field labeled "0" -- confirms the numeric fallback in
    // `Val::inspect` kicks in here rather than leaving the field unlabeled.
    let Inspect::Instance {
        type_name: color_name,
        fields: color_fields,
    } = &top_fields[1].1
    else {
        panic!(
            "expected `color` to be an Instance, got {:?}",
            top_fields[1].1
        );
    };
    assert_eq!(color_name, "Color::Named");
    assert_eq!(color_fields[0].0, "0");
    assert!(matches!(&color_fields[0].1, Inspect::Str(s) if s == "red"));
}

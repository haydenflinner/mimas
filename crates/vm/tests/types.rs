#[macro_use]
mod vm_test_utils;

use vm::Captured::*;

test_vm!(
    instance,
    "struct Foo {}",
    "Foo {}" => instance!(Foo {})
);

test_vm!(
    instance_with_fields,
    "struct Foo { a: int, b: int }",
    "Foo { a = 0, b = 1 }" => instance!(Foo { a = Int(0), b = Int(1) })
);

test_vm!(
    get_instance_field,
    "struct Foo { x: int }
    let foo = Foo { x = 42 };",
    "foo.x" => Int(42)
);

test_vm!(
    set_instance_field,
    "struct Foo { x: int }
    let foo = Foo { x = 42 };
    foo.x = 0;",
    "foo.x" => Int(0)
);

test_vm!(
    instance_option_field,
    "struct Foo { a: int }
    let foo: Foo? = null;",
    "foo?.a" => Null,
);

// a tuple-struct constructor is a first-class value -- bind it, then call it
test_vm!(
    tuple_struct_ctor_as_value,
    "struct Wrap(int);
     let make = Wrap;
     let w = make(7);",
    "if let Wrap(x) = w { x } else { -1 }" => Int(7),
);

// `print`/`display`/f-strings render an instance as `Name { .. }`, not the bare `@id { .. }`
// the runtime used to fall back to before struct/variant names were threaded through to the VM.
test_vm!(
    instance_display_uses_struct_name,
    "struct Node { next: Node?, prev: Node?, val: int }
     let n = Node { next = null, prev = null, val = 3 };",
    r#"f"{n}""# => str!("Node { null, null, 3 }"),
);

test_vm!(
    tuple_struct_display_uses_struct_name,
    "struct Wrap(int);
     let w = Wrap(9);",
    r#"f"{w}""# => str!("Wrap { 9 }"),
);

// enum variant instances display qualified as `Enum::Variant { .. }`
test_vm!(
    enum_variant_display_uses_qualified_name,
    "enum Shape { Circle { radius: int }, Square(int) }",
    r#"f"{Shape::Circle { radius = 5 }}""# => str!("Shape::Circle { 5 }"),
    r#"f"{Shape::Square(4)}""# => str!("Shape::Square { 4 }"),
);

// a self-referential instance (`a.next = b; b.prev = a;`) used to blow the native stack --
// `render_into` recurses through instance fields with no cycle detection, so `print`/`display`
// would recurse forever. It must now surface as a normal, located runtime error instead of
// crashing the process.
test_fail!(
    displaying_a_reference_cycle_errors_instead_of_crashing,
    "struct Node { next: Node?, prev: Node?, val: int }
     let a = Node { next = null, prev = null, val = 1 };
     let b = Node { next = null, prev = a, val = 2 };
     a.next = b;
     print(a);"
);

// `S::default()` -- the implicit memberwise-default ctor every struct carries:
// numbers 0, strs "", bools false, collections empty, options null, nested
// structs defaulted recursively. an `impl` member named `default` shadows it.
test_vm!(
    default_ctor_all_member_kinds,
    "struct Foo { a: int, f: float, s: str, ok: bool, xs: [int], m: ~{int}, t: (int, str), maybe: int? }",
    "Foo::default()" => instance!(Foo {
        a = Int(0), f = Float(0.0), s = str!(""), ok = Bool(false),
        xs = array!(), m = dict!(~{}), t = array!(Int(0), str!("")), maybe = Null,
    }),
);

test_vm!(
    default_ctor_nested_struct,
    "struct In { n: int }
     struct Out { i: In, tag: str }",
    "Out::default()" => instance!(Out { i = instance!(In { n = Int(0) }), tag = str!("") }),
);

test_vm!(
    default_ctor_tuple_struct,
    "struct W(int, str)",
    "W::default()" => instance!(W { x = Int(0), y = str!("") }),
);

test_vm!(
    default_ctor_impl_member_shadows,
    "struct Foo { a: int }
     impl Foo { fn default() -> Foo { Foo { a = 7 } } }",
    "Foo::default().a" => Int(7),
);

// an enum can't pick its own variant; a generic member has no value to default
// to; and the ctor takes no arguments.
test_fail!(default_ctor_enum, "enum E { A, B } let e = E::default();");
test_fail!(default_ctor_generic_member, "struct P<T> { x: T } let p = P::default();");
test_fail!(default_ctor_with_args, "struct S { a: int } let s = S::default(1);");

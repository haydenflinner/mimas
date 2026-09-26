use crate::components::Ty::*;

// -- generic functions: every use gets fresh type variables ----------------------

test_ty!(
    generic_fn_single_use,
    "fn id<T>(x: T) -> T { x }",
    "id(1)" => Int,
);

test_ty!(
    generic_fn_instantiates_fresh_per_use,
    "fn id<T>(x: T) -> T { x }",
    "id(1)" => Int,
    "id(\"s\")" => Str,
    "id(true)" => Bool,
);

test_ty!(
    generic_fn_two_params,
    "fn pick<T, U>(a: T, b: U) -> T { a }",
    "pick(1, \"s\")" => Int,
    "pick(\"s\", 1)" => Str,
);

test_ty!(
    generic_fn_fn_param,
    "fn apply<T, U>(x: T, f: (T) -> U) -> U { f(x) }",
    "apply(1, |n| n + 1)" => Int,
    "apply(\"s\", |s| 1)" => Int,
    "apply(1, |n| \"s\")" => Str,
);

test_ty!(
    generic_fn_body_can_annotate_params,
    "fn wrap<T>(x: T) -> [T] {
        let y: T = x;
        let xs: [T] = [y];
        xs
    }",
    "wrap(1)" => array!(Int),
);

test_ty!(
    generic_fn_called_inside_generic_fn,
    "fn id<T>(x: T) -> T { x }
    fn twice<T>(x: T) -> T { id(id(x)) }",
    "twice(1)" => Int,
);

// the signature itself stays polymorphic (params survive in the decl's type)
test_ty!(
    generic_fn_signature_has_params,
    "fn id<T>(x: T) -> T { x }",
    "id" => func!((unknown!()) -> unknown!()),
);

// -- generic structs ------------------------------------------------------------

test_ty!(
    generic_struct_fields,
    "struct Pair<A, B> { first: A, second: B }",
    "(Pair { first = 1, second = \"x\" }).first" => Int,
    "(Pair { first = 1, second = \"x\" }).second" => Str,
    "(Pair { first = \"s\", second = 2 }).first" => Str,
);

// tuple structs
test_ty!(
    generic_tuple_struct_ctor,
    "struct Pair2<A, B>(A, B)",
    "Pair2(1, \"x\").0" => Int,
    "Pair2(1, \"x\").1" => Str,
);

// -- generic enums --------------------------------------------------------------

test_ty!(
    generic_enum_variant_ctor,
    "enum Opt<T> { Nope, Yep(T) }",
    "match Opt::Yep(1) { Opt::Yep(v) => v, Opt::Nope => 0 }" => Int,
    "match Opt::Yep(\"s\") { Opt::Yep(v) => v, Opt::Nope => \"\" }" => Str,
);

test_ty!(
    generic_enum_recursive,
    "enum List<T> { E, C { first: T, rest: List<T> } }
    fn head_or<T>(l: List<T>, default: T) -> T {
        match l {
            List::E => default,
            List::C { first, rest } => first,
        }
    }",
    "head_or(List::C { first = 1, rest = List::E }, 0)" => Int,
    "head_or(List::E, \"s\")" => Str,
);

// -- generic impls --------------------------------------------------------------

test_ty!(
    generic_impl_methods,
    "enum List<T> { E, C { first: T, rest: List<T> } }
    impl List {
        fn push(self, v: T) -> List<T> {
            List::C { first = v, rest = self }
        }
        fn head_or(self, default: T) -> T {
            match self {
                List::E => default,
                List::C { first, rest } => first,
            }
        }
    }",
    "List::E.push(1).head_or(0)" => Int,
    "List::E.push(\"s\").head_or(\"\")" => Str,
);

test_ty!(
    generic_method_on_generic_adt,
    "struct Box<T> { v: T }
    impl Box {
        fn get(self) -> T { self.v }
        fn map<U>(self, f: (T) -> U) -> Box<U> { Box { v = f(self.v) } }
    }",
    "(Box { v = 1 }).get()" => Int,
    "(Box { v = 1 }).map(|x| \"s\").get()" => Str,
);

// -- failures -------------------------------------------------------------------

test_fail!(
    generic_struct_wrong_arity,
    "struct Pair<A, B> { first: A, second: B } let p: Pair<int> = Pair { first = 1, second = 2 };",
    "struct Pair<A, B> { first: A, second: B } let p: Pair<int, int, int> = Pair { first = 1, second = 2 };",
);

test_fail!(
    generic_annotation_mismatch,
    "struct Pair<A, B> { first: A, second: B } let p: Pair<int, int> = Pair { first = 1, second = \"x\" };",
);

test_fail!(
    generic_enum_wrong_arity,
    "enum Opt<T> { Nope, Yep(T) } let o: Opt<int, int> = Opt::Nope;",
);

test_fail!(
    generic_enum_inconsistent_element,
    "enum List<T> { E, C { first: T, rest: List<T> } }
    let l = List::C { first = 1, rest = List::C { first = \"s\", rest = List::E } };",
);

test_fail!(
    non_generic_applied,
    "struct Foo { x: int } let f: Foo<int> = Foo { x = 1 };",
);

// generic items reached through a module path instantiate fresh params per use.
test_multi_file!(
    generic_fn_cross_module,
    foo => "module @; pub fn id<T>(x: T) -> T { x }",
    main => "module @; pub fn use_it() -> int { foo::id(1) + foo::id(2) }";
    "main::use_it()" => Int,
);

test_multi_file!(
    generic_enum_cross_module,
    foo => "module @; pub enum List<T> { E, C { first: T, rest: List<T> } }",
    main => "module @; pub fn len(l: foo::List<int>) -> int { match l { foo::List::E => 0, foo::List::C{first, rest} => 1 + len(rest) } }";
    "main::len(foo::List::C{first=1, rest=foo::List::E})" => Int,
);

// a generic param named `A` must not be read as the ampere unit -- params take
// precedence over unit lookup in both annotation resolution and dims.
test_ty!(
    generic_param_shadows_unit,
    "fn tick<A>(x: A) -> A { x } let y = tick(3);",
    "y" => Int,
);

test_ty!(
    generic_param_named_like_unit_in_struct,
    "struct Holder<A> { v: A } let h: Holder<int> = Holder { v = 7 };",
    "h.v" => Int,
);

test_fail!(
    generic_param_shadows_unit_mismatch,
    "fn tick<A>(x: A) -> A { x } let y: str = tick(3);",
);

//! Reflection on declarations: `sig!`, `fns!`, `fields!`, `variants!` and `for_fields!`,
//! expanded by the parser (see `parse::parser::reflect`).

const FNS: &str = r#"
/// Twice a number.
fn double(x: float) -> float { x * 2.0 }

/// The long side of a right triangle
/// with short sides `a` and `b`.
#[machine]
fn hyp(a: float, b: float) -> float { (a * a + b * b).sqrt() }

#[machine]
fn majority(a: bool, b: bool, c: bool) -> bool { (a && b) || (b && c) || (a && c) }

fn shout() { }

fn names(sigs: [FnSig]) -> str { sigs.map(|s| s.name).join(" ") }
let MACHINES = fns!(machine);
"#;

test_run!(
    // r[verify mimas.reflect.sig]
    sig_is_the_signature_as_data,
    FNS,
    "sig!(double).name" => "\"double\"",
    "sig!(double).params.len()" => "1",
    "sig!(double).params[0].name" => "\"x\"",
    "sig!(double).params[0].ty" => "\"float\"",
    "sig!(double).ret" => "\"float\"",
    "sig!(double).doc" => "\"Twice a number.\"",
    // a doc comment of two lines is one line of text
    "sig!(hyp).doc" => "\"The long side of a right triangle with short sides `a` and `b`.\"",
    "sig!(hyp).params.map(|p| p.name).join(\",\")" => "\"a,b\"",
    "sig!(hyp).marks" => "[\"machine\"]",
    "sig!(double).marks.len()" => "0",
    // no return type is `()`, no params is no params
    "sig!(shout).ret" => "\"()\"",
    "sig!(shout).params.len()" => "0",
);

test_run!(
    // r[verify mimas.reflect.fns]
    // r[verify mimas.reflect.mark]
    fns_lists_the_marked_functions,
    FNS,
    "names(MACHINES)" => "\"hyp majority\"",
    "names(fns!(machine))" => "\"hyp majority\"",
    // every function declared so far, in order
    "names(fns!())" => "\"double hyp majority shout names\"",
    // a mark nobody carries gives an empty list, with a type
    "fns!(nothing).len()" => "0",
);

const LIFT: &str = r#"
fn double(x: float) -> float { x * 2.0 }
#[op]
fn add(a: float, b: float) -> float { a + b }
#[op]
fn neg(a: float) -> float { 0.0 - a }

struct Op { name: str, arity: int, run: ([float]) -> float }

// A macro of the program's own takes the declaration as tokens and
// writes an adapter: one uniform callable for functions of any arity.
macro op_of($f:ident; ($($p:ident : $t:ty),*) -> $r:ty; $doc:literal) {
    Op { name = sig!($f).name, arity = sig!($f).params.len(), run = |args| {
        let i = -1
        $(
            i += 1
            let $p = args[i]
        )*
        $f($($p),*)
    } }
}

let OPS = fns!(op => op_of);
fn run(name: str, args: [float]) -> float {
    for o in OPS { if o.name == name { let f = o.run; return f(args); } }
    -1.0
}
"#;

test_run!(
    // r[verify mimas.reflect.hand]
    a_macro_writes_an_adapter_from_a_signature,
    LIFT,
    "OPS.len()" => "2",
    "OPS.map(|o| o.name).join(\" \")" => "\"add neg\"",
    "OPS.map(|o| o.arity)" => "[2, 1]",
    "run(\"add\", [2.0, 3.0])" => "5.0",
    "run(\"neg\", [4.0])" => "-4.0",
    // one function, handed over by name
    "{ let o = sig!(double => op_of); let f = o.run; f([21.0]) }" => "42.0",
);

const TYPES: &str = r#"
struct Ball { id: float, v: [int], home: str? }
enum Shape { Dot, Circle(float), Rect { w: float, h: float }, Pair(float, float) }

macro names_of($t:ident; $($f:ident : $ty:ty),*) { [$(sig_name!($f)),*] }
macro sig_name($f:ident) { "f" }
macro count_of($t:ident; $($v:tt)*) { 0 $(+ one!($v))* }
macro one($v:tt) { 1 }
"#;

test_run!(
    // r[verify mimas.reflect.fields]
    // r[verify mimas.reflect.variants]
    fields_and_variants_are_data,
    TYPES,
    "fields!(Ball).map(|f| f.name).join(\",\")" => "\"id,v,home\"",
    "fields!(Ball).map(|f| f.ty).join(\",\")" => "\"float,[int],str?\"",
    "variants!(Shape).map(|v| v.name).join(\",\")" => "\"Dot,Circle,Rect,Pair\"",
    "variants!(Shape).map(|v| v.fields.len())" => "[0, 1, 2, 2]",
    "variants!(Shape)[2].fields.map(|f| f.name + \":\" + f.ty).join(\",\")" => "\"w:float,h:float\"",
    "variants!(Shape)[3].fields.map(|f| f.name).join(\",\")" => "\"0,1\"",
    // the `=>` form hands a struct's fields to a macro
    "fields!(Ball => names_of).len()" => "3",
);

test_fail!(
    // r[verify mimas.reflect.mark]
    // r[verify mimas.reflect.visible]
    reflection_errors,
    // a mark that no `fns!` reads is still an unknown attribute
    "#[tets]\nfn f() -> int { 1 }\nlet TEST_VALUE = f();",
    // a mark goes on a top-level function only
    "#[machine]\nstruct S { x: int }\nlet m = fns!(machine);\nlet TEST_VALUE = 1;",
    // the name must be a function declared above the call
    "let TEST_VALUE = sig!(later).name;\nfn later() -> int { 1 }",
    "let TEST_VALUE = sig!(nothing).name;",
    "struct S { x: int }\nlet TEST_VALUE = sig!(S).name;",
    "let TEST_VALUE = fields!(Nothing).len();",
    "let TEST_VALUE = variants!(Nothing).len();",
    // a call takes a name, or a name and `=> macro`
    "fn f() -> int { 1 }\nlet TEST_VALUE = sig!(f, f).name;",
    "let TEST_VALUE = sig!().name;",
);

test_run!(
    // r[verify mimas.reflect.visible]
    // a macro of the program's own with a reflection macro's name wins
    a_declared_macro_shadows_a_reflection_macro,
    "macro sig($e:expr) { $e + 1 }",
    "sig!(41)" => "42",
);

test_run!(
    // r[verify mimas.reflect.mark]
    // a read mark is legal, and `#[test]` is not a mark
    a_read_mark_is_legal,
    "#[shiny]\nfn a() -> int { 1 }\n#[shiny]\nfn b() -> int { 2 }\nlet N = fns!(shiny).len();",
    "N" => "2",
    "fns!(shiny).map(|s| s.marks.len())" => "[1, 1]",
);

test_run!(
    // r[verify mimas.reflect.sig]
    // a `>>` earlier in the source once moved every later doc comment
    // off its item
    doc_survives_a_nested_generic_above_it,
    "struct Pair<A, B> { a: A, b: B }\nfn wrap(p: Pair<int, Pair<int, int>>) -> int { 1 }\n/// Still here.\nfn after() -> int { 2 }",
    "sig!(after).doc" => "\"Still here.\"",
    "sig!(wrap).params[0].ty" => "\"Pair<int, Pair<int, int>>\"",
);

const GRAPH: &str = r#"
fn half_adder(a: int, b: int) -> (int, int) { (a, b) }

fn full_adder(a: int, b: int, c: int) -> (int, int) {
    let (s1, c1) = half_adder(a, b);
    let (s2, c2) = half_adder(s1, c);
    (s2, c1)
}

fn opaque(a: int, b: int) -> int {
    match (a, b) {
        (1, 1) => 0,
        _ => 1,
    }
}
"#;

test_run!(
    // r[verify mimas.reflect.graph]
    graph_is_a_functions_body_as_data,
    GRAPH,
    "graph!(full_adder).name" => "\"full_adder\"",
    "graph!(full_adder).ins" => "[\"a\", \"b\", \"c\"]",
    "graph!(full_adder).outs" => "[\"0\", \"1\"]",
    "graph!(full_adder).opaque" => "false",
    // in-nodes, then the two calls, then the sels the destructures made
    "graph!(full_adder).nodes.map(|n| n.op).join(\",\")" => "\"in,in,in,call,sel,sel,call,sel,sel\"",
    "graph!(full_adder).nodes[3].name" => "\"half_adder\"",
    "graph!(full_adder).nodes[3].args" => "[0, 1]",
    "graph!(full_adder).nodes[4].name" => "\"0\"",
    "graph!(full_adder).nodes[6].args" => "[4, 2]",
    "graph!(full_adder).rets" => "[7, 5]",
    // a tuple result of single out-parts is its own wires
    "graph!(half_adder).rets" => "[0, 1]",
    // what is not dataflow is opaque, and says why
    "graph!(opaque).opaque" => "true",
    "graph!(opaque).why" => "\"a match expression\"",
    "graph!(opaque).ins" => "[\"a\", \"b\"]",
);

test_run!(
    // r[verify mimas.reflect.hand]
    // r[verify mimas.reflect.graph]
    graph_hands_the_graph_to_a_macro,
    "macro pick($f:ident; $g:expr) { $g.name + \"!\" }\nfn half(a: int) -> int { a }",
    "graph!(half => pick)" => "\"half!\"",
);

test_fail!(
    // r[verify mimas.reflect.visible]
    graph_errors,
    "let TEST_VALUE = graph!(later).name;\nfn later() -> int { 1 }",
    "let TEST_VALUE = graph!(nothing).name;",
);

// r[verify mimas.reflect.for-fields]
// r[verify mimas.reflect.for-fields-filter]
test_run!(
    for_fields_typed_fields_without_runtime_metadata,
    "pact Entity { fn value(self) -> int; }
     struct A { n: int } struct B { n: int }
     impl Entity for A { fn value(self) -> int { self.n } }
     impl Entity for B { fn value(self) -> int { self.n * 2 } }
     struct World { aa: [A], count: int, bb: [B] }
     let w = World { aa = [A { n = 3 }], count = 99, bb = [B { n = 4 }] };
     let out: [int] = []; let names: [str] = [];
     for_fields!(World, w, [Entity] => |field| {
        names.push(field.key);
        for item in field.value { out.push(item.value()); }
     });",
    "out" => "[3, 8]",
    "names" => "[\"a\", \"b\"]",
);

test_run!(
    for_fields_generate_mutable_accessors,
    "pact Entity { fn erase(self); }
     struct A { gone: bool } struct B { gone: bool }
     impl Entity for A { fn erase(self) { self.gone = true; } }
     impl Entity for B { fn erase(self) { self.gone = true; } }
     struct World { aa: [A], bb: [B] }
     for_fields!(World, [Entity] => |field| {
        impl World { subscript field.key_ident(i: int) -> field.item_type {
            inout { yield self.field.ident[i]; }
            remove { self.field.ident[i].erase(); }
        } }
     });
     let w = World { aa = [A { gone = false }], bb = [B { gone = false }] };
     w.a[0].remove(); w.b[0].gone = true;",
    "w.aa[0].gone" => "true",
    "w.bb[0].gone" => "true",
);

test_run!(
    for_fields_generated_accessors_keep_each_dimension,
    "struct W { lengths: [px], times: [s] }
     for_fields!(W, [_] => |field| { impl W {
        subscript field.key_ident(i: int) -> field.item_type { inout { yield self.field.ident[i]; } }
     } });
     let w = W { lengths = [1px], times = [1s] }; w.px[0] = 3px; w.s[0] = 2s;",
    "w.px[0].to(1px)" => "3.0",
    "w.s[0].to(1s)" => "2.0",
);

// r[verify mimas.reflect.for-fields-cost]
#[test]
fn for_fields_cost_the_same_as_typed_loops() {
    let mut vm = vm::Vm::execute(
        "struct W { a: [int], b: [int], ignore: str }
         let xs: [int] = []; for i in 1000 { xs.push(i); }
         let w = W { a = xs, b = xs, ignore = \"skip\" };
         fn direct() -> int { let sum = 0; for n in w.a { sum += n; } for n in w.b { sum += n; } sum }
         fn expanded() -> int { let sum = 0; for_fields!(W, w, [int] => |field| { for n in field.value { sum += n; } }); sum }",
        library::std,
    ).unwrap();
    vm.set_op_budget(100_000);
    let direct = vm.call_fn_result("direct").unwrap().to_string();
    let direct_ops = 100_000 - vm.ops_left();
    vm.set_op_budget(100_000);
    assert_eq!(vm.call_fn_result("expanded").unwrap().to_string(), direct);
    let for_fields_ops = 100_000 - vm.ops_left();
    assert!(for_fields_ops <= direct_ops + 8, "for_fields {for_fields_ops}, direct {direct_ops}");
}

// r[verify mimas.reflect.for-fields]
test_run!(
    for_fields_interpolation_preserves_literal_text_and_captured_locals,
    "struct W { a: [int] } let w = W { a = [7] }; let n = 2; let out: [str] = [];
     for_fields!(W, w, [int] => |field| { out.push(f\"field.name {{literal}} {field.name}:{n}:{field.value[0]}\"); });",
    "out" => "[\"field.name {literal} a:2:7\"]",
);

test_fail!(
    for_fields_report_bad_filters_and_properties,
    "struct W { a: int } let w = W { a = 3 }; for_fields!(W, w, NotAType => |f| { f.value; });",
    "struct W { a: int } let w = W { a = 3 }; for_fields!(W, w, _ => |f| { f.missing; });",
    "struct W { a: int } for_fields!(W, _ => |f| { let x = f.value; });",
    "struct W { a: [px], b: [s] } for_fields!(W, [_] => |f| { impl W { subscript f.key_ident(i: int) -> f.item_type { inout { yield self.f.ident[i]; } } } }); let w = W { a = [1px], b = [1s] }; w.px[0] = 2s;",
);

test_fail!(
    removed_field_expansion_spellings_are_not_builtins,
    "struct W { a: int } let w = W { a = 3 }; mirror!(W, w, _ => |f| { f.value; });",
    "struct W { a: int } let w = W { a = 3 }; 🪞!(W, w, _ => |f| { f.value; });",
);

test_run!(
    nested_for_fields_can_shadow_the_field_parameter,
    "struct W { a: [int], b: [int] } let w = W { a = [1], b = [2] }; let out: [str] = [];
     for_fields!(W, w, [int] => |field| {
        let outer = field.name;
        for_fields!(W, w, [int] => |field| { out.push(outer + field.name); });
     });",
    "out" => "[\"aa\", \"ab\", \"ba\", \"bb\"]",
);

// r[verify mimas.reflect.for-fields-once]
test_run!(
    for_fields_evaluate_receiver_once_even_without_matching_fields,
    "struct W { a: [int] } fn owner(log: [int]) -> W { log.push(1); W { a = [3] } }
     let log: [int] = [];
     for_fields!(W, owner(log), [int] => |f| { let _x = f.value[0]; });
     for_fields!(W, owner(log), [str] => |f| { let _x = f.value[0]; });",
    "log" => "[1, 1]",
);

// r[verify mimas.reflect.for-fields-order]
test_run!(
    for_fields_priorities_preserve_unlisted_fields,
    "struct W { a: int, b: int, c: int } let w = W { a = 1, b = 2, c = 3 }; let out: [int] = [];
     for_fields!(W, w, int, [\"b\"] => |f| { out.push(f.value); });",
    "out" => "[2, 1, 3]",
);

test_run!(
    for_fields_expand_inside_accessor_arms_and_keep_later_arms,
    "struct W { a: [int], b: [int] }
     impl W {
        subscript total() -> int {
            get { let n = 0; for_fields!(W, self, [int] => |f| { for x in f.value { n += x; } }); n }
            set { self.a[0] = new_value; }
        }
     }
     let w = W { a = [1], b = [2] }; let before = w.total; w.total = 7;",
    "before" => "3",
    "w.total" => "9",
);

test_run!(
    for_fields_keep_nested_fixed_list_element_types,
    "struct W { rows: [[int; 2]] } let w = W { rows = [[1, 2]] }; let names: [str] = [];
     for_fields!(W, w, [_] => |f| { names.push(f.item_name); let row: f.item_type = f.value[0]; names.push(f\"{row[1]}\"); });",
    "names" => "[\"[int;2]\", \"2\"]",
);

// `sig!` reports a param's lend convention: `x: inout T` and `inout self`
// read `inout = true`, a plain param `false`, and `ty` keeps the spelling.
test_run!(
    sig_reports_inout_params,
    "fn bump(x: inout int) { x += 1 }
     fn peek(x: int) -> int { x }
     let s1 = sig!(bump); let s2 = sig!(peek);",
    "s1.params[0].inout" => "true",
    "s2.params[0].inout" => "false",
    "s1.params[0].ty" => "\"inout int\"",
);

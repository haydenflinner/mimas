//! Reflection on declarations: `sig!`, `fns!`, `fields!` and `variants!`,
//! expanded by the parser (see `parse::parser::reflect`).
#[macro_use]
mod test_runner;

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

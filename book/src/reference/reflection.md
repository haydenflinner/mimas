# Reflection

A program can ask what it declared. Built-in macros answer:
`sig!` for one function, `fns!` for many, `fields!` for a struct,
`variants!` for an enum, `graph!` for a body, and `mirror!` for typed field templates. The parser expands
them, as it expands a macro you declare, before anything is checked.

## A function as data

```mimas
/// Twice a number.
fn double(x: float) -> float { x * 2.0 }

let s = sig!(double);
s.name              // "double"
s.params[0].name    // "x"
s.params[0].ty      // "float"
s.ret               // "float"
s.doc               // "Twice a number."
```

`sig!(name)` is a `FnSig`:

```mimas
struct FnParam { name: str, ty: str }
struct FnSig { name: str, params: [FnParam], ret: str, doc: str, marks: [str] }
```

A type is text, as it was written in the declaration. A function with
no return type has the `ret` `"()"`. A doc comment of several lines is
one line of text.

## Marks

An attribute that the language does not know is a mark on a function.
`fns!(name)` is the list of the functions that carry it, in the order
they were declared:

```mimas
#[machine]
fn hyp(a: float, b: float) -> float { (a * a + b * b).sqrt() }

#[machine]
fn majority(a: bool, b: bool, c: bool) -> bool { (a && b) || (b && c) || (a && c) }

fns!(machine).map(|f| f.name)   // ["hyp", "majority"]
fns!()                          // every function declared so far
```

A mark is legal when some `fns!` in the program reads it. A mark that
nothing reads is an error, so a misspelled `#[tets]` does not pass in
silence. `#[test]` is not a mark.

## Handing a declaration to a macro

The value forms tell you about a function. They cannot call it with
arguments they do not know at compile time. For that, the `=>` form
hands the declaration to a macro of yours as tokens, and the macro
writes the code:

```mimas
sig!(hyp => m)        // m!(hyp; (a: float, b: float) -> float; "doc comment")
fns!(machine => m)    // [m!(hyp; …), m!(majority; …)]
```

A macro that takes those tokens can write one adapter for functions of
any arity:

```mimas
struct Op { name: str, arity: int, run: ([float]) -> float }

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

let OPS = fns!(machine => op_of);
```

A macro's rules can match a type by its name (`(float, $v:expr) => …`),
so an adapter can treat each parameter by its type.

## Structs and enums

```mimas
struct Ball { id: float, v: [int], home: str? }
enum Shape { Dot, Circle(float), Rect { w: float, h: float } }

fields!(Ball)          // [StructField { name = "id", ty = "float" }, …]
fields!(Ball => m)     // m!(Ball; id: float, v: [int], home: str?)

variants!(Shape)       // [EnumVariant { name = "Dot", fields = [] }, …]
variants!(Shape => m)  // m!(Shape; Dot, Circle(float), Rect { w: float, h: float })
```

```mimas
struct StructField { name: str, ty: str }
struct EnumVariant { name: str, fields: [StructField] }
```

A variant written `Circle(float)` has fields named `0`, `1`, and so on.

## Typed field mirrors

A mirror expands a template once per selected field. `🪞!` is the same
operation, with a shorter spelling.

```mimas
pact Entity { fn tick(self); }
struct Ball { ticks: int }
impl Entity for Ball { fn tick(self) { self.ticks += 1; } }
struct World { balls: [Ball], paused: bool }
let w = World { balls = [Ball { ticks = 0 }], paused = false };

mirror!(World, w, [Entity] => |field| {
    for entity in field.value { entity.tick(); }
});
```

The filter `[Entity]` selects list fields whose declared element type
implements `Entity`. `[int]` selects integer lists, `[_]` all lists, and
`_` every field. A bare type or pact selects fields directly. The receiver
runs once, including when no fields match. It must have the named type.

Each copy is ordinary checked code. The example becomes a direct loop over
`w.balls`; it does not build a list of entities, enumerate metadata at
runtime, or pack each object into a pact value. Type, dimension, effect,
const and iteration checks still apply to the generated code.

The template parameter supplies these properties:

| Property | Expansion |
|---|---|
| `field.value` | The receiver's field, with its original type |
| `field.name` | Field name as a string |
| `field.ident` | Field name as an identifier |
| `field.type` | Declared field type as tokens |
| `field.type_name` | Declared field type as a string |
| `field.item_type` | List element type as tokens |
| `field.item_name` | List element type as a string |
| `field.key` | Lowercase type name, with underscores before capitals; list fields use the element type |
| `field.key_ident` | The same key as an identifier |

Properties also work in interpolation expressions. Literal string text and
escaped braces keep their original meaning.

Omit the receiver to declare items. This derives mutable accessors and a
collection-owned removal policy from the same list declarations:

```mimas
mirror!(World, [Entity] => |field| {
    impl World {
        subscript field.key_ident(i: int) -> field.item_type {
            inout { yield self.field.ident[i]; }
            remove { self.field.ident[i].remove(); }
        }
    }
});
```

Here `w.ball[0]` is mutable, and `w.ball[0].remove()` removes the list slot.
A world that uses tombstones can instead call the entity's erase method
in its removal arm. The caller's spelling stays the same.

An optional list of field names sets priorities:

```mimas
mirror!(World, w, [Entity], ["balls"] => |field| {
    for entity in field.value { entity.tick(); }
});
```

Listed fields come first. Remaining matches keep declaration order, so a
new entity list is included without another registry entry. Unknown or
repeated priority names are errors.

Put mirrors after the struct and the pact implementations they inspect.
The body is a template, not a runtime closure. A `return` returns from its
enclosing function. A loop inside the template has ordinary loop rules.

## A body as data

`graph!(name)` lowers a function's body to a dataflow, so a kit can
read how a result is made instead of only how the function is called:

```mimas
fn half_adder(a: int, b: int) -> (int, int) { (a, b) }

fn full_adder(a: int, b: int, c: int) -> (int, int) {
    let (s1, c1) = half_adder(a, b);
    let (s2, c2) = half_adder(s1, c);
    (s2, c1)
}

let g = graph!(full_adder);
g.ins                 // ["a", "b", "c"]
g.outs                // ["0", "1"]
g.nodes[3]            // FnNode { op = "call", name = "half_adder", args = [0, 1] }
g.rets                // [7, 5]
```

```mimas
struct FnNode { op: str, name: str, args: [int] }
struct FnGraph { name: str, ins: [str], outs: [str], nodes: [FnNode],
                 rets: [int], opaque: bool, why: str }
```

Each node is one value a wire can carry: `in` is a parameter, `const`
a literal, `call` a call by name, `sel` one part of a call's result.
`args` are node indexes, the wires a call reads. `rets` names the
nodes the result is made of, and `outs` names the result's parts —
`["0", "1"]` for a pair, `["out"]` for one.

The lowering is small on purpose. Calls, tuples, `let`, parameters and
literals are dataflow; a `match`, a loop or an operator is not, and the
answer's `opaque` is true with `why` saying what the body was. It is a
value, not an error, so one kit can draw what it can and report what it
cannot. `graph!(name => m)` hands the graph to a macro as
`m!(name; FnGraph { … })`.

## What it sees

The answer is what the parser has read so far. A declaration is visible
below it, in the same file and in every file of the program parsed
after it, as a macro is. `sig!(later)` above `fn later` is an error.

Only top-level functions count: not methods, and not the functions std
ships. A macro you declare with one of these names takes its place.

Types are text. `sig!` reports `float` for a parameter written `float`
and reports nothing useful for a parameter with no annotation; it does
not know what the checker inferred.

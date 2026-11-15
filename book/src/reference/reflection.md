# Reflection

A program can ask what it declared. Four built-in macros answer:
`sig!` for one function, `fns!` for many, `fields!` for a struct and
`variants!` for an enum. The parser expands them, as it expands a
macro you declare, before anything is checked.

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

## What it sees

The answer is what the parser has read so far. A declaration is visible
below it, in the same file and in every file of the program parsed
after it, as a macro is. `sig!(later)` above `fn later` is an error.

Only top-level functions count: not methods, and not the functions std
ships. A macro you declare with one of the four names takes its place.

Types are text. `sig!` reports `float` for a parameter written `float`
and reports nothing useful for a parameter with no annotation; it does
not know what the checker inferred.

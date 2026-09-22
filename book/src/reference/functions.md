# Functions & Closures

Functions are declared with `fn`. Parameters must be annotated; the return type follows `->` and defaults to `()` when omitted.

```mimas
fn greet(name: str) {
    print(f"hello, {name}!");
}

fn square(n: int) -> int {
    n * n
}
```

The body is a [block](./blocks.md), so its final bare expression is the return value. The annotated return type is enforced:

```mimas
fn oops() {
    0 // compile error: expected (), found int
}
```

`return` exits early and, being [never-typed](./special-types.md), fits anywhere:

```mimas
fn clamp_low(n: int) -> int {
    if n < 0 {
        return 0;
    }
    n
}
```

## Optional parameters

A parameter with a default value becomes optional. Like a `let`, its type can be inferred from the default, so the annotation is optional too.

```mimas
fn connect(host: str, port=8080) {
    // ...
}

connect("localhost");        // port defaults to 8080
connect("localhost", 9090);  // or pass it explicitly
```

Defaults must be [const-foldable](./variables.md#constants), and every optional parameter has to come **after** all the required ones.

```mimas
fn bad(a: int, b=0, c: int) {} // compile error: a required parameter can't follow an optional one
fn good(a: int, c: int, b=0) {} // valid
```

## Named arguments

Optional arguments may be passed by name -- again, only once all the required positional arguments are supplied. Naming lets you skip over defaults you don't care about and set one further along:

```mimas
fn style(text: str, bold=false, italic=false, size=12) {}

style("hi", size=18, bold=true); // skip `italic`, set the other two
```

## Tests

A function marked `#[test]` is a check the inspector runs when it loads the file -- it is not part of the script's top-to-bottom execution. Tests take no parameters and are called with none; a panic (or any other runtime error) is a failure, anything else is a pass.

```mimas
fn add(a: int, b: int) -> int {
    a + b
}

#[test]
fn add_works() {
    assert!(add(1, 2) == 3)
}
```

`assert!(expr)` is rewritten at parse time into an `if`/`panic` check. Comparisons (`==`, `!=`, `<`, `>`, `<=`, `>=`) and membership (`in`, `!in`) evaluate each side once and interpolate the values into the panic message, pytest-style; any other expression becomes `if !expr { panic("assertion failed: ...") }`. `assert` is a normal name -- only `assert!(...)` is special.

`#[test]` sits in front of `pub` when both are present (`#[test] pub fn ...`). It is only valid on free functions -- not methods, and not other items.

A `#[tests]` list is a compact form of the same thing -- a list of boolean expressions the inspector evaluates when it loads the file. `true` is a pass; `false` or a panic is a failure. The check's source is its name.

```mimas
fn add(a: int, b: int) -> int { a + b }

#[tests] [
    add(1, 2) == 3,
    add(0, 0) == 0,
]
```

The literate spelling of the same list is a `where:`/`examples`/`example` check block, which lowers to exactly a `#[tests]` item. Each line is an expression, or `expr is expr` where `is` means `==`. A `where:` paragraph runs to the next blank line or item; braces make the boundary explicit.

```mimas
fn add(a: int, b: int) -> int { a + b }

where:
    add(1, 2) is 3
    add(0, 0) is 0

examples {
    add(-1, 1) is 0
    add(2, 2) is 4
}
```

## Closures

A closure is an inline, anonymous function written with pipes. Its return type is optional and inferred when left off.

```mimas
let add = |a: int, b: int| -> int { a + b };
let double = |n: int| { n * 2 }; // return type inferred

let total = add(double(3), 4); // 10
```

A no-argument closure uses an empty pair of pipes:

```mimas
let now = || current_time();
```

```admonish note
Functions cannot currently be passed to native functions and called. This is a high priority to be added after `v0.1.0`.
```

### Closures vs. functions

The two differ in two important ways.

**First- vs. second-class.** Closures are *first-class*: assign them to bindings, store them in arrays, pass them around freely. Functions are *second-class* -- you can pass one as an argument, but you can't bind or store it as a value.

```mimas
let add = |a: int, b: int| -> int { a + b };
fn sub(a: int, b: int) -> int { a - b }

apply(add); // legal -- passing a closure
apply(sub); // legal -- passing a function as an argument is fine

let f = add; // legal
let g = sub; // compile error: a function isn't a value you can bind
```

**Capturing.** A closure can capture variables from the scope around it. A function cannot -- it sees only its parameters and top-level items.

```mimas
let greeting = "hello!";

let say = || print(greeting); // legal -- captures `greeting`

fn say_fn() {
    print(greeting); // compile error: `greeting` is undefined in here
}
```

````admonish note title="Functions live at the top level"
`fn` declarations are only allowed at the root of a file. Need a function-like value deeper inside another function or a block? That's exactly what closures are for.

```mimas
fn outer() {
    fn inner() {}      // compile error: no nested functions
    let inner = || {}; // do this instead
}
```
````

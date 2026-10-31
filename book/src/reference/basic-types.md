# Primitive Types

mimas has four primitive types: `int`, `float`, `bool`, and `str`.

## Integers

`int` is a 64-bit signed integer. Integer literals can be written in decimal or hexadecimal, and `_` may be used anywhere as a visual separator (it is ignored).

```mimas
let a: int = 0;
let b: int = 0xff;          // 255, in hex
let c: int = 1_000_000;     // underscores are just for readability
```

Integer arithmetic is **checked**: an operation that overflows the 64-bit range is a runtime error rather than silently wrapping around.

```mimas
let big = 9223372036854775807; // i64::MAX
let oops = big + 1;            // runtime error: integer arithmetic overflowed
```

## Floats

`float` is a 64-bit IEEE-754 floating-point number.

```mimas
let a: float = 0.0;
let b: float = 1_000.000_1;
```

An `int` and a `float` can be combined in one expression, but the result is always a `float` -- the `int` is promoted. There is no implicit `float` -> `int`.

```mimas
let a: float = 1 + 0.1; // valid -- (int + float) => float
let b: int = 1 + 0.1;   // compile error: expected int, found float
```

````admonish warning title="Division always yields a float"
The `/` operator *always* produces a `float`, even between two `int`s. When you want integer (truncating) division, use the `~/` operator.

```mimas
let a: float = 7 / 2;  // 3.5
let b: int   = 7 ~/ 2; // 3
```
````

To convert deliberately between the two, use the conversion methods rather than a cast -- mimas has no `as`.

```mimas
let n: int = 42;
let f: float = n.to_float(); // 42.0
let back: int = f.to_int();  // 42 (truncates toward zero)
```

## Booleans

`bool` is `true` or `false`. Comparisons and logical operators produce `bool`s, and conditions in `if`, `while`, and friends must be `bool`.

```mimas
let a: bool = true;
let b = 3 > 2;        // true
let c = a && !b;      // false
```

See [Operators](./operations.md) for the full set of comparison and logical operators.

## Strings

Text is always `str`. mimas has no separate character type -- a single character is just a one-character `str`. Strings use double quotes only (`"`), never single (`'`).

```mimas
let a: str = "hello!";
let b: str = "x"; // a one-character str, not a `char`
```

Any string literal may span multiple lines; the newline becomes part of the value.

```mimas
let poem = "this string has
multiple lines!"; // -> "this string has\nmultiple lines!"
```

Index a string with an `int` to read the character at that position. Positions count characters rather than bytes, so they line up with `len()` and with `for c in s`. Reading past the end is a runtime error, and since strings are immutable, assigning into an index is a compile error.

```mimas
let word = "café";
let first = word[0];              // -> "c"
let last = word[word.len() - 1];  // -> "é"
word[0] = "C";                    // compile error: strings can't be assigned into
```

Indexing a string that is entirely ASCII is constant-time; a string holding any other characters is scanned from the start.

### Interpolation (f-strings)

Prefix a literal with `f` to interpolate expressions inside `{ }`. Any expression is allowed.

```mimas
let name = "world";
let greeting = f"hello, {name}! 1 + 1 = {1 + 1}"; // -> "hello, world! 1 + 1 = 2"
```

To write a literal brace, double it:

```mimas
let s = f"{{not interpolated}} but {1 + 1} is"; // -> "{not interpolated} but 2 is"
```

### Tagged literals

A name glued to a string — no blank between — is a *tagged literal*:

```mimas
let beat = pat"bd ~ [sd sd] <cp hh>";
let walk = keys"right@5 up@2";
```

`tag"…"` is sugar for `__lit_tag("…")`: a call of a prelude native the
host registers per tag, so the literal is typed by that native's return
and — because its one argument is a literal — checked by the native's
validator when the program compiles. A tag no native backs is a compile
error that names the tag. Keywords never tag, and `f"…"` is the f-string.

A program declares its own tag with `notation`, which is `fn __lit_<tag>`
under a working name:

```mimas
notation twice(s: str) -> str { s + s }
let x = twice"ab"; // "abab"
```

Validators (`api::LitValidator`) run whenever *every* argument of a call
to a native is a literal; a fault may name the argument and a char span
inside it, and the error lands there.

```mimas
let s = f"{{not interpolated}} but {1 + 1} is"; // -> "{not interpolated} but 2 is"
```

````admonish tip title="String methods chain"
`str` has a wide set of built-in methods, and they chain:

```mimas
let cleaned = "  Hello, World  ".trim().to_lower(); // -> "hello, world"
```
````

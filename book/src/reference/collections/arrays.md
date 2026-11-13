# Arrays

An array is an ordered, growable sequence of values that all share one type -- like a `list` in Python or a `Vec` in Rust. Its type is written `[T]`.

```mimas
let xs: [int] = [0, 1, 2, 4];
let first = xs[0]; // 0 -- index with []
```

Indexing is zero-based, and reading past the end is a runtime error:

```mimas
let xs = [1, 2, 3];
let oops = xs[9]; // runtime error: index out of bounds
```

Arrays carry a set of built-in methods for inspecting and growing them, such as `len`, `push`, and `contains`:

```mimas
let xs = [3, 1, 2];
xs.push(4);       // xs is now [3, 1, 2, 4]
let n = xs.len(); // 4
```

To build a *new* array by transforming or filtering an existing one, use a [`collect`](../control-flow/collection.md) loop:


```mimas
let doubled = for x in [1, 2, 3] collect x * 2; // [2, 4, 6]
```

## Gather and pervasive comparison

Indexing with a list of ints gathers: `xs[[2, 0]]` is `[xs[2], xs[0]]`, and a nested index list gathers a nested result. Comparing a list to a scalar compares every element and answers a list of bools, nested as deep as the list, the way a tensor does (`[1, 2, 3] == 2` is `[false, true, false]`; two lists still compare as wholes). Together with the reductions this is tic-tac-toe's win check in one line:

```mimas
const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
fn won(b: [Player?], p: Player) -> bool { ⨆⨅ (b[LINES] == p) }   // gather, compare, meet each line, join the lines
```

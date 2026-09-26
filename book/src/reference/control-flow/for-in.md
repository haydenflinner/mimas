# For

A `for` loop walks over the elements of something iterable, binding each one in turn.

```mimas
let xs = [10, 20, 30];
for x in xs {
    print(x); // 10, then 20, then 30
}
```

## What you can iterate

| Iterable | Each binding is |
| :--- | :--- |
| `[T]` (array) | an element, `T` |
| `~{V}` (dict) | a `(str, V)` pair of key and value |
| `str` | each character, as a one-character `str` |
| `int` | the numbers `0` up to (but not including) the value |

```mimas
for pair in ~{ x = 1, y = 2 } {
    print(pair); // [x, 1], then [y, 2]
}

for c in "hi" {
    print(c); // "h", then "i"
}

for i in 3 {
    print(i); // 0, 1, 2
}
```

````admonish tip title="Need the index?"
Call `.enumerate()` on an array to pair each element with its position:

```mimas
for pair in ["a", "b"].enumerate() {
    print(pair); // [0, a], then [1, b]
}
```
````

Tuples are intentionally *not* iterable: each position can hold a different type, so a single loop binding would have no consistent type. Reach into a tuple by index instead (`t.0`, `t.1`). See [Tuples](../collections/tuples.md).

## Breaking a value

Like [`while`](./while.md), a `for` loop isn't guaranteed to run -- the collection might be empty -- so a value it breaks comes back as an [option](../options.md).

```mimas
let first_big: int? = for n in numbers {
    if n > 100 {
        break n; // -> int?, since `numbers` could be empty
    }
};
```

A `for` loop that never breaks a value evaluates to `()`. To *build* a value out of every iteration instead of breaking once, use [`collect`](./collection.md).

```admonish warning title="Mutation during iteration"
Do not mutate a collection while you are iterating over it. The compiler rejects direct cases -- writing to it by index (`xs[i] = v`) or calling a method that modifies it (`xs.push(0)`, `d.insert(k, v)`) inside the loop body:

```mimas
let xs = [1, 2];
for x in xs {
    xs.push(x); // error: cannot mutate `xs` while iterating over it
}
```

The guard follows the place being iterated, so it also sees through field and index accesses (`for x in w.arr { w.arr[0] = 0 }` is likewise rejected). It is a lint for the common trap, not an alias analysis: mutation reached through another binding for the same collection (`let ys = xs; ys.push(0)`), through a helper function, or through an indirect call is still possible and can skip elements, visit them twice, or loop forever. <!-- TODO: a runtime check (e.g. a mutation counter or an iteration lock on the collection) could catch the indirect cases at some per-write cost. -->

If you need to change a collection as you walk it, gather the changes and apply them afterward, loop over a snapshot instead (`let snapshot = for x in xs collect x;` then `for x in snapshot`), or use a `loop` with the bounds handled yourself. Mutating the *elements* (`inner.push(0)` on an array of arrays, `xs[i].field = v`) is fine -- it is changes to the collection's own contents that are rejected.

When the compiler can prove a loop does not change the length of what it iterates, it computes that length once instead of on every pass. This is only an optimization, and the cost of missing it is negligible unless you run many thousands of iterations.
```
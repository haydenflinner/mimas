# Pacts

mimas is built for embedded use, so Rust's full trait machinery -- associated types, blanket impls, coherence rules, and the constraint solver behind them -- is intentionally out of scope. Even so, abstracting over types is sometimes genuinely useful: without any way to do it, every function that wants to handle "anything with this behavior" has to be rewritten per concrete type.

It comes up most at the FFI line. A host project large enough to span several crates often leans on traits at its boundaries, and a mimas script needs some way to interoperate with at least the simplest of those patterns.

A **pact** fills that role. It's a named set of method, associated-function, and constant signatures -- a contract a type satisfies by writing `impl PactName for TheType { ... }` and supplying the listed items. Anywhere a type is expected, a pact can stand in, and the checker will accept any value whose type fulfills it.

```mimas
pact Draw {
    fn draw(self);
}

struct Square;
impl Draw for Square {
    fn draw(self) { print("a square"); }
}

struct Circle;
impl Draw for Circle {
    fn draw(self) { print("a circle"); }
}

fn render_all(items: [Draw]) {
    for item in items {
        item.draw();
    }
}

render_all([Square, Circle]);
```

A collection literal holding several types settles on the pacts they share, which is how you build "a list of things that implement `Draw`" without writing a type parameter. If the elements have no pact in common, it's an ordinary type mismatch.

## Bounded type parameters

When a function or type needs to stay generic, a declared type parameter carries pact bounds: `fn f<T: Draw + Vary>(x: T)`. Inside the item, a value of the parameter's type reaches the bounds' members, dispatching at run time to the concrete type's implementation, and `Self` in a pact's signatures reads as the parameter.

```mimas
pact Show {
    fn show(self) -> str;
}

fn label<T: Show>(x: T) -> str {
    x.show()
}
```

Every instantiation is checked against the bounds once its type settles, and a member call on an unbounded parameter is an error that names the missing bound. Type parameters work without bounds too (`fn id<T>(x: T) -> T`, `struct Pair<A, B>`); a bound is what buys a parameter its members.

## Constants

A pact can require constants as well as methods.

```mimas
pact Named {
    const NAME: str;
}

struct Square;
impl Named for Square {
    const NAME = "Square";
}

let n = Square::NAME; // "Square"
```

Constants are read off a **concrete type**. Unlike methods, they don't dispatch -- see [Limitations](#limitations) below.

## Default implementations

A pact method can ship with a default body, which an implementer may use as-is or override.

```mimas
pact Greet {
    fn name(self) -> str;

    fn hello(self) -> str {
        f"hi, {self.name()}"
    }
}

struct Dog { tag: str }
impl Greet for Dog {
    fn name(self) -> str { self.tag }
    // `hello` is inherited
}

print(Dog { tag = "rex" }.hello()); // "hi, rex"
```

A default body can call the pact's other methods through `self`, including ones the implementer supplies. An impl that defines the method itself overrides the default.

## `Self`

Inside a pact `Self` is the implementing type. A method that works on two values of the same type spells the second one `Self`, and each implementer puts its own type there:

```mimas
pact Plus {
    fn plus(self, other: Self) -> Self;
}

struct V { x: int }
impl Plus for V {
    fn plus(self, other: V) -> V {
        V { x = self.x + other.x }
    }
}

struct W { y: int }
impl Plus for W {
    fn plus(self, other: W) -> W {
        W { y = self.y + other.y }
    }
}

V { x = 1 }.plus(V { x = 2 }); // V { x = 3 }
V { x = 1 }.plus(W { y = 2 }); // error: expected Self but found W
```

`Self` is not the same as writing the pact's name. `other: Plus` means any implementer at all; `other: Self` means the receiver's own type, and an impl has to name exactly that type -- `impl Plus for V` can't take `other: W`. Inside a default body, `self` has type `Self`, so it can be passed wherever the pact asks for `Self`, and also wherever it asks for the pact itself, since a `Self` is always an implementer.

## Reaching pact methods through a value

A value known only by its pact -- or by a bounded type parameter -- reaches the pact's members through [dot access](./types/structs.md#reaching-items-through-a-value): `.` is how you get at its methods.

```mimas
fn announce(thing: Greet) {
    print(thing.hello());
}
```

## Binding multiple pacts

An annotation can require several pacts at once with `+`:

```mimas
struct Widget {
    item: Named + Greet,
}
```

To make a multi-pact bound optional or a result, wrap it in parentheses first so the `?`/`!` applies to the whole thing:

```mimas
struct Widget {
    item: (Named + Greet)?,
}
```

## Requiring another pact

A pact can say "whoever implements me must also implement these" after a colon:

```mimas
pact Game: Draw {
    fn tick(self, inp: Input, evs: [Ev]) -> Self;
}
```

Every type that implements `Game` must implement `Draw` too -- a missing impl is an error at the `impl Game` that owes it, naming the pact and the impl to write. The requirement is transitive, and the check runs once every impl is gathered, so the order impls are written in never decides it.

Wherever `Game` bounds a value, the required pact's members are reachable through it: `thing.draw()` on a `thing: Game` dispatches at run time like any `Game` member, and a `Game` value satisfies a `Draw` bound.

## Forwarding members with `via`

An impl can hand the members it doesn't write to another pact's impl of the same type:

```mimas
impl Named for Widget via Greet
```

Each member of `Named` the block omits is answered by the `Greet` member of the same name, called through `Widget`'s `Greet` implementation, and dispatches exactly as a member written out would. The block may still write members; a written one always wins over forwarding. The braces may be dropped when the block is empty.

The coverage rules:

- A `Greet` member covers the `Named` member only when its signature fulfils the `Named` member's once `Self` reads as `Widget` on both sides. A mismatch is an error naming the member and both pacts.
- `Widget` must implement `Greet`. Where no `impl Greet for Widget` exists, `via` builds exactly what `impl Greet for Widget {}` would -- possible only while every `Greet` member carries a default. The built impl owes `Greet`'s required pacts as a written one would.
- An omitted member no `Greet` member names keeps `Named`'s own default where it has one; with neither, it joins the missing-member report like a member an ordinary impl forgot.
- An overloaded member -- one name declared for several operand types -- is never matched by name and must be written in the block. A `const` member is covered only when the type's own surface already holds the name.

## Limitations

Pacts are the youngest part of the language, and 0.1.0 ships a deliberately small slice of them. Each of the following is rejected with a clear error rather than silently misbehaving.

**Constants don't dispatch.** A pact constant can only be read off a concrete type. Reaching one through a pact-typed value has no answer at runtime -- every impl declares its own value, and the receiver's concrete type isn't known:

```mimas
fn tag_of(thing: Named) -> str {
    thing.NAME // error: pact constant `NAME` can't be reached through `Named`
}
```

The workaround is to require a method instead, which does dispatch:

```mimas
pact Named {
    fn name(self) -> str;
}
```

**Methods that take `Self` can't be called through a pact.** `Self` in a parameter means "the same type as the receiver", and a value known only by its pact doesn't say what that is -- `a.plus(b)` with both typed `Plus` could pair a `V` with a `W`. The call is rejected wherever the receiver is only known by its bound, even when the arguments happen to match:

```mimas
fn combine(a: Plus, b: Plus) -> Plus {
    a.plus(b) // error: pact method `plus` can't be called through `Plus`
}
```

Call it on a concrete type instead (`fn combine(a: V, b: V) -> V`). The rule is per method, not per pact: the pact's other methods stay callable through the bound, a `Self` that appears only in the return type is fine (the result comes back as the bound), and inside a default body `self.plus(self)` works because every `Self` there is the same type.

**`Self::` doesn't resolve inside a default body.** A default body is compiled once and shared by every implementer, so `Self` there is a type that hasn't been decided yet, and there's no single constant or associated function for `Self::` to reach. `Self::CONST` and `Self::assoc_fn()` are both rejected inside one. Lowercase `self` -- fields, methods, parameters -- works normally, and `Self::` resolves as expected inside an ordinary `impl` block.

**The rest of trait machinery.** Generics are parameter-bounded only: `fn f<T: Draw>(x: T)` and `struct Box<T: Show>` dispatch at run time, not by monomorphization. There are no associated types and no blanket impls, and none are planned for 0.1.0.

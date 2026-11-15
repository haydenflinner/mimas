# Notable Exclusions

Anything not described in this reference can be assumed not to exist in mimas -- the language is intentionally small. This page calls out a few absences that programmers coming from other languages are most likely to reach for, and why they're left out.

## Pointers & references

There are no pointers or references. They're a sharp tool that invites whole categories of instability, and guarding against that safely would demand a lifetime system as involved as Rust's. Like Python, mimas's answer is not to have them. If part of your project genuinely needs that level of control, that's a sign it belongs on the Rust side. ([Memory](./memory.md) covers the managed model that replaces them.)

## Union types

Arbitrary unions (`A | B`) undercut type inference and blur the line between static and dynamic typing in ways that get unpleasant fast. Where you'd reach for a union to write code generic over several types, mimas points you at a [pact](./pacts.md) instead. The only built-in "this or that" types are [options](./options.md) and [results](./error-handling.md), which are deliberate, well-behaved special cases.

## The rest of generics

mimas does have type parameters -- `fn id<T>(x: T) -> T`, `struct Pair<A, B>`, `enum Opt<T>` -- and [pact](./pacts.md) bounds on them (`fn f<T: Draw + Vary>(x: T)`). What it leaves out is Rust's heavier end: associated types, blanket impls, and the coherence machinery they drag in. Dispatch through a bound happens at run time, not by monomorphization, which is what keeps the checker small enough to stay embeddable.

## A borrow checker

There is no borrow checker -- garbage collection removes the need for one. Trading manual memory rigor for a managed runtime is exactly what buys mimas its forgiving, script-like feel.

## A `let` / `mut` split

There's no `mut`. Without a borrow checker there's little to gain from distinguishing mutable and immutable bindings, so the model is as simple as it gets: `let` is reassignable, `const` is not, and that's all. See [Variables & Constants](./variables.md).

## First-class `fn`s

A `fn` declaration is not a value -- it can't be bound (`let g = sub` is a compile error), can't nest, and can't capture. Languages that merge `fn` and lambda into one concept (Python, Swift) pay for it by giving every named function an environment it may not need. Mimas keeps the split because the split means something: a `fn` is context-free -- it sees only its parameters and top-level items -- which makes it the portable unit in a system where code is replayed, serialized, and shared between pages. A capture is a hidden dependency that never appears in a signature.

The value use cases are already covered by [closures](./functions.md#closures), which are first-class and do capture. What remains parked, contingent on a real use case (a dispatch table of named handlers is the candidate so far), is a narrower coercion: letting a `fn` name evaluate to a non-capturing function value, with defaults padded callee-side so `f("localhost")` through a `let` still sees `port = 8080`. The wider merge -- nested `fn`, capturing `fn`, callee-side default padding everywhere -- was considered and rejected rather than deferred: a named closure is just a `let` closure already.

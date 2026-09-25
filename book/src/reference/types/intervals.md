# Intervals

An estimate isn't a point. The grant might be $24k or $36k; the sun might deliver 3.8 or 4.6 peak hours. An `Interval` carries the plausible range and a best guess together, through ordinary arithmetic:

```mimas
let grant = Interval::pm(30000.0, 0.10);   // 27000 … 33000, best guess 30000
let net = 60000.0 - grant * 1.0;           // 27000 … 33000, still an Interval
print(net.lo);                              // 27000
print(net.mid);                             // 30000
print(net.hi);                              // 33000
```

`Interval` is a built-in struct with three `float` fields: `lo`, `mid` (the best guess) and `hi`.

## Making one

| | |
| :--- | :--- |
| `Interval::pm(x, rel)` | `x` give or take a fraction: `pm(100.0, 0.1)` is 90 … 110 |
| `Interval::within(x, d)` | `x` give or take an absolute amount |
| `Interval::span(lo, hi)` | anywhere between; the best guess is the middle |
| `Interval::of(lo, mid, hi)` | an asymmetric estimate |
| `Interval::exact(x)` | no uncertainty — what a plain number becomes next to an interval |

## Arithmetic

`+ - * /` work between two intervals or an interval and a plain number, in either order. The bounds propagate by interval arithmetic and the best guess by the plain operation, so a whole model written over intervals comes out with error bars and no extra code. Unary `-` works too, and `x.pow(n)` raises a non-negative interval to a power.

```admonish warning title="Interval arithmetic is conservative"
Every input is assumed to sit at its worst *independently*, and the same estimate used twice doesn't cancel (`x - x` is not zero). Dividing by a range that includes zero gives an unbounded result. For a tighter picture, sample: `x.sample(u)` draws from the triangular distribution the three numbers describe (`u` in 0..1), which is the hook for a Monte Carlo run.
```

## Comparing

`<`, `<=`, `>` and `>=` against another interval or a plain number are **certain**: `a < b` is true only if it holds for every value in both ranges (`a.hi < b.lo`). So `!(a < b)` does *not* mean `a >= b` — the ranges may overlap. Ask `a.overlaps(b)`, or compare the `mid`s for the best guess. `==` and `!=` are structural, as for any struct.

## Units

Intervals of quantities keep their [units](./units.md): `Interval::pm(5kW, 0.1) * Interval::pm(2h, 0.1)` is an `Interval<kWh>`, and adding it to a time is a compile error.

## Other methods

`x.width()` is `hi - lo`; `x.contains(v)` asks whether `v` is inside the range.

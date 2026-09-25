# Units

Numbers with units keep their meaning. Write a unit straight after a number and mimas checks that the arithmetic makes sense:

```mimas
const LOAD = 25kW;                       // power
const HOURS = 8h;                        // time
let energy = LOAD * HOURS;               // 25kW * 8h is energy: 200 kWh
let bill = energy * 0.14usd/kWh;         // money
let oops = LOAD + HOURS;                 // compile error: cannot add power (W) and time (s)
```

Nothing changes at run time. A quantity is an ordinary `float` kept in base units -- `25kW` is `25000.0`, watts -- so there's no speed cost and no new value type. What you get is a checker that follows each number's *dimension* through your program and refuses what doesn't add up.

## Writing units

A unit is a name glued to a number: `25kW`, `3.5m`, `90deg`, `6pct`, `0.14usd/kWh`, `9.81m/s^2`. A unit with a `/` divides, and `^n` raises to a power. There are no spaces: `25 kW` is the number 25 followed by a variable called `kW`.

Where a type goes, the same names mean "a float measured in this":

```mimas
fn cost(energy: kWh, price: usd/kWh) -> usd {
    energy * price
}

struct Plan { load: kW, tariff: usd/kWh }
```

If your program declares a type with the same name as a unit (a `struct W`, say), yours wins in type position.

## The units

| Kind | Units |
| :--- | :--- |
| length | `m` `km` `cm` `mm` `inch` `ft` `mi` |
| mass | `kg` `g` `mg` `lb` `tonne` |
| time | `s` `ms` `us` `min` `h` `d` `wk` `yr` |
| electricity | `A` `mA` `V` `kV` `C` |
| mechanics | `Hz` `kHz` `N` `kN` `Pa` `kPa` |
| energy | `J` `kJ` `MJ` `Wh` `kWh` `MWh` |
| power | `W` `mW` `kW` `MW` |
| other SI | `K` `mol` `cd` |
| money | `usd` |
| graphics | `px` `frame` (so `px/frame` is a speed in pixels per frame) |
| plain numbers | `rad` `deg` `pct` (`90deg` is `1.5708`, `6pct` is `0.06`) |

Money and the graphics units are dimensions of their own: `usd` can't be added to `s`, and `px` isn't a length.

## What the checker enforces

* `+`, `-`, `%` and comparisons need the same dimension on both sides (comparing against a literal `0` is always fine).
* `*` and `/` combine dimensions: `kW * h` is energy, `usd / kWh` is a price, `m / s` is a speed.
* A plain `float` (or `int`) has **no** unit. It scales anything -- `2.0 * 5kW` -- but is not itself a power: passing `5kW` to a `float` parameter, or writing `let p: float = 5kW`, is an error. That's the point: a parameter that says `float` while you hand it kilowatts is exactly the bug.
* `.abs()`, `.round()`, `.min(q)`, `.max(q)` and `.clamp(a, b)` keep the dimension; `.sqrt()` halves it (all exponents must be even); `.pow(n)` scales it; `.sin()`, `.exp()` and friends need a plain number.
* Struct fields, function parameters, return types and list elements (`[kW]`) take unit annotations, and the dimension follows values through them.

Anything the checker can't follow -- what a native function returns, an unannotated closure -- is simply *unknown*, and unknown never causes an error. So code without units is unaffected, and you can adopt them one function at a time.

## Intervals and columns keep their units

An `Interval` of quantities is an interval *in that unit*. `Interval::pm(30000.0usd, 0.1)` is a range of dollars; `+ - * /` carry the unit through it like they do for a single number, and `.lo`, `.mid`, `.hi` come out as plain quantities. Write the type as `Interval<usd>`:

```mimas
fn payback(net: Interval<usd>, saved: Interval<usd>) -> Interval<yr> {
    net / saved * 1yr
}
```

Table columns are bare numbers until you say what they measure. `df.pull_as("kwh", "kWh")` reads a numeric column as a list of quantities in that unit (scaled to base units, like a literal), so `let load: [kWh] = df.pull_as("kwh", "kWh")!` is checked from there on. In the literate environment a data cell's header can carry the unit -- `month,kwh (kWh)` -- and the page gets a typed accessor for it.

## Reading a number back

`q.to("unit")` turns a quantity into a plain float in that unit, and the compiler checks the unit measures the same kind of thing:

```mimas
let e = 25kW * 4h;
print(e.to("kWh"));   // 100
print(e.to("kW"));    // compile error: cannot express energy (J) in kW
```

Printing a quantity directly shows its base-unit value (`25kW` prints `25000`), so convert with `to` when you want to show one.

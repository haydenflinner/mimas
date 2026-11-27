use api::Intrinsic;
use macros::native;
use shared::Ty;
use vm::{Array, Ctx, RtErr, Val, anon, api::Api};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    api.add_assoc(Ty::array(Ty::Anon(0)), new);
    api.add_assoc(Ty::array(Ty::Anon(0)), new_filled);
    let id = api.add_method(len);
    api.mark_intrinsic(id, Intrinsic::Len);
    let id = api.add_method(contains);
    api.mark_intrinsic(id, Intrinsic::In);
    let id = api.add_method(push);
    api.mark_intrinsic(id, Intrinsic::Push);
    api.add_method(pop);
    api.add_method(remove);
    api.add_method(insert);
    api.add_method(get);
    api.add_method(shuffle);
    // `xs + x`/`xs - x`/`xs + ys`/`xs - ys` — the Z-set surface of `+`/`-`
    // on sequences. `add`/`sub` take an element (append / drop-first);
    // `union`/`minus` take a same-element list (concat / drop-each) and
    // are what `xs + ys`/`xs - ys` dispatch to when `add`/`sub` don't fit.
    api.add_method(add);
    api.add_method(union);
    api.add_method(sub);
    api.add_method(minus);
    api.add_method(extend);
    api.add_method(enumerate);
    api.add_method(flatten);
    api.add_method(choose);
    api.add_method(join);
    api.add_method(is_empty);
    api.add_method_named("max", max_int);
    api.add_method_named("min", min_int);
    api.add_method_named("max", max_float);
    api.add_method_named("min", min_float);
    api.add_method_named("sum", sum_int);
    api.add_method_named("sum", sum_float);
    // the reductions behind `∑ ∏ ⨅ ⨆`: `product`, and meet/join -- `inf` is
    // all-of for bools and min for numbers, `sup` any-of and max
    // `a ⋅ b` on two lists: the dot product (r[impl mimas.glyph.dot.list])
    api.add_method_named("dot", dot_int);
    api.add_method_named("dot", dot_float);
    api.add_method_named("product", product_int);
    api.add_method_named("product", product_float);
    api.add_method_named("inf", inf_bool);
    api.add_method_named("inf", inf_int);
    api.add_method_named("inf", inf_float);
    api.add_method_named("sup", sup_bool);
    api.add_method_named("sup", sup_int);
    api.add_method_named("sup", sup_float);
    // on a list of lists each reduction eats the innermost axis -- one answer per
    // row -- so `⨆⨅ (b[LINES] == p)` meets each line, then joins the lines
    api.add_method_named("sum", sum_int_rows);
    api.add_method_named("sum", sum_float_rows);
    api.add_method_named("product", product_int_rows);
    api.add_method_named("product", product_float_rows);
    api.add_method_named("inf", inf_bool_rows);
    api.add_method_named("inf", inf_int_rows);
    api.add_method_named("inf", inf_float_rows);
    api.add_method_named("sup", sup_bool_rows);
    api.add_method_named("sup", sup_int_rows);
    api.add_method_named("sup", sup_float_rows);
    api.add_method_named("argsort", argsort_int);
    api.add_method_named("argsort", argsort_float);
    api.add_method(reorder);
    // r[impl mimas.slice]
    api.add_method(slice);
    api.add_method(slice_from);
    // r[impl mimas.sorted]
    api.add_method_named("sorted", sorted_int);
    api.add_method(reversed);
    api.add_method(shuffled);
    api.add_method(reverse);
    api.add_method_named("sorted", sorted_float);
    api.add_method_named("sorted", sorted_str);
    api.add_method(zip);
    // higher-order: signatures only -- each lowers to a generated loop that calls the
    // closure itself (a `#[native]` can't call back into the VM). see emit_intrinsic.
    let id = api.add_method(map);
    api.mark_intrinsic(id, Intrinsic::Map);
    let id = api.add_method(filter);
    api.mark_intrinsic(id, Intrinsic::Filter);
    let id = api.add_method(fold);
    api.mark_intrinsic(id, Intrinsic::Fold);
    let id = api.add_method(find);
    api.mark_intrinsic(id, Intrinsic::Find);
    let id = api.add_method(any);
    api.mark_intrinsic(id, Intrinsic::Any);
    let id = api.add_method(all);
    api.mark_intrinsic(id, Intrinsic::All);
    let id = api.add_method(flat_map);
    api.mark_intrinsic(id, Intrinsic::FlatMap);
    let id = api.add_method(mapi);
    api.mark_intrinsic(id, Intrinsic::MapI);
    let id = api.add_method(foldi);
    api.mark_intrinsic(id, Intrinsic::FoldI);
}

#[native]
/// Creates an empty array. This is the same as writing `[]`, and like `[]` it needs a type
/// annotation if nothing else tells the compiler what it will hold.
///
/// ```mimas
/// let xs: [int] = array::new();
/// xs.push(1); // xs is now [1]
/// ```
fn new<'gc>() -> Vec<anon::T<'gc>> {
    Vec::new()
}

#[native]
/// Creates an array of `len` elements, each a copy of `value`. The copies are deep, so filling
/// with an array gives each slot its own array.
///
/// ```mimas
/// let grid = array::new_filled([0, 0], 2);
/// grid[0].push(1);
/// // grid is [[0, 0, 1], [0, 0]]
/// ```
fn new_filled<'gc>(ctx: Ctx<'gc>, val: anon::T<'gc>, len: i64) -> Result<Vec<anon::T<'gc>>, RtErr> {
    let len = len.max(0) as usize;
    ctx.check_alloc_n(len, std::mem::size_of::<Val>())?;
    Ok((0..len)
        .map(|_| anon::Anon(ctx.deep_clone(val.0)))
        .collect())
}

#[native]
/// Returns the number of elements.
///
/// ```mimas
/// let n = [3, 1, 2].len(); // 3
/// ```
fn len(_arr: &[Val<'gc>]) -> usize {
    unreachable!("intrinsics cannot be reached")
}

#[native]
/// Returns whether any element equals `value`. This is the same check as the
/// [`in` operator](../reference/collections/in-expressions.md).
///
/// ```mimas
/// let xs = [1, 2, 3];
/// let a = xs.contains(2); // true
/// let b = 2 in xs;        // true
/// ```
fn contains(_arr: &[anon::T<'gc>], _val: anon::T<'gc>) -> bool {
    unreachable!("intrinsics cannot be reached")
}

#[native]
/// Appends `value` to the end of the array.
///
/// ```mimas
/// let xs = [1, 2];
/// xs.push(3); // xs is now [1, 2, 3]
/// ```
fn push(_arr: &mut Vec<anon::T<'gc>>, _val: anon::T<'gc>) {
    unreachable!("intrinsics cannot be reached")
}

#[native]
/// Removes the last element and returns it, or returns `null` if the array is empty.
///
/// ```mimas
/// let xs = [1, 2];
/// let last = xs.pop(); // 2, and xs is now [1]
/// ```
fn pop(arr: &mut Vec<anon::T<'gc>>) -> Option<anon::T<'gc>> {
    arr.pop()
}

#[native]
#[effects(rng)]
/// Puts the elements in a random order.
///
/// ```mimas
/// let deck = [1, 2, 3, 4];
/// deck.shuffle();
/// ```
fn shuffle<'gc>(ctx: Ctx<'gc>, arr: &mut Vec<anon::T<'gc>>) {
    // Fisher-Yates on the Vm's stream — the same draws `mrt::shuffle`
    // makes, so a seeded run shuffles identically either way
    for i in (1..arr.len()).rev() {
        let j = (super::rng::unit(ctx) * (i as f64 + 1.0)) as usize;
        arr.swap(i, j.min(i));
    }
}

#[native]
/// `xs + x` — one more `x`, as a new array (`xs.push` is the in-place
/// form). This is the Z-set's +1 weight spelled arithmetically; `xs +
/// ys` below is the union.
///
/// ```mimas
/// let xs = [1, 2] + 3; // [1, 2, 3]
/// ```
fn add<'gc>(
    ctx: Ctx<'gc>,
    arr: &[anon::T<'gc>],
    val: anon::T<'gc>,
) -> Result<Vec<anon::T<'gc>>, RtErr> {
    ctx.check_alloc_n(arr.len() + 1, std::mem::size_of::<Val>())?;
    ctx.charge(std::mem::size_of::<Val>());
    let mut out = arr.to_vec();
    out.push(val);
    Ok(out)
}

// `other` stays as the gc handle like `extend` — `xs + xs` would
// double-borrow otherwise, and unioning a list with itself just doubles it.
#[native]
/// `xs + ys` — the union: every `y` after every `x`, one new array.
///
/// ```mimas
/// let xs = [1, 2] + [3, 4]; // [1, 2, 3, 4]
/// ```
fn union<'gc>(
    ctx: Ctx<'gc>,
    arr: &[anon::T<'gc>],
    other: Array<'gc>,
) -> Result<Vec<anon::T<'gc>>, RtErr> {
    let mut out = arr.to_vec();
    let more: Vec<anon::T<'gc>> = match other.0.try_borrow() {
        Ok(v) => v.iter().map(|x| anon::Anon(*x)).collect(),
        Err(_) => out.clone(),
    };
    ctx.check_alloc_n(
        out.len().saturating_add(more.len()),
        std::mem::size_of::<Val>(),
    )?;
    ctx.charge(more.len() * std::mem::size_of::<Val>());
    out.extend(more);
    Ok(out)
}

#[native]
/// `xs - x` — the Z-set's -1: drops the first element `== x` (a quiet
/// no-op when `x` isn't there; `{int}` Z-sets can go negative, arrays
/// can't). `xs - ys` below removes each `y` in turn.
///
/// ```mimas
/// let xs = [1, 2, 1] - 1; // [2, 1]
/// ```
fn sub<'gc>(_ctx: Ctx<'gc>, arr: &[anon::T<'gc>], val: anon::T<'gc>) -> Vec<anon::T<'gc>> {
    let mut out = arr.to_vec();
    if let Some(i) = out.iter().position(|v| v.0 == val.0) {
        out.remove(i);
    }
    out
}

#[native]
/// `xs - ys` — drops the first `==` match of each `y`, in order.
/// `xs - xs` empties.
///
/// ```mimas
/// let xs = [1, 2, 3] - [3, 9]; // [1, 2]
/// ```
fn minus<'gc>(_ctx: Ctx<'gc>, arr: &[anon::T<'gc>], other: Array<'gc>) -> Vec<anon::T<'gc>> {
    let mut out = arr.to_vec();
    match other.0.try_borrow() {
        Ok(v) => {
            for d in v.iter() {
                if let Some(i) = out.iter().position(|x| x.0 == *d) {
                    out.remove(i);
                }
            }
        }
        // `xs - xs`: every element is a match — the list empties.
        Err(_) => out.clear(),
    }
    out
}

// `other` stays as the gc handle because we already hold a borrow on `arr`. if `other`
// also went through auto-borrow we'd hold two borrows on the same RefLock if the user
// happens to call `arr.extend(arr)`
#[native]
/// Appends every element of `other`, in order. `other` is unchanged.
///
/// ```mimas
/// let xs = [1, 2];
/// xs.extend([3, 4]); // xs is now [1, 2, 3, 4]
/// ```
fn extend<'gc>(ctx: Ctx<'gc>, arr: &mut Vec<Val<'gc>>, other: Array<'gc>) -> Result<(), RtErr> {
    let copy: Vec<Val<'gc>> = match other.0.try_borrow() {
        Ok(v) => v.iter().copied().collect(),
        Err(_) => {
            // presumably the user has tried to extend this array with itself, which is kind of
            // nuts, but technically legal as far as mimas is concerned.
            arr.clone()
        }
    };
    ctx.check_alloc_n(
        arr.len().saturating_add(copy.len()),
        std::mem::size_of::<Val>(),
    )?;
    ctx.charge(copy.len() * std::mem::size_of::<Val>());
    arr.extend(copy);
    Ok(())
}

#[native]
/// Returns a new array pairing each element with its index.
///
/// ```mimas
/// for (i, name) in ["ant", "bee"].enumerate() {
///     print(f"{i}: {name}"); // 0: ant, then 1: bee
/// }
/// ```
fn enumerate(arr: &[anon::T<'gc>]) -> Vec<(usize, anon::T<'gc>)> {
    arr.iter().enumerate().map(|(i, v)| (i, *v)).collect()
}

// todo, should be "flat" or "flattened"
#[native]
/// Returns a new array with the elements of each inner array, in order. Only one level is
/// removed: a `[[[int]]]` flattens to a `[[int]]`.
///
/// ```mimas
/// let xs = [[1, 2], [], [3]].flatten(); // [1, 2, 3]
/// ```
fn flatten(arr: Vec<Vec<anon::T<'gc>>>) -> Vec<anon::T<'gc>> {
    arr.into_iter().flatten().collect()
}

#[native]
#[effects(rng)]
/// Returns a random element, or `null` if the array is empty.
///
/// ```mimas
/// let loot = ["sword", "shield", "potion"];
/// let drop = loot.choose() ?? "nothing";
/// ```
fn choose<'gc>(ctx: Ctx<'gc>, arr: &[Val<'gc>]) -> Option<anon::T<'gc>> {
    if arr.is_empty() {
        return None;
    }
    let i = (super::rng::unit(ctx) * arr.len() as f64) as usize;
    Some(anon::Anon(arr[i.min(arr.len() - 1)]))
}

#[native]
/// Returns the strings joined together with `separator` between each one. Only `[str]` has
/// `join`, so convert other arrays first:
///
/// ```mimas
/// let a = ["a", "b", "c"].join(", "); // "a, b, c"
/// let b = (for n in [1, 2, 3] collect n.to_str()).join("-"); // "1-2-3"
/// ```
fn join(arr: &[&str], sep: &str) -> String {
    let parts: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
    parts.join(sep)
}

#[native]
/// Returns whether the array has no elements.
///
/// ```mimas
/// let xs: [int] = [];
/// let empty = xs.is_empty(); // true
/// ```
fn is_empty(arr: &[Val<'gc>]) -> bool {
    arr.is_empty()
}

#[native]
/// Returns the maximum value present in the array.
///
/// ```mimas
/// let a: [int] = [0, 1, 2];
/// let int_max = a.max(); // 2
///
/// let b: [float] = [0.0, 1.0, 2.0];
/// let float_max = b.max(); // 2.0
/// ```
fn max_int(arr: &[i64]) -> Option<i64> {
    arr.iter().max().copied()
}

#[native]
/// Returns the minimum value present in the array.
///
/// ```mimas
/// let a: [int] = [0, 1, 2];
/// let int_min = a.min(); // 0
///
/// let b: [float] = [0.0, 1.0, 2.0];
/// let float_min = b.min(); // 0.0
/// ```
fn min_int(arr: &[i64]) -> Option<i64> {
    arr.iter().min().copied()
}

#[native]
/// Returns the maximum value present in the array.
///
/// ```mimas
/// let a: [int] = [0, 1, 2];
/// let int_max = a.max(); // 2
///
/// let b: [float] = [0.0, 1.0, 2.0];
/// let float_max = b.max(); // 2.0
/// ```
fn max_float(arr: &[f64]) -> Option<f64> {
    arr.iter().copied().max_by(f64::total_cmp)
}

#[native]
/// Returns the minimum value present in the array.
///
/// ```mimas
/// let a: [int] = [0, 1, 2];
/// let int_min = a.min(); // 0
///
/// let b: [float] = [0.0, 1.0, 2.0];
/// let float_min = b.min(); // 0.0
/// ```
fn min_float(arr: &[f64]) -> Option<f64> {
    arr.iter().copied().min_by(f64::total_cmp)
}

#[native]
/// Returns the sum of all values in the array.
///
/// ```mimas
/// let a: [int] = [0, 1, 2];
/// let int_sum = a.sum(); // 3
///
/// let b: [float] = [0.0, 1.0, 2.0];
/// let float_sum = b.sum(); // 3.0
/// ```
fn sum_int(arr: &[i64]) -> i64 {
    arr.iter().sum()
}

#[native]
/// Returns the sum of all values in the array.
///
/// ```mimas
/// let a: [int] = [0, 1, 2];
/// let int_sum = a.sum(); // 3
///
/// let b: [float] = [0.0, 1.0, 2.0];
/// let float_sum = b.sum(); // 3.0
/// ```
fn sum_float(arr: &[f64]) -> f64 {
    crate::fsum(arr.iter().copied())
}


// the indices that would sort `keys` ascending. dispatched on the receiver (int/float key array),
// so each arm is its own typed overload; always returns `[int]` positions.
#[native]
/// Compares all elements in the array and returns a new array with the indicies sorted. For
/// example, if the maximum value in this array is at index 3, the first element of the returned
/// array will be `3`.
///
/// ```mimas
/// let a = [5, 0, 2, 4];
/// let a_sorted = [0, 3, 2, 1];
///
/// let b = [5.0, 0.0, 2.0, 4.0];
/// let b_sorted = [0, 3, 2, 1];
/// ```
fn argsort_int(keys: &[i64]) -> Vec<i64> {
    let mut idx: Vec<i64> = (0..keys.len() as i64).collect();
    idx.sort_by_key(|&i| keys[i as usize]);
    idx
}

#[native]
/// Compares all elements in the array and returns a new array with the indicies sorted. For
/// example, if the maximum value in this array is at index 3, the first element of the returned
/// array will be `3`.
///
/// ```mimas
/// let a = [5, 0, 2, 4];
/// let a_sorted = [0, 3, 2, 1];
///
/// let b = [5.0, 0.0, 2.0, 4.0];
/// let b_sorted = [0, 3, 2, 1];
/// ```
fn argsort_float(keys: &[f64]) -> Vec<i64> {
    let mut idx: Vec<i64> = (0..keys.len() as i64).collect();
    idx.sort_by(|&a, &b| keys[a as usize].total_cmp(&keys[b as usize]));
    idx
}

/// `xs.map(f)` -> `[U]` -- `f: (T) -> U` applied element-wise.
#[native]
fn map<'gc>(
    _arr: &[anon::T<'gc>],
    _f: anon::Fn1<'gc, anon::T<'gc>, anon::U<'gc>>,
) -> Vec<anon::U<'gc>> {
    unreachable!("intrinsics cannot be reached")
}

/// `xs.filter(f)` -> `[T]` -- keeps the elements `f` returns `true` for.
#[native]
fn filter<'gc>(_arr: &[anon::T<'gc>], _f: anon::Fn1<'gc, anon::T<'gc>, bool>) -> Vec<anon::T<'gc>> {
    unreachable!("intrinsics cannot be reached")
}

/// `xs.fold(init, f)` -> `U` -- `f: (U, T) -> U` threads an accumulator left to right.
#[native]
fn fold<'gc>(
    _arr: &[anon::T<'gc>],
    _init: anon::U<'gc>,
    _f: anon::Fn2<'gc, anon::U<'gc>, anon::T<'gc>, anon::U<'gc>>,
) -> anon::U<'gc> {
    unreachable!("intrinsics cannot be reached")
}

/// `xs.find(f)` -> `T?` -- the first element `f` returns `true` for, else `null`.
#[native]
fn find<'gc>(
    _arr: &[anon::T<'gc>],
    _f: anon::Fn1<'gc, anon::T<'gc>, bool>,
) -> Option<anon::T<'gc>> {
    unreachable!("intrinsics cannot be reached")
}

/// `xs.any(f)` -> `bool` -- whether `f` returns `true` for any element.
#[native]
fn any<'gc>(_arr: &[anon::T<'gc>], _f: anon::Fn1<'gc, anon::T<'gc>, bool>) -> bool {
    unreachable!("intrinsics cannot be reached")
}

/// `xs.all(f)` -> `bool` -- whether `f` returns `true` for every element.
#[native]
fn all<'gc>(_arr: &[anon::T<'gc>], _f: anon::Fn1<'gc, anon::T<'gc>, bool>) -> bool {
    unreachable!("intrinsics cannot be reached")
}

/// `xs.flat_map(f)` -> `[U]` -- `f: (T) -> [U]`, each element of each result appended in order.
#[native]
fn flat_map<'gc>(
    _arr: &[anon::T<'gc>],
    _f: anon::Fn1<'gc, anon::T<'gc>, Vec<anon::U<'gc>>>,
) -> Vec<anon::U<'gc>> {
    unreachable!("intrinsics cannot be reached")
}

/// `xs.mapi(f)` -> `[U]` -- `f: (int, T) -> U` gets each element's index.
#[native]
fn mapi<'gc>(
    _arr: &[anon::T<'gc>],
    _f: anon::Fn2<'gc, i64, anon::T<'gc>, anon::U<'gc>>,
) -> Vec<anon::U<'gc>> {
    unreachable!("intrinsics cannot be reached")
}

/// `xs.foldi(init, f)` -> `U` -- `f: (int, U, T) -> U` threads an accumulator with the index.
#[native]
fn foldi<'gc>(
    _arr: &[anon::T<'gc>],
    _init: anon::U<'gc>,
    _f: anon::Fn3<'gc, i64, anon::U<'gc>, anon::T<'gc>, anon::U<'gc>>,
) -> anon::U<'gc> {
    unreachable!("intrinsics cannot be reached")
}

/// `xs.zip(ys)` -> `[(T, U)]` -- index-paired elements, truncated to the shorter array.
#[native]
fn zip<'gc>(a: &[anon::T<'gc>], b: &[anon::U<'gc>]) -> Vec<(anon::T<'gc>, anon::U<'gc>)> {
    a.iter().copied().zip(b.iter().copied()).collect()
}

#[native]
/// Replaces the array's contents with the elements at `indices`, in that order. The indices don't
/// have to cover every element once: the array ends up as long as `indices`, and an index can
/// repeat.
///
/// An index that is negative or past the end is a runtime error, and the array is left unchanged.
///
/// ```mimas
/// let xs = ["a", "b", "c"];
/// xs.reorder([2, 0, 1]); // xs is now ["c", "a", "b"]
/// xs.reorder([0, 0]);    // xs is now ["c", "c"]
/// ```
fn reorder<'gc>(ctx: Ctx<'gc>, arr: &mut Vec<Val<'gc>>, perm: Vec<i64>) -> Result<(), RtErr> {
    let out = perm
        .iter()
        .map(|&i| {
            usize::try_from(i)
                .ok()
                .and_then(|u| arr.get(u).copied())
                .ok_or(RtErr::IndexOutOfBounds)
        })
        .collect::<Result<Vec<_>, _>>()?;
    *arr = out;
    Ok(())
}

/// `n` clamped into `0..=len`, as Python clamps a slice's bounds.
fn clamp_index(n: i64, len: usize) -> usize {
    n.clamp(0, len as i64) as usize
}

#[native]
/// The elements from position `a` up to, not including, `b`. `xs[a..b]`
/// is this. Bounds past either end are clamped, so `[1, 2, 3].slice(1, 99)`
/// is `[2, 3]`, and a slice that runs backward is empty.
///
/// ```mimas
/// let xs = [10, 20, 30, 40];
/// let mid = xs[1..3]; // [20, 30]
/// ```
fn slice(arr: Vec<anon::T<'gc>>, a: i64, b: i64) -> Vec<anon::T<'gc>> {
    let (a, b) = (clamp_index(a, arr.len()), clamp_index(b, arr.len()));
    if a >= b { Vec::new() } else { arr[a..b].to_vec() }
}

#[native]
/// The elements from position `a` to the end. `xs[a..]` is this.
///
/// ```mimas
/// let rest = [10, 20, 30][1..]; // [20, 30]
/// ```
fn slice_from(arr: Vec<anon::T<'gc>>, a: i64) -> Vec<anon::T<'gc>> {
    let a = clamp_index(a, arr.len());
    arr[a..].to_vec()
}

#[native]
/// A copy of the list, last item first. The list itself is unchanged.
///
/// ```mimas
/// let xs = [1, 2, 3].reversed(); // [3, 2, 1]
/// ```
// r[impl mimas.list.reversed]
fn reversed(arr: Vec<anon::T<'gc>>) -> Vec<anon::T<'gc>> {
    let mut out = arr;
    out.reverse();
    out
}

#[native]
/// Reverses the list in place, last item first.
///
/// ```mimas
/// let xs = [1, 2, 3];
/// xs.reverse(); // xs is now [3, 2, 1]
/// ```
// r[impl mimas.list.reversed]
fn reverse(arr: &mut Vec<anon::T<'gc>>) {
    arr.reverse();
}

#[native]
/// A sorted copy of the list, smallest first. The list itself is unchanged.
///
/// ```mimas
/// let xs = [3, 1, 2].sorted(); // [1, 2, 3]
/// ```
fn sorted_int(arr: &[i64]) -> Vec<i64> {
    let mut out = arr.to_vec();
    out.sort();
    out
}

#[native]
/// A sorted copy of the list, smallest first, `NaN` last.
fn sorted_float(arr: &[f64]) -> Vec<f64> {
    let mut out = arr.to_vec();
    out.sort_by(|a, b| a.total_cmp(b));
    out
}

#[native]
/// A sorted copy of the list, in dictionary order.
fn sorted_str(arr: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = arr.iter().map(|s| s.to_string()).collect();
    out.sort();
    out
}

#[native]
/// The product of every value in the array (`∏ xs`); `1` for an empty one.
fn product_int(arr: &[i64]) -> Result<i64, RtErr> {
    arr.iter()
        .try_fold(1i64, |acc, &x| acc.checked_mul(x))
        .ok_or_else(|| RtErr::Custom("integer product overflowed".into()))
}

#[native]
/// The dot product of two lists of one length, `a ⋅ b`: the sum of the
/// products of matching items. Lists of different lengths are an error.
///
/// ```mimas
/// let d = [1.0, 2.0] ⋅ [3.0, 4.0]; // 11.0
/// ```
fn dot_float(arr: &[f64], other: Vec<f64>) -> Result<f64, RtErr> {
    if arr.len() != other.len() {
        return Err(RtErr::Custom(format!(
            "`⋅` needs two lists of one length, found {} and {}",
            arr.len(),
            other.len()
        )));
    }
    Ok(crate::fsum(arr.iter().zip(&other).map(|(a, b)| a * b)))
}

#[native]
/// The dot product of two int lists of one length, `a ⋅ b`.
fn dot_int(arr: &[i64], other: Vec<i64>) -> Result<i64, RtErr> {
    if arr.len() != other.len() {
        return Err(RtErr::Custom(format!(
            "`⋅` needs two lists of one length, found {} and {}",
            arr.len(),
            other.len()
        )));
    }
    arr.iter()
        .zip(&other)
        .try_fold(0i64, |acc, (a, b)| acc.checked_add(a.checked_mul(*b)?))
        .ok_or_else(|| RtErr::Custom("integer dot product overflowed".into()))
}

#[native]
/// The product of every value in the array (`∏ xs`); `1.0` for an empty one.
fn product_float(arr: &[f64]) -> f64 {
    arr.iter().product()
}

#[native]
/// `⨅ xs` on bools -- the meet: every value is `true` (`true` for an empty list).
fn inf_bool(arr: &[bool]) -> bool {
    arr.iter().all(|&b| b)
}

#[native]
/// `⨅ xs` on ints -- the meet: the minimum, `null` for an empty list.
fn inf_int(arr: &[i64]) -> Option<i64> {
    arr.iter().min().copied()
}

#[native]
/// `⨅ xs` on floats -- the meet: the minimum, `null` for an empty list.
fn inf_float(arr: &[f64]) -> Option<f64> {
    arr.iter().copied().reduce(f64::min)
}

#[native]
/// `⨆ xs` on bools -- the join: some value is `true` (`false` for an empty list).
fn sup_bool(arr: &[bool]) -> bool {
    arr.iter().any(|&b| b)
}

#[native]
/// `⨆ xs` on ints -- the join: the maximum, `null` for an empty list.
fn sup_int(arr: &[i64]) -> Option<i64> {
    arr.iter().max().copied()
}

#[native]
/// `⨆ xs` on floats -- the join: the maximum, `null` for an empty list.
fn sup_float(arr: &[f64]) -> Option<f64> {
    arr.iter().copied().reduce(f64::max)
}

#[native]
/// `∑ xss` on a list of lists -- each row's sum.
fn sum_int_rows(arr: Vec<Vec<i64>>) -> Vec<i64> {
    arr.iter().map(|r| r.iter().sum()).collect()
}

#[native]
/// `∑ xss` on a list of float lists -- each row's sum.
fn sum_float_rows(arr: Vec<Vec<f64>>) -> Vec<f64> {
    arr.iter().map(|r| r.iter().sum()).collect()
}

#[native]
/// `∏ xss` on a list of lists -- each row's product.
fn product_int_rows(arr: Vec<Vec<i64>>) -> Result<Vec<i64>, RtErr> {
    arr.iter()
        .map(|r| {
            r.iter()
                .try_fold(1i64, |acc, &x| acc.checked_mul(x))
                .ok_or_else(|| RtErr::Custom("integer product overflowed".into()))
        })
        .collect()
}

#[native]
/// `∏ xss` on a list of float lists -- each row's product.
fn product_float_rows(arr: Vec<Vec<f64>>) -> Vec<f64> {
    arr.iter().map(|r| r.iter().product()).collect()
}

#[native]
/// `⨅ xss` on bool rows -- each row's meet (all).
fn inf_bool_rows(arr: Vec<Vec<bool>>) -> Vec<bool> {
    arr.iter().map(|r| r.iter().all(|&b| b)).collect()
}

#[native]
/// `⨅ xss` on int rows -- each row's minimum (`null` for an empty row).
fn inf_int_rows(arr: Vec<Vec<i64>>) -> Vec<Option<i64>> {
    arr.iter().map(|r| r.iter().min().copied()).collect()
}

#[native]
/// `⨅ xss` on float rows -- each row's minimum (`null` for an empty row).
fn inf_float_rows(arr: Vec<Vec<f64>>) -> Vec<Option<f64>> {
    arr.iter().map(|r| r.iter().copied().reduce(f64::min)).collect()
}

#[native]
/// `⨆ xss` on bool rows -- each row's join (any).
fn sup_bool_rows(arr: Vec<Vec<bool>>) -> Vec<bool> {
    arr.iter().map(|r| r.iter().any(|&b| b)).collect()
}

#[native]
/// `⨆ xss` on int rows -- each row's maximum (`null` for an empty row).
fn sup_int_rows(arr: Vec<Vec<i64>>) -> Vec<Option<i64>> {
    arr.iter().map(|r| r.iter().max().copied()).collect()
}

#[native]
/// `⨆ xss` on float rows -- each row's maximum (`null` for an empty row).
fn sup_float_rows(arr: Vec<Vec<f64>>) -> Vec<Option<f64>> {
    arr.iter().map(|r| r.iter().copied().reduce(f64::max)).collect()
}

/// Removes one slot and shifts the following elements left.
// r[impl mimas.place.list-remove]
#[native]
fn remove<'gc>(ctx: Ctx<'gc>, arr: anon::SeqOf<'gc, 0>, index: i64) -> Result<anon::T<'gc>, RtErr> {
    let index = usize::try_from(index).map_err(|_| RtErr::IndexOutOfBounds)?;
    arr.0.seq_remove(ctx, index).map(anon::Anon)
}

// `SeqOf` carries a shared handle, so the Rust signature cannot express
// receiver mutation with `&mut`. Keep the solver's iteration guard explicit.
vm::inventory::submit! {
    vm::api::NativeMutates { path: concat!(module_path!(), "::remove"), index: 0 }
}

/// Inserts `value` before slot `index` — `xs.insert(xs.len(), v)` is
/// `xs.push(v)` — shifting the following elements right.
// r[impl mimas.place.list-insert]
#[native]
fn insert<'gc>(
    ctx: Ctx<'gc>,
    arr: anon::SeqOf<'gc, 0>,
    index: i64,
    value: anon::T<'gc>,
) -> Result<(), RtErr> {
    let index = usize::try_from(index).map_err(|_| RtErr::IndexOutOfBounds)?;
    arr.0.seq_insert(ctx, index, value.0)
}

vm::inventory::submit! {
    vm::api::NativeMutates { path: concat!(module_path!(), "::insert"), index: 0 }
}

// r[impl mimas.list.get]
#[native]
#[effects()]
fn get<'gc>(arr: anon::SeqOf<'gc, 0>, index: i64) -> Option<anon::T<'gc>> {
    usize::try_from(index).ok().and_then(|i| arr.0.seq_get(i)).map(anon::Anon)
}

#[native]
#[effects(rng)]
/// A copy of the list in a random order. The list itself is unchanged
/// (`shuffle` mixes it in place).
// r[impl mimas.list.shuffled]
fn shuffled<'gc>(ctx: Ctx<'gc>, arr: Vec<anon::T<'gc>>) -> Vec<anon::T<'gc>> {
    let mut out = arr;
    for i in (1..out.len()).rev() {
        let j = (super::rng::unit(ctx) * (i as f64 + 1.0)) as usize;
        out.swap(i, j.min(i));
    }
    out
}

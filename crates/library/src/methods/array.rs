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
    api.add_method(shuffle);
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
    api.add_method(sort_by_int);
    api.add_method(sort_by_float);
    api.add_method_named("argsort", argsort_int);
    api.add_method_named("argsort", argsort_float);
    api.add_method(reorder);
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
fn new_filled<'gc>(ctx: Ctx<'gc>, val: anon::T<'gc>, len: i64) -> Vec<anon::T<'gc>> {
    (0..len.max(0))
        .map(|_| anon::Anon(ctx.deep_clone(val.0)))
        .collect()
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
fn extend(arr: &mut Vec<Val<'gc>>, other: Array<'gc>) {
    let copy: Vec<Val<'gc>> = match other.0.try_borrow() {
        Ok(v) => v.iter().copied().collect(),
        Err(_) => {
            // presumably the user has tried to extend this array with itself, which is kind of
            // nuts, but technically legal as far as mimas is concerned.
            arr.clone()
        }
    };
    arr.extend(copy);
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
    arr.iter().sum()
}

#[native]
/// Sorts the array in ascending order of `keys`, where `keys[i]` is the key for element `i`. Equal
/// keys keep their original order. `keys` is unchanged.
///
/// ```mimas
/// let names = ["cat", "ant", "bee"];
/// names.sort_by_int([3, 1, 2]); // names is now ["ant", "bee", "cat"]
/// ```
///
/// `keys` should be as long as the array. An element without a key is dropped from the array, and
/// extra keys are ignored.
fn sort_by_int(arr: &mut Vec<anon::T<'gc>>, keys: Vec<i64>) {
    let mut pairs: Vec<(i64, anon::T<'gc>)> = keys.into_iter().zip(arr.iter().copied()).collect();
    pairs.sort_by_key(|&(k, _)| k);
    *arr = pairs.into_iter().map(|(_, v)| v).collect();
}

#[native]
/// Sorts the array in ascending order of `keys`, where `keys[i]` is the key for element `i`. Equal
/// keys keep their original order. `keys` is unchanged.
///
/// ```mimas
/// let names = ["far", "near", "mid"];
/// names.sort_by_float([9.5, 0.5, 3.0]); // names is now ["near", "mid", "far"]
/// ```
///
/// `keys` should be as long as the array. An element without a key is dropped from the array, and
/// extra keys are ignored.
fn sort_by_float(arr: &mut Vec<anon::T<'gc>>, keys: Vec<f64>) {
    let mut pairs: Vec<(f64, anon::T<'gc>)> = keys.into_iter().zip(arr.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    *arr = pairs.into_iter().map(|(_, v)| v).collect();
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

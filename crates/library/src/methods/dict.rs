use macros::native;
use shared::Ty;
use vm::{Ctx, DictMap, RtErr, Val, anon, api::Api};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    // r[impl mimas.dict.type] `dict::new()` is the empty dict
    api.add_assoc(Ty::dict_of(Ty::Anon(anon::KEY_SLOT), Ty::Anon(0)), new);
    api.add_method(len);
    api.add_method(contains_key);
    api.add_method(insert);
    api.add_method(remove);
    api.add_method(pairs);
    // r[impl mimas.dict.keys]
    api.add_method(keys);
    api.add_method(values);
}

#[native]
/// Creates an empty dictionary. It needs a type annotation if nothing else
/// tells the compiler what it will hold.
///
/// ```mimas
/// let ids: {int} = dict::new();
/// ids["a"] = 1; // ids is now {"a": 1}
/// let seen: {(int, int): bool} = dict::new();
/// seen[(2, 3)] = true;
/// ```
fn new<'gc>(ctx: Ctx<'gc>) -> anon::DictOf<'gc, 0> {
    anon::DictOf(ctx.new_dict(DictMap::new()))
}

#[native]
/// Returns the number of entries.
///
/// ```mimas
/// let n = { a: 1, b: 2 }.len(); // 2
/// ```
fn len(d: &DictMap<'gc>) -> usize {
    d.len()
}

#[native]
/// Returns whether the dictionary has an entry for `key`. Unlike indexing, this tells a missing
/// key apart from one whose value is `null`.
///
/// ```mimas
/// let slots: {int?} = dict::new();
/// slots["a"] = null;
/// let a = slots.contains_key("a"); // true, though slots["a"] is null
/// let b = slots.contains_key("b"); // false
/// ```
fn contains_key<'gc>(d: &DictMap<'gc>, key: anon::K<'gc>) -> bool {
    d.contains(key.0)
}

#[native]
/// Sets `key` to `value` and returns the value it replaced, or `null` if the key is new. A new key
/// is added at the end of the iteration order. Replacing a value doesn't move its key.
///
/// Assigning with `dict[key] = value` does the same without returning anything.
///
/// ```mimas
/// let stock = { apples: 3 };
/// let old = stock.insert("apples", 5); // 3
/// let new = stock.insert("pears", 2);  // null
/// ```
fn insert<'gc>(
    ctx: Ctx<'gc>,
    d: &mut DictMap<'gc, anon::T<'gc>>,
    key: anon::K<'gc>,
    val: anon::T<'gc>,
) -> Result<Option<anon::T<'gc>>, RtErr> {
    if let Some(slot) = d.get_key_mut(key.0) {
        return Ok(Some(std::mem::replace(slot, val)));
    }
    Ok(d.insert_key(dict_key(ctx, key.0)?, val))
}

/// A key as the dict stores it: checked, and a tuple or struct copied and frozen
/// (`Ctx::dict_key`).
fn dict_key<'gc>(ctx: Ctx<'gc>, key: Val<'gc>) -> Result<Val<'gc>, RtErr> {
    ctx.dict_key(key)
}

#[native]
/// Removes the entry for `key` and returns its value, or returns `null` if there was no such
/// entry. The remaining entries keep their order.
///
/// ```mimas
/// let stock = { apples: 3, pears: 2 };
/// let a = stock.remove("apples"); // 3, and stock is now { pears: 2 }
/// let b = stock.remove("plums");  // null
/// ```
fn remove(d: &mut DictMap<'gc, anon::T<'gc>>, key: anon::K<'gc>) -> Option<anon::T<'gc>> {
    d.remove_key(key.0)
}

#[native]
/// Returns every entry as a `(key, value)` tuple, in the order the keys were first inserted.
///
/// ```mimas
/// let scores = { ada: 3, bob: 5 };
/// let entries = scores.pairs(); // [("ada", 3), ("bob", 5)]
/// ```
///
/// A `for` loop over the dictionary visits the same pairs without building an array:
///
/// ```mimas
/// for (name, score) in scores {
///     print(f"{name} has {score}");
/// }
/// ```
fn pairs(d: &DictMap<'gc, anon::T<'gc>>) -> Vec<(anon::K<'gc>, anon::T<'gc>)> {
    d.iter().map(|(&k, &v)| (anon::Anon(k), v)).collect()
}

#[native]
/// The keys, in the order they were added.
///
/// ```mimas
/// let ages = { ann: 31, bo: 7 };
/// let names = ages.keys(); // ["ann", "bo"]
/// ```
fn keys(d: &DictMap<'gc, anon::T<'gc>>) -> Vec<anon::K<'gc>> {
    d.iter().map(|(&k, _)| anon::Anon(k)).collect()
}

#[native]
/// The values, in the order their keys were added.
fn values(d: &DictMap<'gc, anon::T<'gc>>) -> Vec<anon::T<'gc>> {
    d.iter().map(|(_, &v)| v).collect()
}

use std::{
    cell::{Ref, RefCell},
    collections::HashMap,
    rc::Rc,
    sync::Arc,
};

use compile::{BinFault, BinOp, Scalar, UnaryOp};
#[cfg(any(feature = "dataframe", feature = "darkly"))]
use gc_arena::Static;
use gc_arena::{Collect, Gc, RefLock};
use shared::{BodyId, FnHeader};
use smallvec::SmallVec;

use crate::{RtErr, RtResult, heap::Ctx};

pub type SharedStr = Arc<str>;

#[derive(Copy, Clone, Default, Collect, Debug)]
#[collect(no_drop)]
pub enum Val<'gc> {
    #[default]
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Fn(#[collect(require_static)] BodyId),
    Str(Str<'gc>),
    Array(Array<'gc>),
    /// A sequence backed by a homogeneous primitive store (`Vec<i64>` /
    /// `Vec<f64>`) instead of `Vec<Val>` — the structure-of-arrays twin of
    /// `Array`, born from `[]`/typed literals and demoting to a real `Array`
    /// in place on the first out-of-kind write. Distinct tag on purpose:
    /// every consumer that pattern-matches `Val::Array` (bcgen/JIT inline
    /// paths included) misses this and falls through to the helpers that box
    /// elements back into `Val`s. Semantically identical to `Array`.
    IntArray(IntArray<'gc>),
    /// Same storage as [`Val::IntArray`]; the tag records that the sequence
    /// was born from float content (see [`ArrayStore`]).
    FloatArray(FloatArray<'gc>),
    Dict(Dict<'gc>),
    Instance(Instance<'gc>),
    Closure(Closure<'gc>),
    Raised(Str<'gc>),
    #[cfg(feature = "dataframe")]
    DataFrame(DataFrame<'gc>),
    /// A `polars::prelude::Expr` under construction -- `col("x")`, `col("x") > 5`, and so on all
    /// build one of these instead of evaluating anything. Immutable once built (every polars
    /// `Expr` combinator consumes and returns a new node), so unlike `DataFrame` this doesn't
    /// need a `RefLock` -- same reasoning as `Str`.
    #[cfg(feature = "dataframe")]
    PlExpr(PlExpr<'gc>),
    /// A `polars::prelude::LazyGroupBy` -- `df.group_by([..])`'s result, before `.agg([..])`
    /// turns it into a `DataFrame`. Same no-`RefLock` reasoning as `PlExpr`: nothing mutates one
    /// in place, `.agg()` consumes it.
    #[cfg(feature = "dataframe")]
    GroupBy(GroupBy<'gc>),
    /// Decoded pixel bytes from a `.darkly` raster/mask layer (`std::darkly::open`). Immutable
    /// once decoded (nothing mutates one in place), so no `RefLock` -- same reasoning as `PlExpr`.
    #[cfg(feature = "darkly")]
    DarklyImage(DarklyImage<'gc>),
    /// An N-D tensor on the Burn NdArray backend (`std::tensor`). The payload is
    /// `TensorPrimitive` -- rank-erased, shape lives at runtime -- so one variant covers every
    /// rank. No `RefLock`: every burn op is functional (consumes and returns a new tensor), same
    /// reasoning as `PlExpr`.
    #[cfg(feature = "tensor")]
    Tensor(Tensor<'gc>),
}

impl<'gc> PartialEq for Val<'gc> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Val::Null, Val::Null) => true,
            (Val::Bool(a), Val::Bool(b)) => a == b,
            (Val::Int(a), Val::Int(b)) => a == b,
            (Val::Float(a), Val::Float(b)) => a == b,
            (Val::Fn(a), Val::Fn(b)) => a == b,
            (Val::Str(a), Val::Str(b)) => a == b,
            (Val::Array(a), Val::Array(b)) => {
                Gc::ptr_eq(a.0, b.0) || *a.0.borrow() == *b.0.borrow()
            }
            // typed arrays compare element-wise against everything array-like --
            // `[1, 2] == [1, 2]` regardless of which side stores raw `i64`s
            (Val::IntArray(a), Val::IntArray(b))
            | (Val::IntArray(a), Val::FloatArray(b))
            | (Val::FloatArray(a), Val::IntArray(b))
            | (Val::FloatArray(a), Val::FloatArray(b)) => {
                Gc::ptr_eq(a.0, b.0) || a.0.borrow().store_eq(&b.0.borrow())
            }
            (Val::Array(a), Val::IntArray(b)) | (Val::Array(a), Val::FloatArray(b)) => {
                b.0.borrow().eq_slice(&a.0.borrow())
            }
            (Val::IntArray(b), Val::Array(a)) | (Val::FloatArray(b), Val::Array(a)) => {
                b.0.borrow().eq_slice(&a.0.borrow())
            }
            (Val::Dict(a), Val::Dict(b)) => Gc::ptr_eq(a.0, b.0) || *a.0.borrow() == *b.0.borrow(),
            (Val::Instance(a), Val::Instance(b)) => {
                Gc::ptr_eq(a.0, b.0) || {
                    let (a, b) = (a.0.borrow(), b.0.borrow());
                    a.struct_id == b.struct_id && a.fields.as_slice() == b.fields.as_slice()
                }
            }
            (Val::Closure(a), Val::Closure(b)) => Gc::ptr_eq(a.0, b.0),
            (Val::Raised(a), Val::Raised(b)) => a == b,
            // an in-progress Expr tree has no structural equality -- same handle only, like
            // Closure. (A DataFrame compares by content, below.)
            #[cfg(feature = "dataframe")]
            (Val::DataFrame(a), Val::DataFrame(b)) => {
                // same columns, in order, holding the same cells (nulls equal nulls)
                Gc::ptr_eq(a.0, b.0) || a.0.borrow().0.equals_missing(&b.0.borrow().0)
            }
            #[cfg(feature = "dataframe")]
            (Val::PlExpr(a), Val::PlExpr(b)) => Gc::ptr_eq(a.0, b.0),
            #[cfg(feature = "dataframe")]
            (Val::GroupBy(a), Val::GroupBy(b)) => Gc::ptr_eq(a.0, b.0),
            #[cfg(feature = "darkly")]
            (Val::DarklyImage(a), Val::DarklyImage(b)) => Gc::ptr_eq(a.0, b.0),
            #[cfg(feature = "tensor")]
            // structural like Array: `tensor([[1]]) == tensor([[1]])`. bitwise-compare the
            // flat f32 payload (a DataFrame compares by content the same way).
            (Val::Tensor(a), Val::Tensor(b)) => {
                Gc::ptr_eq(a.0, b.0) || crate::tensor::all_equal(&a.0.0, &b.0.0)
            }
            _ => false,
        }
    }
}

impl<'gc> Val<'gc> {
    #[inline]
    pub fn as_int(self) -> Option<i64> {
        if let Val::Int(n) = self {
            Some(n)
        } else {
            None
        }
    }

    #[inline]
    pub fn as_float(self) -> Option<f64> {
        if let Val::Float(n) = self {
            Some(n)
        } else {
            None
        }
    }

    #[inline]
    pub fn as_bool(self) -> Option<bool> {
        if let Val::Bool(b) = self {
            Some(b)
        } else {
            None
        }
    }

    #[inline]
    pub fn as_str(self) -> Option<Str<'gc>> {
        if let Val::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }
    /// The untyped `Array` handle only — deliberately does *not* match the
    /// typed variants, so call sites that borrow the `Vec<Val>` directly (the
    /// bcgen/JIT inline paths) can't misread a primitive store. Anything that
    /// wants "any sequence" uses [`seq_len`](Self::seq_len)/
    /// [`seq_get`](Self::seq_get) or [`as_untyped_array`](Self::as_untyped_array).
    #[inline]
    pub fn as_array(self) -> Option<Array<'gc>> {
        if let Val::Array(a) = self {
            Some(a)
        } else {
            None
        }
    }

    /// `true` for every sequence shape — `Array`, `IntArray`, `FloatArray`.
    #[inline]
    pub fn is_seq(&self) -> bool {
        matches!(self, Val::Array(_) | Val::IntArray(_) | Val::FloatArray(_))
    }

    /// Length of any sequence shape, `None` for non-sequences.
    #[inline]
    pub fn seq_len(&self) -> Option<usize> {
        match *self {
            Val::Array(a) => Some(a.0.borrow().len()),
            Val::IntArray(a) | Val::FloatArray(a) => Some(a.0.borrow().len()),
            _ => None,
        }
    }

    /// Element `i` of any sequence shape, boxed back into a `Val`
    /// (`IntArray[i]` yields `Val::Int`, `FloatArray[i]` `Val::Float`).
    /// `None` for non-sequences and out-of-bounds.
    #[inline]
    pub fn seq_get(&self, i: usize) -> Option<Val<'gc>> {
        match *self {
            Val::Array(a) => a.0.borrow().get(i).copied(),
            Val::IntArray(a) | Val::FloatArray(a) => a.0.borrow().get(i),
            _ => None,
        }
    }

    /// An `Array` handle on this sequence's contents, whatever the backing.
    /// For a typed array this *demotes in place*: the store becomes
    /// [`ArrayStore::Vals`] wrapping a fresh `Array`, and that inner handle is
    /// returned — so a native handed it keeps mutating the very elements the
    /// script's `IntArray`/`FloatArray` value still points at. `None` for
    /// non-sequences.
    pub fn as_untyped_array(self, ctx: Ctx<'gc>) -> Option<Array<'gc>> {
        match self {
            Val::Array(a) => Some(a),
            Val::IntArray(a) | Val::FloatArray(a) => Some(a.0.borrow_mut(&ctx).demote(ctx)),
            _ => None,
        }
    }

    #[inline]
    pub fn as_dict(self) -> Option<Dict<'gc>> {
        if let Val::Dict(d) = self {
            Some(d)
        } else {
            None
        }
    }

    #[inline]
    pub fn as_instance(self) -> Option<Instance<'gc>> {
        if let Val::Instance(i) = self {
            Some(i)
        } else {
            None
        }
    }

    #[inline]
    pub fn as_closure(self) -> Option<Closure<'gc>> {
        if let Val::Closure(c) = self {
            Some(c)
        } else {
            None
        }
    }

    #[inline]
    pub fn as_fn(self) -> Option<BodyId> {
        if let Val::Fn(b) = self { Some(b) } else { None }
    }

    /// The signature of the header and return from what the compiler found. Read in place out of
    /// the loaded program's table rather than copied out of it -- see [`Ctx::signature`].
    pub fn signature(self, ctx: Ctx<'gc>) -> Option<Ref<'gc, FnHeader>> {
        let body = self
            .as_fn()
            .or_else(|| self.as_closure().map(|c| c.0.function))?;
        ctx.signature(body)
    }

    #[inline]
    pub fn as_raised(self) -> Option<Str<'gc>> {
        if let Val::Raised(s) = self {
            Some(s)
        } else {
            None
        }
    }

    #[cfg(feature = "dataframe")]
    #[inline]
    pub fn as_dataframe(self) -> Option<DataFrame<'gc>> {
        if let Val::DataFrame(d) = self {
            Some(d)
        } else {
            None
        }
    }

    #[cfg(feature = "dataframe")]
    #[inline]
    pub fn as_plexpr(self) -> Option<PlExpr<'gc>> {
        if let Val::PlExpr(e) = self {
            Some(e)
        } else {
            None
        }
    }

    #[cfg(feature = "dataframe")]
    #[inline]
    pub fn as_group_by(self) -> Option<GroupBy<'gc>> {
        if let Val::GroupBy(g) = self {
            Some(g)
        } else {
            None
        }
    }

    #[cfg(feature = "tensor")]
    #[inline]
    pub fn as_tensor(self) -> Option<Tensor<'gc>> {
        if let Val::Tensor(t) = self {
            Some(t)
        } else {
            None
        }
    }

    #[cfg(feature = "darkly")]
    #[inline]
    pub fn as_darkly_image(self) -> Option<DarklyImage<'gc>> {
        if let Val::DarklyImage(i) = self {
            Some(i)
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub struct Str<'gc>(pub Gc<'gc, SharedStr>);

#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub struct Array<'gc>(pub Gc<'gc, RefLock<Vec<Val<'gc>>>>);

#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub struct Dict<'gc>(pub Gc<'gc, RefLock<DictMap<'gc>>>);

/// The payload behind both `Val::IntArray` and `Val::FloatArray` — one GC
/// cell holding an [`ArrayStore`]. The two `Val` tags share this payload
/// type on purpose: the tag is a birth hint while the store records the
/// truth (see [`ArrayStore`]), and one concrete type lets `IntArray(a) |
/// FloatArray(a)` patterns bind the same handle.
#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub struct Seq<'gc>(pub Gc<'gc, RefLock<ArrayStore<'gc>>>);

/// `Val::IntArray`'s payload — alias of [`Seq`] so the typed tags share one
/// handle type.
pub type IntArray<'gc> = Seq<'gc>;

/// `Val::FloatArray`'s payload — alias of [`Seq`].
pub type FloatArray<'gc> = Seq<'gc>;

/// What's inside an [`IntArray`]/[`FloatArray`]'s `RefLock` — the
/// structure-of-arrays backing for `Vec<Val>`-free sequences.
///
/// ```text
///        push(1)                push(9.9)               push("x")
///  Empty ──────► Ints ──┐         Empty ──────► Floats ──┐
///      │                │                  │             │ (any non-f64)
///      └────────────────┴──────────────────┴─────────────▼─────► Vals
///                    (first out-of-kind write demotes)
/// ```
///
/// `Vals` holds a real `Array` handle rather than a bare `Vec<Val>` so a
/// native that received `Array` via [`Val::as_untyped_array`] keeps writing
/// the same contents the script's typed value observes. A store never moves
/// backward: once `Vals`, always `Vals`. `Empty` is the pending state `[]`
/// starts in — the first push picks the store kind. Either tag can hold any
/// store shape (an `IntArray` that received `push(1.5)` as its first write
/// holds `Floats`); the tag is only a birth hint, the store is the truth.
#[derive(Collect, Debug)]
#[collect(no_drop)]
pub enum ArrayStore<'gc> {
    Empty,
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Vals(Array<'gc>),
}

impl<'gc> ArrayStore<'gc> {
    pub fn len(&self) -> usize {
        match self {
            ArrayStore::Empty => 0,
            ArrayStore::Ints(v) => v.len(),
            ArrayStore::Floats(v) => v.len(),
            ArrayStore::Vals(a) => a.0.borrow().len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Element `i` boxed to a `Val`, `None` out of bounds.
    pub fn get(&self, i: usize) -> Option<Val<'gc>> {
        match self {
            ArrayStore::Empty => None,
            ArrayStore::Ints(v) => v.get(i).map(|&x| Val::Int(x)),
            ArrayStore::Floats(v) => v.get(i).map(|&x| Val::Float(x)),
            ArrayStore::Vals(a) => a.0.borrow().get(i).copied(),
        }
    }

    /// Element `i` boxed to a `Val`, panicking out of bounds exactly like
    /// indexing a `Vec<Val>` does (this replaces `a.0.borrow()[i]` sites).
    pub fn at(&self, i: usize) -> Val<'gc> {
        self.get(i).expect("index out of bounds on typed array")
    }

    /// Write `v` at `i`, demoting to `Vals` first if `v` doesn't fit the
    /// current primitive store. `i` is bounds-checked against the store the
    /// same way `v[i] = x` on a `Vec` is — a mismatch panics, never writes.
    pub fn set(&mut self, ctx: Ctx<'gc>, i: usize, v: Val<'gc>) {
        match self {
            ArrayStore::Ints(vs) => match v {
                Val::Int(x) => vs[i] = x,
                _ => self.demote(ctx).0.borrow_mut(&ctx)[i] = v,
            },
            ArrayStore::Floats(vs) => match v {
                Val::Float(x) => vs[i] = x,
                _ => self.demote(ctx).0.borrow_mut(&ctx)[i] = v,
            },
            ArrayStore::Vals(a) => a.0.borrow_mut(&ctx)[i] = v,
            // `Empty[i]` has no slot — panic like `Vec::new()[i]`
            ArrayStore::Empty => {
                let empty: &[i64] = &[];
                empty[i];
            }
        }
    }

    /// Append `v`, picking the store kind on `Empty` and demoting on an
    /// incompatible write (the "any non-`i64` write to an `Ints`, any
    /// non-`f64` write to a `Floats`" rule — see the enum docs).
    pub fn push(&mut self, ctx: Ctx<'gc>, v: Val<'gc>) {
        match self {
            ArrayStore::Empty => {
                *self = match v {
                    Val::Int(x) => ArrayStore::Ints(vec![x]),
                    Val::Float(x) => ArrayStore::Floats(vec![x]),
                    _ => ArrayStore::Vals(ctx.new_array(vec![v])),
                };
            }
            ArrayStore::Ints(vs) => match v {
                Val::Int(x) => vs.push(x),
                _ => self.demote(ctx).0.borrow_mut(&ctx).push(v),
            },
            ArrayStore::Floats(vs) => match v {
                Val::Float(x) => vs.push(x),
                _ => self.demote(ctx).0.borrow_mut(&ctx).push(v),
            },
            ArrayStore::Vals(a) => a.0.borrow_mut(&ctx).push(v),
        }
    }

    /// Ensure `Vals` state: box the primitive contents into a fresh `Array`,
    /// swap it in, and return the shared handle. `Vals` is a no-op returning
    /// the existing handle, so this never breaks aliasing.
    pub fn demote(&mut self, ctx: Ctx<'gc>) -> Array<'gc> {
        let items: Vec<Val<'gc>> = match self {
            ArrayStore::Empty => Vec::new(),
            ArrayStore::Ints(v) => v.iter().map(|&x| Val::Int(x)).collect(),
            ArrayStore::Floats(v) => v.iter().map(|&x| Val::Float(x)).collect(),
            ArrayStore::Vals(a) => return *a,
        };
        let a = ctx.new_array(items);
        *self = ArrayStore::Vals(a);
        a
    }

    /// `contains(&needle)` semantics across all store shapes.
    pub fn contains(&self, needle: Val<'gc>) -> bool {
        match self {
            ArrayStore::Empty => false,
            ArrayStore::Ints(v) => matches!(needle, Val::Int(i) if v.contains(&i)),
            ArrayStore::Floats(v) => matches!(needle, Val::Float(f) if v.contains(&f)),
            ArrayStore::Vals(a) => a.0.borrow().contains(&needle),
        }
    }

    /// Every element boxed to `Val` — for the snapshot/capture paths that
    /// need owned materialized contents.
    pub fn to_vals(&self) -> Vec<Val<'gc>> {
        match self {
            ArrayStore::Empty => Vec::new(),
            ArrayStore::Ints(v) => v.iter().map(|&x| Val::Int(x)).collect(),
            ArrayStore::Floats(v) => v.iter().map(|&x| Val::Float(x)).collect(),
            ArrayStore::Vals(a) => a.0.borrow().clone(),
        }
    }

    /// Element-wise equality against another store — `Vec` equality where the
    /// stores match kinds, boxed `Val` comparison otherwise.
    pub fn store_eq(&self, other: &ArrayStore<'gc>) -> bool {
        match (self, other) {
            (ArrayStore::Ints(a), ArrayStore::Ints(b)) => a == b,
            (ArrayStore::Floats(a), ArrayStore::Floats(b)) => a == b,
            _ => self.len() == other.len() && (0..self.len()).all(|i| self.at(i) == other.at(i)),
        }
    }

    /// Element-wise equality against a `Vec<Val>` (the `Val::Array` side of a
    /// cross-shape `==`).
    pub fn eq_slice(&self, other: &[Val<'gc>]) -> bool {
        self.len() == other.len() && (0..self.len()).all(|i| self.at(i) == other[i])
    }
}

// generic over the value slot so natives can take a type-safe `DictMap<'gc, anon::T<'gc>>` view
// (see `anon::as_dict_mut`); gc storage is always the `V = Val` default.
#[derive(Collect, Debug)]
#[collect(no_drop)]
pub struct DictMap<'gc, V = Val<'gc>> {
    // entries holds (key, value) in insertion order for stable positional iteration; index maps
    // each key to its slot in entries. invariant: index[k] == position of k in entries.
    entries: Vec<(Str<'gc>, V)>,
    index: HashMap<Str<'gc>, usize>,
}

impl<'gc, V> Default for DictMap<'gc, V> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            index: HashMap::new(),
        }
    }
}

impl<'gc, V> DictMap<'gc, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, key: &Str<'gc>) -> Option<&V> {
        self.index.get(key).map(|&i| &self.entries[i].1)
    }

    pub fn contains_key(&self, key: &Str<'gc>) -> bool {
        self.index.contains_key(key)
    }

    pub fn insert(&mut self, key: Str<'gc>, val: V) -> Option<V> {
        if let Some(&i) = self.index.get(&key) {
            Some(std::mem::replace(&mut self.entries[i].1, val))
        } else {
            self.index.insert(key, self.entries.len());
            self.entries.push((key, val));
            None
        }
    }

    pub fn remove(&mut self, key: &Str<'gc>) -> Option<V> {
        let i = self.index.remove(key)?;
        let (_, val) = self.entries.remove(i);
        for (k, _) in &self.entries[i..] {
            *self.index.get_mut(k).unwrap() -= 1;
        }
        Some(val)
    }

    pub fn entry_at(&self, i: usize) -> (Str<'gc>, V)
    where
        V: Copy,
    {
        self.entries[i]
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Str<'gc>, &V)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }
}

impl<'gc, V: PartialEq> PartialEq for DictMap<'gc, V> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|(k, v)| other.get(k) == Some(v))
    }
}

impl<'gc, V> FromIterator<(Str<'gc>, V)> for DictMap<'gc, V> {
    fn from_iter<I: IntoIterator<Item = (Str<'gc>, V)>>(iter: I) -> Self {
        let mut map = DictMap::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub struct Instance<'gc>(pub Gc<'gc, RefLock<InstanceData<'gc>>>);

#[derive(Collect, Debug)]
#[collect(no_drop)]
pub struct InstanceData<'gc> {
    #[collect(require_static)]
    pub struct_id: u32,
    pub fields: Fields<'gc>,
}

/// The number of fields an ADT can have where we will inline its fields instead of spilling into a
/// newly allocated Vec.
///
/// This could/should eventually be configurable by the user. Depending on your project, you could
/// probably target what the exact cutoff is for your hot-path ADTs.
pub const INLINE_FIELDS: usize = 4;

#[derive(Clone, Collect, Debug)]
#[collect(no_drop)]
pub enum Fields<'gc> {
    Inline {
        #[collect(require_static)]
        len: u8,
        data: [Val<'gc>; INLINE_FIELDS],
    },
    Spilled(Vec<Val<'gc>>),
}

impl<'gc> Fields<'gc> {
    pub fn new(values: Vec<Val<'gc>>) -> Self {
        if values.len() <= INLINE_FIELDS {
            let len = values.len() as u8;
            let mut data = [Val::Null; INLINE_FIELDS];
            for (slot, v) in values.into_iter().enumerate() {
                data[slot] = v;
            }
            Fields::Inline { len, data }
        } else {
            Fields::Spilled(values)
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Fields::Inline { len, .. } => *len as usize,
            Fields::Spilled(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[Val<'gc>] {
        match self {
            Fields::Inline { len, data } => &data[..*len as usize],
            Fields::Spilled(v) => v,
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [Val<'gc>] {
        match self {
            Fields::Inline { len, data } => &mut data[..*len as usize],
            Fields::Spilled(v) => v,
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Val<'gc>> {
        self.as_slice().iter()
    }
}

impl<'gc> std::ops::Index<usize> for Fields<'gc> {
    type Output = Val<'gc>;
    fn index(&self, i: usize) -> &Val<'gc> {
        &self.as_slice()[i]
    }
}

impl<'gc> std::ops::IndexMut<usize> for Fields<'gc> {
    fn index_mut(&mut self, i: usize) -> &mut Val<'gc> {
        &mut self.as_mut_slice()[i]
    }
}

impl<'gc> IntoIterator for Fields<'gc> {
    type Item = Val<'gc>;
    type IntoIter = smallvec::IntoIter<[Val<'gc>; INLINE_FIELDS]>;
    fn into_iter(self) -> Self::IntoIter {
        let values: SmallVec<[Val<'gc>; INLINE_FIELDS]> = match self {
            Fields::Inline { len, data } => SmallVec::from_slice(&data[..len as usize]),
            Fields::Spilled(v) => SmallVec::from_vec(v),
        };
        values.into_iter()
    }
}

#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub struct Closure<'gc>(pub Gc<'gc, ClosureData<'gc>>);

#[derive(Collect, Debug)]
#[collect(no_drop)]
pub struct ClosureData<'gc> {
    #[collect(require_static)]
    pub function: BodyId,
    pub captures: Vec<Val<'gc>>,
}

/// A `polars::frame::DataFrame`, heap-allocated like `Array`/`Dict`/`Instance` -- `RefLock` for
/// the same reason those have it (aliasing, mutation-through-reference). `Static` is
/// gc-arena's zero-cost "trust me, no `Gc` pointers inside" wrapper (`NEEDS_TRACE = false`):
/// a `polars::frame::DataFrame` is an ordinary `'static` Rust value (its own internal sharing is
/// `Arc`-based, nothing gc-arena needs to trace), so this is exactly what it's for.
#[cfg(feature = "dataframe")]
#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub struct DataFrame<'gc>(pub Gc<'gc, RefLock<Static<polars::frame::DataFrame>>>);

/// A `polars::prelude::Expr` under construction. No `RefLock`: `Expr` combinators are purely
/// functional (each one consumes and returns a new node), so nothing ever mutates one in place.
#[cfg(feature = "dataframe")]
#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub struct PlExpr<'gc>(pub Gc<'gc, Static<polars::prelude::Expr>>);

/// A `polars::prelude::LazyGroupBy`, between `df.group_by([..])` and `.agg([..])`. No `RefLock`
/// for the same reason as `PlExpr` -- `.agg()` consumes it, nothing mutates one in place.
/// `LazyGroupBy` doesn't implement `Debug` (unlike `Expr`/`DataFrame`), so this gets a manual,
/// placeholder `Debug` impl below instead of deriving one.
#[cfg(feature = "dataframe")]
#[derive(Copy, Clone, Collect)]
#[collect(no_drop)]
pub struct GroupBy<'gc>(pub Gc<'gc, Static<polars::prelude::LazyGroupBy>>);

#[cfg(feature = "dataframe")]
impl std::fmt::Debug for GroupBy<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GroupBy(..)")
    }
}

/// A Burn `TensorPrimitive` on the NdArray backend -- a plain `'static` Rust value whose
/// internal sharing is `Arc`-based, so `Static` (no GC tracing) is what it's for. No `RefLock`:
/// burn ops are functional -- `a.matmul(b)` returns a new tensor rather than mutating -- so a
/// mimas-level mutation like `t[0] = x` would have to be implemented as read-modify-swap anyway.
#[cfg(feature = "tensor")]
#[derive(Copy, Clone, Collect)]
#[collect(no_drop)]
pub struct Tensor<'gc>(pub Gc<'gc, Static<crate::tensor::Prim>>);

#[cfg(feature = "tensor")]
impl std::fmt::Debug for Tensor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor{:?}", crate::tensor::dims(&self.0.0))
    }
}

#[cfg(feature = "tensor")]
impl<'gc> Tensor<'gc> {
    /// The backend primitive -- `Clone` is an `Arc` bump, so ops take it by value cheaply.
    pub fn inner(self) -> crate::tensor::Prim {
        self.0.0.clone()
    }
}

/// Decoded pixel bytes from a `.darkly` raster/mask layer -- `width * height * channels` bytes,
/// tightly packed, no compression (that's exactly how `.darkly`'s own `.pixels` files store them
/// on disk, so `std_lib::darkly::open` reads them straight in with no decode step). `Static`
/// because `RawImage` is a plain owned Rust value with no `Gc` pointers inside, same reasoning as
/// `DataFrame`/`PlExpr`'s own wrapped external types. No `RefLock`: immutable once decoded.
#[cfg(feature = "darkly")]
#[derive(Copy, Clone, Collect, Debug)]
#[collect(no_drop)]
pub struct DarklyImage<'gc>(pub Gc<'gc, Static<RawImage>>);

#[cfg(feature = "darkly")]
#[derive(Debug)]
pub struct RawImage {
    pub width: u32,
    pub height: u32,
    /// 4 for `rgba8unorm` (a raster layer's own pixels), 1 for `r8unorm` (a mask).
    pub channels: u8,
    pub bytes: Box<[u8]>,
}

impl<'gc> Str<'gc> {
    pub fn as_str(self) -> &'gc str {
        Gc::as_ref(self.0).as_ref()
    }
}

impl<'gc> PartialEq for Str<'gc> {
    fn eq(&self, other: &Self) -> bool {
        Gc::ptr_eq(self.0, other.0)
    }
}
impl<'gc> Eq for Str<'gc> {}

impl<'gc> std::hash::Hash for Str<'gc> {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        Gc::as_ptr(self.0).hash(h);
    }
}

/// A gc-free snapshot of a [`Val<'gc>`] tree. Used for tests.
#[derive(Debug, Clone, PartialEq)]
pub enum Captured {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Captured>),
    Dict(Vec<(String, Captured)>),
    Instance(Vec<Captured>),
    Fn(BodyId),
    Raised(String),
    /// Catch-all for Fn / Closure -- these aren't expected to appear as test outputs, but
    /// if they do, comparing against `Other` will fail loudly rather than panic.
    Other,
    /// An array/dict/instance that's already an ancestor of itself on the current path --
    /// e.g. a doubly-linked list, where a node's `next` and `prev` both lead back into the
    /// same live cycle. Detected by tracking the `Gc` pointers on the path down from the
    /// snapshot's root, not by a recursion-depth cutoff: depth alone doesn't catch this once a
    /// node has *two* live edges back into the cycle (as `next`/`prev` both do here) -- each
    /// recursion re-enters the cycle from a different field, so the *work* is exponential in
    /// the depth limit even though the call stack itself would happily fit. A depth cutoff only
    /// ever bounded the crash; this bounds the work.
    Cycle,
}

/// How many rows [`Inspect::Table`] carries — mirrors the `show_*`
/// `TableView` cap in spirit: enough to typeset a real table, bounded so
/// a million-row frame can't flood a snapshot (the rest report through
/// `dropped`).
pub const INSPECT_TABLE_ROWS: usize = 40;

/// [`Inspect::Table`] for one frame: scalars keep their kind, everything
/// else (dates, nested lists, …) falls back to `AnyValue`'s display
/// text — the same fallback [`crate::table_data`] uses.
#[cfg(feature = "dataframe")]
fn inspect_frame(df: &polars::frame::DataFrame) -> Inspect {
    use polars::prelude::AnyValue;
    let cols: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|n| n.as_str().to_string())
        .collect();
    let series: Vec<polars::prelude::Series> = df
        .get_column_names()
        .iter()
        .filter_map(|n| df.column(n.as_str()).ok())
        .map(|c| c.as_materialized_series().clone())
        .collect();
    let shown = df.height().min(INSPECT_TABLE_ROWS);
    let mut rows = Vec::with_capacity(shown);
    for i in 0..shown {
        rows.push(
            series
                .iter()
                .map(|s| match s.get(i) {
                    Ok(AnyValue::Null) => Inspect::Null,
                    Ok(AnyValue::Boolean(b)) => Inspect::Bool(b),
                    Ok(AnyValue::Int8(v)) => Inspect::Int(v as i64),
                    Ok(AnyValue::Int16(v)) => Inspect::Int(v as i64),
                    Ok(AnyValue::Int32(v)) => Inspect::Int(v as i64),
                    Ok(AnyValue::Int64(v)) => Inspect::Int(v),
                    Ok(AnyValue::UInt8(v)) => Inspect::Int(v as i64),
                    Ok(AnyValue::UInt16(v)) => Inspect::Int(v as i64),
                    Ok(AnyValue::UInt32(v)) => Inspect::Int(v as i64),
                    Ok(AnyValue::UInt64(v)) => i64::try_from(v)
                        .map(Inspect::Int)
                        .unwrap_or(Inspect::Float(v as f64)),
                    Ok(AnyValue::Float32(v)) => Inspect::Float(v as f64),
                    Ok(AnyValue::Float64(v)) => Inspect::Float(v),
                    Ok(AnyValue::String(s)) => Inspect::Str(s.to_string()),
                    Ok(AnyValue::StringOwned(s)) => Inspect::Str(s.as_str().to_string()),
                    Ok(v) => Inspect::Str(format!("{v}")),
                    Err(_) => Inspect::Null,
                })
                .collect(),
        );
    }
    Inspect::Table {
        cols,
        rows,
        dropped: df.height() - shown,
    }
}

impl std::fmt::Display for Captured {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Captured::Null => f.write_str("null"),
            Captured::Bool(b) => write!(f, "{b}"),
            Captured::Int(i) => write!(f, "{i}"),
            Captured::Float(n) => write!(f, "{n}"),
            Captured::Str(s) => write!(f, "{s:?}"),
            Captured::Array(items) => {
                f.write_str("[")?;
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{v}")?;
                }
                f.write_str("]")
            }
            Captured::Dict(entries) => {
                f.write_str("~{")?;
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{k} = {v}")?;
                }
                f.write_str("}")
            }
            Captured::Instance(fields) => {
                f.write_str("{ ")?;
                for (i, v) in fields.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{v}")?;
                }
                f.write_str(" }")
            }
            Captured::Fn(body_id) => write!(f, "fn({})", body_id.index()),
            Captured::Raised(s) => write!(f, "raised({s:?})"),
            Captured::Other => f.write_str("<other>"),
            Captured::Cycle => f.write_str("∞"),
        }
    }
}

impl<'gc> Val<'gc> {
    /// Recursively snapshot `self` into a gc-free [`Captured`] tree. Must be called
    /// inside the arena's `mutate` scope; the resulting `Captured` can safely escape.
    ///
    /// Cuts a cycle to [`Captured::Cycle`] as soon as it re-enters an array/dict/instance
    /// that's already on the path down from `self` -- see that variant's docs for why a plain
    /// recursion-depth cutoff isn't enough once a node has more than one live edge back into
    /// the cycle (an ordinary doubly-linked list already does).
    pub fn capture(self) -> Captured {
        let mut path = std::collections::HashSet::new();
        self.capture_at(&mut path)
    }

    fn capture_at(self, path: &mut std::collections::HashSet<*const ()>) -> Captured {
        // shared by the three `Gc`-backed cases: bail with `Cycle` if `ptr` is already an
        // ancestor on this path, otherwise mark it visited for the duration of `body` and
        // unmark it again on the way back out -- so sibling branches that happen to reach the
        // same shared (non-cyclic) value are still captured in full, only a true cycle is cut.
        fn guarded(
            ptr: *const (),
            path: &mut std::collections::HashSet<*const ()>,
            body: impl FnOnce(&mut std::collections::HashSet<*const ()>) -> Captured,
        ) -> Captured {
            if !path.insert(ptr) {
                return Captured::Cycle;
            }
            let result = body(path);
            path.remove(&ptr);
            result
        }

        match self {
            Val::Null => Captured::Null,
            Val::Bool(b) => Captured::Bool(b),
            Val::Int(i) => Captured::Int(i),
            Val::Float(f) => Captured::Float(f),
            Val::Str(s) => Captured::Str(s.as_str().to_string()),
            Val::Array(a) => guarded(Gc::as_ptr(a.0) as *const (), path, |path| {
                Captured::Array(a.0.borrow().iter().map(|v| v.capture_at(path)).collect())
            }),
            // typed arrays capture as ordinary `Captured::Array` — the
            // representation stays invisible at the snapshot boundary, and a
            // `Vals` store boxes its elements back out.
            Val::IntArray(a) | Val::FloatArray(a) => {
                guarded(Gc::as_ptr(a.0) as *const (), path, |path| {
                    Captured::Array(
                        a.0.borrow()
                            .to_vals()
                            .iter()
                            .map(|v| v.capture_at(path))
                            .collect(),
                    )
                })
            }
            Val::Dict(d) => guarded(Gc::as_ptr(d.0) as *const (), path, |path| {
                Captured::Dict(
                    d.0.borrow()
                        .iter()
                        .map(|(k, v)| (k.as_str().to_string(), v.capture_at(path)))
                        .collect(),
                )
            }),
            Val::Instance(inst) => guarded(Gc::as_ptr(inst.0) as *const (), path, |path| {
                Captured::Instance(
                    inst.0
                        .borrow()
                        .fields
                        .iter()
                        .map(|v| v.capture_at(path))
                        .collect(),
                )
            }),
            Val::Fn(body_id) => Captured::Fn(body_id),
            Val::Raised(s) => Captured::Raised(s.as_str().to_string()),
            Val::Closure(_) => Captured::Other,
            #[cfg(feature = "dataframe")]
            Val::DataFrame(_) | Val::PlExpr(_) | Val::GroupBy(_) => Captured::Other,
            #[cfg(feature = "darkly")]
            Val::DarklyImage(_) => Captured::Other,
            #[cfg(feature = "tensor")]
            Val::Tensor(_) => Captured::Other,
        }
    }
}

/// Like [`Captured`], but keeps the type information `Captured` throws away -- an instance's
/// struct name and each field's declared name, resolved via `Vm`'s `struct_names`/`field_names`
/// tables -- so a debugger's structural inspector can show `Node { val: 391, next: .. }` instead
/// of `{ 391, .. }`. Kept separate from `Captured` rather than adding names to it directly:
/// `Captured` is compared for equality by the test suite, which should stay indifferent to what
/// a struct's fields happen to be named.
#[derive(Debug, Clone)]
pub enum Inspect {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Inspect>),
    Dict(Vec<(String, Inspect)>),
    Instance {
        type_name: String,
        fields: Vec<(String, Inspect)>,
    },
    Fn(BodyId),
    Raised(String),
    /// A `DataFrame` flattened for display — column names plus up to
    /// [`INSPECT_TABLE_ROWS`] rows of typed cells. A host gets no raw
    /// frame handle, so a table it can typeset (or graph — cells keep
    /// their scalar `Inspect` kind) is the honest shape. `dropped`
    /// counts rows the cap left unread.
    Table {
        cols: Vec<String>,
        rows: Vec<Vec<Inspect>>,
        dropped: usize,
    },
    Other,
    /// A true cycle back to an ancestor on the current path (see [`Captured::Cycle`]) -- *or*,
    /// since `seen` in [`Val::inspect`] is shared across everything the caller inspects with it,
    /// a value already shown in full somewhere earlier in that same pass (e.g. two locals that
    /// alias into the same linked structure). Neither is a depth-limit cutoff: both are exact,
    /// pointer-identity dedup, so a shared-but-acyclic DAG can't blow up into copies of itself
    /// once per alias the way a naive per-value walk would.
    Cycle,
}

impl<'gc> Val<'gc> {
    /// Recursively snapshot `self` into a gc-free, name-labeled [`Inspect`] tree. Must be called
    /// inside the arena's `mutate` scope, like [`Val::capture`]. `struct_names`/`field_names` are
    /// `Vm`'s tables, indexed by `struct_id`; an id past either's end (shouldn't happen, but
    /// isn't a safety issue if it does) falls back to `@<id>` / positional numeric names.
    ///
    /// `seen` is caller-owned and never cleared here: pass a fresh one for a single self-
    /// contained snapshot, or thread the same one through several calls (e.g. every local in a
    /// frame) to dedup sharing *across* them too -- see [`Inspect::Cycle`].
    pub fn inspect(
        self,
        struct_names: &[String],
        field_names: &[Vec<String>],
        seen: &mut std::collections::HashSet<*const ()>,
    ) -> Inspect {
        // unlike `capture_at`'s guard, this never un-marks a pointer on the way back out --
        // "seen" here means "already shown in full somewhere in this pass", not just "an
        // ancestor on the current path", so a value reachable two different (non-cyclic) ways
        // still only gets expanded once. See `Inspect::Cycle`.
        fn guarded(
            ptr: *const (),
            seen: &mut std::collections::HashSet<*const ()>,
            body: impl FnOnce(&mut std::collections::HashSet<*const ()>) -> Inspect,
        ) -> Inspect {
            if !seen.insert(ptr) {
                return Inspect::Cycle;
            }
            body(seen)
        }

        match self {
            Val::Null => Inspect::Null,
            Val::Bool(b) => Inspect::Bool(b),
            Val::Int(i) => Inspect::Int(i),
            Val::Float(f) => Inspect::Float(f),
            Val::Str(s) => Inspect::Str(s.as_str().to_string()),
            Val::Array(a) => guarded(Gc::as_ptr(a.0) as *const (), seen, |seen| {
                Inspect::Array(
                    a.0.borrow()
                        .iter()
                        .map(|v| v.inspect(struct_names, field_names, seen))
                        .collect(),
                )
            }),
            Val::IntArray(a) | Val::FloatArray(a) => {
                guarded(Gc::as_ptr(a.0) as *const (), seen, |seen| {
                    Inspect::Array(
                        a.0.borrow()
                            .to_vals()
                            .iter()
                            .map(|v| v.inspect(struct_names, field_names, seen))
                            .collect(),
                    )
                })
            }
            Val::Dict(d) => guarded(Gc::as_ptr(d.0) as *const (), seen, |seen| {
                Inspect::Dict(
                    d.0.borrow()
                        .iter()
                        .map(|(k, v)| {
                            (
                                k.as_str().to_string(),
                                v.inspect(struct_names, field_names, seen),
                            )
                        })
                        .collect(),
                )
            }),
            Val::Instance(inst) => guarded(Gc::as_ptr(inst.0) as *const (), seen, |seen| {
                let inst_ref = inst.0.borrow();
                let struct_id = inst_ref.struct_id as usize;
                let type_name = struct_names
                    .get(struct_id)
                    .cloned()
                    .unwrap_or_else(|| format!("@{struct_id}"));
                let names = field_names.get(struct_id);
                let fields = inst_ref
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let name = names
                            .and_then(|n| n.get(i))
                            .cloned()
                            .unwrap_or_else(|| i.to_string());
                        (name, v.inspect(struct_names, field_names, seen))
                    })
                    .collect();
                Inspect::Instance { type_name, fields }
            }),
            Val::Fn(body_id) => Inspect::Fn(body_id),
            Val::Raised(s) => Inspect::Raised(s.as_str().to_string()),
            Val::Closure(_) => Inspect::Other,
            #[cfg(feature = "dataframe")]
            Val::DataFrame(d) => guarded(Gc::as_ptr(d.0) as *const (), seen, |_| {
                inspect_frame(&d.0.borrow().0)
            }),
            #[cfg(feature = "dataframe")]
            Val::PlExpr(_) | Val::GroupBy(_) => Inspect::Other,
            #[cfg(feature = "darkly")]
            Val::DarklyImage(_) => Inspect::Other,
            #[cfg(feature = "tensor")]
            Val::Tensor(_) => Inspect::Other,
        }
    }
}

/// `a <op> b` on an ADT instance — libraries register impls keyed by
/// `struct_id` through [`crate::api::Api::add_bin_op`]; the impl receives
/// the `op` and declines the ones it doesn't handle via
/// `RtErr::invalid_bin`. `Rc` so the dispatch can clone the impl out of
/// the `RefCell` before calling it: the impl may itself recurse into
/// [`bin`]/[`unary`].
pub type InstanceBin = dyn for<'gc> Fn(Ctx<'gc>, Val<'gc>, Val<'gc>, BinOp) -> RtResult<Val<'gc>>;
pub type InstanceUnary = dyn for<'gc> Fn(Ctx<'gc>, Val<'gc>, UnaryOp) -> RtResult<Val<'gc>>;

#[derive(Default)]
pub struct InstanceOps {
    pub bin: RefCell<HashMap<u32, Rc<InstanceBin>>>,
    pub unary: RefCell<HashMap<u32, Rc<InstanceUnary>>>,
}

/// Infix dispatch for `Val::Instance` operands — the impl registered on the
/// LHS's `struct_id` wins; when only the RHS is an instance its impl sees
/// both operands in source order, so one impl can cover `v * s` and
/// `s * v` alike. Returns `None` when no impl is registered.
fn instance_bin<'gc>(
    a: Val<'gc>,
    ctx: Ctx<'gc>,
    b: Val<'gc>,
    op: BinOp,
) -> Option<RtResult<Val<'gc>>> {
    let sid = |v: Val<'gc>| match v {
        Val::Instance(i) => Some(i.0.borrow().struct_id),
        _ => None,
    };
    let ops = &ctx.fixture::<InstanceOps>().bin;
    for v in [a, b] {
        let f = sid(v).and_then(|id| ops.borrow().get(&id).cloned());
        if let Some(f) = f {
            return Some(f(ctx, a, b, op));
        }
    }
    None
}

pub fn bin<'gc>(this: Val<'gc>, ctx: Ctx<'gc>, other: Val<'gc>, op: BinOp) -> RtResult<Val<'gc>> {
    // inner helpers rebuild the operand `Val`s from their primitives so the
    // `InvalidBinOperands` error gets concrete context. for coerced operands
    // (e.g. Float * Int, both lowered to f64) the rebuilt vals reflect the
    // coerced form -- the lossless `this`/`other` are at the outer match.
    fn bool_bin<'gc>(op: BinOp, a: bool, b: bool) -> RtResult<Val<'gc>> {
        Ok(match op {
            BinOp::And | BinOp::BitAnd => Val::Bool(a && b),
            BinOp::Or | BinOp::BitOr => Val::Bool(a || b),
            BinOp::Xor | BinOp::BitXor => Val::Bool(a ^ b),
            BinOp::Identity => Val::Bool(a == b),
            BinOp::NotEqual => Val::Bool(a != b),
            _ => Err(RtErr::invalid_bin(Val::Bool(a), op, Val::Bool(b)))?,
        })
    }

    fn scalar_val<'gc>(s: Scalar) -> Val<'gc> {
        match s {
            Scalar::Int(i) => Val::Int(i),
            Scalar::Bool(b) => Val::Bool(b),
            Scalar::Float(f) => Val::Float(f),
        }
    }

    fn float_bin<'gc>(op: BinOp, a: f64, b: f64) -> RtResult<Val<'gc>> {
        op.eval_float(a, b)
            .map(scalar_val)
            .map_err(|_| RtErr::invalid_bin(Val::Float(a), op, Val::Float(b)))
    }

    fn int_bin<'gc>(op: BinOp, a: i64, b: i64) -> RtResult<Val<'gc>> {
        op.eval_int(a, b)
            .map(scalar_val)
            .map_err(|fault| match fault {
                BinFault::Overflow => RtErr::IntegerOverflow,
                BinFault::DivByZero => RtErr::DivByZero,
                BinFault::ModByZero => RtErr::ModByZero,
                BinFault::InvalidShift => RtErr::InvalidShift,
                BinFault::Type => RtErr::invalid_bin(Val::Int(a), op, Val::Int(b)),
            })
    }

    // `col("age") > 30 & col("city") == "SF"` -- builds a real `polars::prelude::Expr` tree
    // instead of evaluating anything, so any op where *either* side is already a `PlExpr` takes
    // over here first (ahead of the generic `NotEqual`/`Identity` catch-alls below, which would
    // otherwise compare `Val`s by `PartialEq` -- i.e. by-handle, not "build a polars `.neq()`").
    // A scalar on the other side is auto-promoted via `lit(..)`, matching how you'd write the
    // same comparison directly against polars' own Rust API. Note this is single `&`/`|`
    // (EvaluationOp::And/Or), not `&&`/`||` -- mimas's `&&`/`||` hard-require `Bool` on both
    // sides and short-circuit at codegen (see `Logical::solve`), so they never reach `bin()` at
    // all; `&`/`|` don't short-circuit and already double as mimas's boolean and/or, so they're
    // the only combinator PlExpr composition can actually reach.
    #[cfg(feature = "dataframe")]
    {
        fn to_pl_expr(v: Val<'_>) -> Option<polars::prelude::Expr> {
            use polars::prelude::lit;
            Some(match v {
                Val::PlExpr(e) => e.0.0.clone(),
                Val::Int(i) => lit(i),
                Val::Float(f) => lit(f),
                Val::Bool(b) => lit(b),
                Val::Str(s) => lit(s.as_str()),
                _ => return None,
            })
        }
        if matches!(this, Val::PlExpr(_)) || matches!(other, Val::PlExpr(_)) {
            let (Some(a), Some(b)) = (to_pl_expr(this), to_pl_expr(other)) else {
                Err(RtErr::invalid_bin(this, op, other))?
            };
            let expr = match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mult => a * b,
                BinOp::Div => a / b,
                BinOp::GreaterThan => a.gt(b),
                BinOp::GreaterEqual => a.gt_eq(b),
                BinOp::LessThan => a.lt(b),
                BinOp::LessEqual => a.lt_eq(b),
                BinOp::Identity => a.eq(b),
                BinOp::NotEqual => a.neq(b),
                // single `&`/`|` (`EvaluationOp::And`/`Or`) lower to `BinOp::BitAnd`/`BitOr`, not
                // `BinOp::And`/`Or` -- those are `&&`/`||`'s (`LogicalOp`), which short-circuit at
                // codegen and never reach `bin()` at all (see the note on `to_pl_expr` above).
                BinOp::BitAnd => a.and(b),
                BinOp::BitOr => a.or(b),
                _ => Err(RtErr::invalid_bin(this, op, other))?,
            };
            return Ok(Val::PlExpr(ctx.new_plexpr(expr)));
        }
    }

    Ok(match (op, this, other) {
        (BinOp::Add, Val::Str(a), Val::Str(b)) => {
            let combined = format!("{}{}", a.as_str(), b.as_str());
            Val::Str(ctx.intern(&combined))
        }
        (BinOp::LessThan, Val::Str(a), Val::Str(b)) => Val::Bool(a.as_str() < b.as_str()),
        (BinOp::LessEqual, Val::Str(a), Val::Str(b)) => Val::Bool(a.as_str() <= b.as_str()),
        (BinOp::GreaterThan, Val::Str(a), Val::Str(b)) => Val::Bool(a.as_str() > b.as_str()),
        (BinOp::GreaterEqual, Val::Str(a), Val::Str(b)) => Val::Bool(a.as_str() >= b.as_str()),
        (BinOp::Coalesce, Val::Null, other) => other,
        (BinOp::Coalesce, this, _) => this,
        // `tensor + tensor` / `tensor * 0.5` / `2 - tensor` -- broadcasting elementwise via the
        // burn ops layer; dim mismatches surface the backend's message as a Custom runtime error
        // rather than unwinding (the ops are `catch_unwind`ed in `tensor::bin`). Tensor ops sit
        // ABOVE the structural `==` catch-all on purpose: `t == u` is pervasive like NumPy/Uiua
        // (elementwise 0/1 mask), while structural `Val::eq` still handles match/dict identity.
        #[cfg(feature = "tensor")]
        (op, Val::Tensor(a), Val::Tensor(b)) => Val::Tensor(
            ctx.new_tensor(crate::tensor::bin(a.inner(), op, b.inner()).map_err(RtErr::Custom)?),
        ),
        #[cfg(feature = "tensor")]
        (op, Val::Tensor(a), Val::Float(s)) => Val::Tensor(
            ctx.new_tensor(crate::tensor::bin_scalar(a.inner(), op, s).map_err(RtErr::Custom)?),
        ),
        #[cfg(feature = "tensor")]
        (op, Val::Tensor(a), Val::Int(s)) => Val::Tensor(ctx.new_tensor(
            crate::tensor::bin_scalar(a.inner(), op, s as f64).map_err(RtErr::Custom)?,
        )),
        #[cfg(feature = "tensor")]
        (op, Val::Float(s), Val::Tensor(b)) => Val::Tensor(
            ctx.new_tensor(crate::tensor::scalar_bin(s, op, b.inner()).map_err(RtErr::Custom)?),
        ),
        #[cfg(feature = "tensor")]
        (op, Val::Int(s), Val::Tensor(b)) => Val::Tensor(ctx.new_tensor(
            crate::tensor::scalar_bin(s as f64, op, b.inner()).map_err(RtErr::Custom)?,
        )),
        // `==` is structural on every non-tensor type, arrays included (element-wise `==` returned
        // a list where the checker promised a `bool`). Ordering and arithmetic still broadcast.
        (BinOp::Identity, left, right) => Val::Bool(left == right),
        (BinOp::NotEqual, left, right) => Val::Bool(left != right),
        (op, Val::Bool(a), Val::Bool(b)) => bool_bin(op, a, b)?,
        (op, Val::Float(a), Val::Float(b)) => float_bin(op, a, b)?,
        (op, Val::Float(a), Val::Int(b)) => float_bin(op, a, b as f64)?,
        (op, Val::Int(a), Val::Float(b)) => float_bin(op, a as f64, b)?,
        (op, Val::Int(a), Val::Int(b)) => int_bin(op, a, b)?,
        // elementwise over any sequence pair — `IntArray`/`FloatArray` read
        // their elements boxed back into `Val`s, so a typed array broadcasts
        // exactly like the `Vec<Val>` it replaces.
        (op, a, b) if a.is_seq() && b.is_seq() => {
            let n = a.seq_len().unwrap();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let l = a.seq_get(i).unwrap();
                let r = b.seq_get(i).unwrap();
                out.push(bin(l, ctx, r, op)?);
            }
            ctx.array_val(out)
        }
        (op, a, s @ (Val::Int(_) | Val::Float(_))) if a.is_seq() => {
            let n = a.seq_len().unwrap();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let l = a.seq_get(i).unwrap();
                out.push(bin(l, ctx, s, op)?);
            }
            ctx.array_val(out)
        }
        (op, s @ (Val::Int(_) | Val::Float(_)), b) if b.is_seq() => {
            let n = b.seq_len().unwrap();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let r = b.seq_get(i).unwrap();
                out.push(bin(s, ctx, r, op)?);
            }
            ctx.array_val(out)
        }
        _ => match instance_bin(this, ctx, other, op) {
            Some(v) => v?,
            None => Err(RtErr::invalid_bin(this, op, other))?,
        },
    })
}

pub fn unary<'gc>(this: Val<'gc>, ctx: Ctx<'gc>, op: UnaryOp) -> RtResult<Val<'gc>> {
    fn float_unary<'gc>(op: UnaryOp, value: f64) -> RtResult<Val<'gc>> {
        Ok(match op {
            UnaryOp::Negative => Val::Float(-value),
            UnaryOp::Positive => Val::Float(value),
            _ => Err(RtErr::InvalidUnaryOperand)?,
        })
    }

    fn int_unary<'gc>(op: UnaryOp, value: i64) -> RtResult<Val<'gc>> {
        Ok(match op {
            UnaryOp::BitwiseNot => Val::Int(!value),
            UnaryOp::Negative => Val::Int(value.checked_neg().ok_or(RtErr::IntegerOverflow)?),
            UnaryOp::Positive => Val::Int(value.checked_abs().ok_or(RtErr::IntegerOverflow)?),
            _ => Err(RtErr::InvalidUnaryOperand)?,
        })
    }

    Ok(match (op, this) {
        (UnaryOp::Not, Val::Bool(value)) => Val::Bool(!value),
        (op, Val::Float(value)) => float_unary(op, value)?,
        (op, Val::Int(value)) => int_unary(op, value)?,
        (op, a) if a.is_seq() => {
            let n = a.seq_len().unwrap();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let v = a.seq_get(i).unwrap();
                out.push(unary(v, ctx, op)?);
            }
            ctx.array_val(out)
        }
        _ => match this {
            Val::Instance(i) => {
                let id = i.0.borrow().struct_id;
                let f = ctx
                    .fixture::<InstanceOps>()
                    .unary
                    .borrow()
                    .get(&id)
                    .cloned();
                match f {
                    Some(f) => f(ctx, this, op)?,
                    None => Err(RtErr::InvalidUnaryOperand)?,
                }
            }
            _ => Err(RtErr::InvalidUnaryOperand)?,
        },
    })
}

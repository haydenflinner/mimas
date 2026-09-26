mod library;
mod records;
mod registry;

pub use library::*;
pub use records::*;
pub use registry::*;

shared::id!(pub NativeId);

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Intrinsic {
    Len,
    In,
    Push,
    ToFloat,
    Sqrt,
    File,
    /// `xs.map(f)` -- lowers to a generated loop that calls `f` per element and
    /// collects the results. The `#[native]` body is unreachable; the signature
    /// (`[T] -> ((T) -> U) -> [U]`) is what the checker checks calls against.
    Map,
    /// `xs.filter(f)` -- like [`Intrinsic::Map`], but pushes the element only when
    /// `f(elem)` returns `true`. Signature: `[T] -> ((T) -> bool) -> [T]`.
    Filter,
    /// `xs.fold(init, f)` -- threads an accumulator through `f(acc, elem)` per
    /// element. Signature: `[T] -> (U, (U, T) -> U) -> U`.
    Fold,
    /// `xs.find(f)` -- first element where `f(elem)` returns `true`, else `null`.
    /// Signature: `[T] -> ((T) -> bool) -> T?`.
    Find,
    /// `xs.any(f)` -- `true` as soon as one `f(elem)` returns `true`. Signature:
    /// `[T] -> ((T) -> bool) -> bool`.
    Any,
    /// `xs.all(f)` -- `false` as soon as one `f(elem)` returns `false`. Signature:
    /// `[T] -> ((T) -> bool) -> bool`.
    All,
    /// `xs.flat_map(f)` -- like [`Intrinsic::Map`], but `f` returns `[U]` and each
    /// element of it lands in `out`. Signature: `[T] -> ((T) -> [U]) -> [U]`.
    FlatMap,
    /// `xs.mapi(f)` -- like [`Intrinsic::Map`], but `f` also gets the index.
    /// Signature: `[T] -> ((int, T) -> U) -> [U]`.
    MapI,
    /// `xs.foldi(init, f)` -- like [`Intrinsic::Fold`], but `f` also gets the
    /// index. Signature: `[T] -> (U, (int, U, T) -> U) -> U`.
    FoldI,
    /// `recv.m(a.., f)` where `f: (Recv) -> Recv` -- a combinator whose
    /// transform is a Mimas closure. A native can't call back into the VM, so
    /// the lowering emits `f(recv)` itself, then hands
    /// `merge(recv, f(recv), a..)` to the payload native, which owns the
    /// combining rule (merge over windows, stack, mask-gated pick, ...).
    /// Convention: `f` is the last declared param.
    ApplyMerge(NativeId),
}

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
}

//! Everything Burn touches lives here. `Val::Tensor`'s payload is the backend
//! *primitive* (`FloatTensor<B>` = `NdArrayTensor`) rather than
//! `burn::Tensor<B, D>`: the public `Tensor` is const-generic over rank, but the
//! primitive carries its `Shape` at runtime, so one `Val` variant covers scalars
//! through N-D tensors.
//!
//! The backend is fixed at `NdArray<f32>` (pure Rust + matrixmultiply -- compiles
//! to wasm, no C deps) behind the `B` alias; swapping in `Autodiff<..>` or `Flex`
//! is a one-line change invisible to mimas code.
//!
//! Ops are `B::float_*` supertrait methods (`FloatTensorOps`, `ActivationOps`,
//! ...). Every fallible-looking call runs under `catch_unwind`: burn validates
//! shapes by panicking inside the op, and a Rust panic must never unwind out of
//! a native into the interpreter loop.

use std::panic::{AssertUnwindSafe, catch_unwind};

use burn_backend::{
    DType, FloatDType, IntDType, Shape, TensorData, TensorMetadata,
    backend::ops::{ActivationOps, FloatTensorOps, IntTensorOps},
    tensor::{Device, FloatTensor, IntTensor},
};
use burn_ndarray::NdArray;

pub type B = NdArray<f32>;
/// The stored payload: dynamic-rank f32 tensor.
pub type Prim = FloatTensor<B>;
/// `argmax`-style results come back as int primitives.
pub type IntPrim = IntTensor<B>;

fn dev() -> Device<B> {
    Default::default()
}

fn shape_of(dims: &[usize]) -> Shape {
    Shape::from(dims.to_vec())
}

/// Turn a burn-internal panic (`matmul` on mismatched inner dims, `reshape` to a
/// wrong element count, ...) into a string so natives can surface it as a mimas
/// `Raised` instead of crashing the host.
fn guard<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "tensor operation failed".into())
    })
}

// -- construction ------------------------------------------------------------

pub fn from_flat(data: Vec<f32>, dims: &[usize]) -> Result<Prim, String> {
    guard(|| B::float_from_data(TensorData::new(data, shape_of(dims)), &dev()))
}

pub fn zeros(dims: &[usize]) -> Result<Prim, String> {
    guard(|| B::float_zeros(shape_of(dims), &dev(), FloatDType::F32))
}

pub fn ones(dims: &[usize]) -> Result<Prim, String> {
    guard(|| B::float_ones(shape_of(dims), &dev(), FloatDType::F32))
}

pub fn full(dims: &[usize], value: f64) -> Result<Prim, String> {
    guard(|| {
        B::float_full(
            shape_of(dims),
            burn_backend::Scalar::new(value, &DType::F32),
            &dev(),
            FloatDType::F32,
        )
    })
}

pub fn randn(dims: &[usize]) -> Result<Prim, String> {
    guard(|| {
        B::float_random(
            shape_of(dims),
            burn_backend::Distribution::Normal(0.0, 1.0),
            &dev(),
            FloatDType::F32,
        )
    })
}

pub fn eye(n: usize) -> Result<Prim, String> {
    let data = (0..n * n)
        .map(|i| if i / n == i % n { 1.0 } else { 0.0 })
        .collect();
    from_flat(data, &[n, n])
}

// -- metadata -----------------------------------------------------------------

pub fn dims(t: &Prim) -> Vec<usize> {
    t.shape().to_vec()
}

pub fn rank(t: &Prim) -> usize {
    t.rank()
}

pub fn numel(t: &Prim) -> usize {
    dims(t).iter().product()
}

/// Flatten to row-major `f32`s plus the shape. `float_into_data` is a future even
/// for the synchronous CPU backend; `read_sync` drives it.
pub fn to_flat(t: &Prim) -> Result<(Vec<f32>, Vec<usize>), String> {
    let d = t.shape().to_vec();
    let data = burn_backend::try_read_sync(B::float_into_data(t.clone()))
        .ok_or("tensor read was not synchronous")?
        .map_err(|e| e.to_string())?;
    let vals: Vec<f32> = data.iter::<f32>().collect();
    Ok((vals, d))
}

/// Structural equality: same shape and same cells. Bitwise on the flat f32
/// payload, not `all_close`.
pub fn all_equal(a: &Prim, b: &Prim) -> bool {
    if dims(a) != dims(b) {
        return false;
    }
    match (to_flat(a), to_flat(b)) {
        (Ok((x, _)), Ok((y, _))) => x == y,
        _ => false,
    }
}

// -- elementwise + broadcasting -------------------------------------------------

fn scalar(v: f64) -> burn_backend::Scalar {
    burn_backend::Scalar::new(v, &DType::F32)
}

/// `t op t` -- broadcasting elementwise. `op` is a mimas `BinOp`.
pub fn bin(a: Prim, op: crate::BinOp, b: Prim) -> Result<Prim, String> {
    use crate::BinOp::*;
    guard(move || {
        Ok(match op {
            Add => B::float_add(a, b),
            Sub => B::float_sub(a, b),
            Mult => B::float_mul(a, b),
            Div => B::float_div(a, b),
            Mod => B::float_remainder(a, b),
            _ => return Err("unsupported tensor operator".into()),
        })
    })?
}

/// `t op s` -- a scalar broadcast onto every element.
pub fn bin_scalar(t: Prim, op: crate::BinOp, s: f64) -> Result<Prim, String> {
    use crate::BinOp::*;
    guard(move || {
        let s = scalar(s);
        Ok(match op {
            Add => B::float_add_scalar(t, s),
            Sub => B::float_sub_scalar(t, s),
            Mult => B::float_mul_scalar(t, s),
            Div => B::float_div_scalar(t, s),
            Mod => B::float_remainder_scalar(t, s),
            _ => return Err("unsupported tensor-scalar operator".into()),
        })
    })?
}

/// `s op t` -- scalar on the left. No `*_scalar` reversal exists for `-`/`/`/`%`,
/// so compose them (`s - t` = `(-t) + s`).
pub fn scalar_bin(s: f64, op: crate::BinOp, t: Prim) -> Result<Prim, String> {
    use crate::BinOp::*;
    match op {
        Add | Mult => bin_scalar(t, op, s),
        Sub => bin_scalar(B::float_neg(t), Add, s),
        Div => bin_scalar(B::float_recip(t), Mult, s),
        Mod => Err("`scalar % tensor` is not supported".into()),
        _ => Err("unsupported scalar-tensor operator".into()),
    }
}

// -- linear algebra -------------------------------------------------------------

/// `a.matmul(b)` -- last-two-dims contraction with leading-dim broadcasting.
/// Burn panics on a mismatched inner dim, so validate first for a real error.
pub fn matmul(a: Prim, b: Prim) -> Result<Prim, String> {
    let (da, db) = (dims(&a), dims(&b));
    if da.len() < 2 || db.len() < 2 {
        return Err(format!("matmul needs rank >= 2, got {da:?} x {db:?}"));
    }
    let (ka, kb) = (da[da.len() - 1], db[db.len() - 2]);
    if ka != kb {
        return Err(format!(
            "matmul dim mismatch: {da:?} x {db:?} -- inner dims {ka} != {kb}"
        ));
    }
    guard(|| B::float_matmul(a, b))
}

/// `a.dot(b)` -- 1-D inner product via rank-2 matmul views, reshaped back to a
/// scalar tensor.
pub fn dot(a: Prim, b: Prim) -> Result<Prim, String> {
    let (da, db) = (dims(&a), dims(&b));
    if da.len() != 1 || db.len() != 1 || da[0] != db[0] {
        return Err(format!("dot needs equal 1-D lengths, got {da:?} . {db:?}"));
    }
    let a2 = guard(|| B::float_reshape(a, shape_of(&[1, da[0]])))?;
    let b2 = guard(|| B::float_reshape(b, shape_of(&[db[0], 1])))?;
    let m = guard(|| B::float_matmul(a2, b2))?;
    // NdArray has no rank-0 tensors, so the scalar comes back as `[1]`
    // (same convention as `sum`/`mean`).
    reshape(m, &[1])
}

// -- shape ops ------------------------------------------------------------------

pub fn reshape(t: Prim, dims: &[usize]) -> Result<Prim, String> {
    let want: usize = dims.iter().product();
    let have = numel(&t);
    if want != have {
        return Err(format!("cannot reshape {have} elements into {dims:?}"));
    }
    guard(|| B::float_reshape(t, shape_of(dims)))
}

/// `t()` -- swap the last two dims (matrix transpose); for rank < 2, identity.
pub fn t(t: Prim) -> Result<Prim, String> {
    let r = rank(&t);
    if r < 2 {
        return Ok(t);
    }
    swap_dims(t, r - 2, r - 1)
}

pub fn swap_dims(t: Prim, a: usize, b: usize) -> Result<Prim, String> {
    let r = rank(&t);
    if a >= r || b >= r {
        return Err(format!("swap_dims({a}, {b}) out of bounds for rank {r}"));
    }
    guard(|| B::float_swap_dims(t, a, b))
}

pub fn permute(t: Prim, axes: &[usize]) -> Result<Prim, String> {
    guard(|| B::float_permute(t, axes))
}

pub fn unsqueeze(t: Prim, dim: usize) -> Result<Prim, String> {
    let mut d = dims(&t);
    if dim > d.len() {
        return Err(format!("unsqueeze({dim}) out of bounds for rank {}", d.len()));
    }
    d.insert(dim, 1);
    reshape(t, &d)
}

pub fn squeeze(t: Prim, dim: usize) -> Result<Prim, String> {
    let mut d = dims(&t);
    if dim >= d.len() {
        return Err(format!("squeeze({dim}) out of bounds for rank {}", d.len()));
    }
    if d[dim] != 1 {
        return Err(format!("squeeze({dim}): dim has size {}, not 1", d[dim]));
    }
    d.remove(dim);
    reshape(t, &d)
}

pub fn flatten(t: Prim) -> Result<Prim, String> {
    let n = numel(&t);
    guard(|| B::float_reshape(t, shape_of(&[n])))
}

pub fn cat(ts: Vec<Prim>, dim: usize) -> Result<Prim, String> {
    guard(|| B::float_cat(ts, dim))
}

// -- reductions ------------------------------------------------------------------

pub fn sum(t: Prim) -> Result<Prim, String> {
    guard(|| B::float_sum(t))
}

pub fn mean(t: Prim) -> Result<Prim, String> {
    guard(|| B::float_mean(t))
}

pub fn sum_dim(t: Prim, dim: usize) -> Result<Prim, String> {
    let r = rank(&t);
    if dim >= r {
        return Err(format!("sum_dim({dim}) out of bounds for rank {r}"));
    }
    guard(|| B::float_sum_dim(t, dim))
}

pub fn mean_dim(t: Prim, dim: usize) -> Result<Prim, String> {
    let r = rank(&t);
    if dim >= r {
        return Err(format!("mean_dim({dim}) out of bounds for rank {r}"));
    }
    guard(|| B::float_mean_dim(t, dim))
}

pub fn max(t: Prim) -> Result<Prim, String> {
    guard(|| B::float_max(t))
}

pub fn min(t: Prim) -> Result<Prim, String> {
    guard(|| B::float_min(t))
}

/// `argmax(dim)` -> flat `i64`s plus the surviving shape (dim kept at size 1),
/// for the native to re-box as a mimas array.
pub fn argmax(t: &Prim, dim: usize) -> Result<(Vec<i64>, Vec<usize>), String> {
    let r = rank(t);
    if dim >= r {
        return Err(format!("argmax({dim}) out of bounds for rank {r}"));
    }
    let idx = guard(|| B::float_argmax(t.clone(), dim, IntDType::I64))?;
    let d = idx.shape().to_vec();
    let data = burn_backend::try_read_sync(B::int_into_data(idx))
        .ok_or("tensor read was not synchronous")?
        .map_err(|e| e.to_string())?;
    Ok((data.iter::<i64>().collect(), d))
}

// -- elementwise math --------------------------------------------------------------

pub fn relu(t: Prim) -> Result<Prim, String> {
    guard(|| B::relu(t))
}

pub fn sigmoid(t: Prim) -> Result<Prim, String> {
    guard(|| B::sigmoid(t))
}

pub fn gelu(t: Prim) -> Result<Prim, String> {
    guard(|| B::gelu(t))
}

pub fn softmax(t: Prim, dim: usize) -> Result<Prim, String> {
    let r = rank(&t);
    if dim >= r {
        return Err(format!("softmax({dim}) out of bounds for rank {r}"));
    }
    guard(|| B::softmax(t, dim))
}

pub fn log_softmax(t: Prim, dim: usize) -> Result<Prim, String> {
    let r = rank(&t);
    if dim >= r {
        return Err(format!("log_softmax({dim}) out of bounds for rank {r}"));
    }
    guard(|| B::log_softmax(t, dim))
}

macro_rules! unary {
    ($name:ident, $op:ident) => {
        pub fn $name(t: Prim) -> Result<Prim, String> {
            guard(|| B::$op(t))
        }
    };
}

unary!(tanh, float_tanh);
unary!(exp, float_exp);
unary!(log, float_log);
unary!(sqrt, float_sqrt);
unary!(abs, float_abs);
unary!(neg, float_neg);
unary!(sin, float_sin);
unary!(cos, float_cos);

pub fn powf(t: Prim, s: f64) -> Result<Prim, String> {
    guard(|| B::float_powf_scalar(t, scalar(s)))
}

// -- scalar extraction -------------------------------------------------------------

/// `item()` -- pull the single element out of a numel-1 tensor of any rank.
pub fn item(t: &Prim) -> Result<f64, String> {
    if numel(t) != 1 {
        return Err(format!(
            "item() needs exactly one element, have shape {:?}",
            dims(t)
        ));
    }
    let (vals, _) = to_flat(t)?;
    Ok(vals[0] as f64)
}

// -- display ----------------------------------------------------------------------

/// `print(t)`/`test_run_display!` rendering: `tensor[4, 8]` plus the values for
/// small tensors, just the shape once it gets big (a 784x128 weight matrix is not
/// a thing you want typeset into a cell output).
pub fn render(t: &Prim) -> String {
    let d = dims(t);
    const MAX_PREVIEW: usize = 16;
    match to_flat(t) {
        Ok((vals, _)) if vals.len() <= MAX_PREVIEW => {
            let vals: Vec<String> = vals.iter().map(|v| format!("{v}")).collect();
            format!("tensor{d:?}([{}])", vals.join(", "))
        }
        Ok((vals, _)) => {
            let head: Vec<String> = vals[..MAX_PREVIEW]
                .iter()
                .map(|v| format!("{v}"))
                .collect();
            format!("tensor{d:?}([{}, ...])", head.join(", "))
        }
        Err(_) => format!("tensor{d:?}"),
    }
}

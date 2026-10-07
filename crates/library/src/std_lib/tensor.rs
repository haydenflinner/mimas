//! `std::tensor` -- N-D `f32` tensors on Burn's Flex backend.
//!
//! `Tensor` is a real `Val` variant (`vm::val::Tensor` wrapping a dynamic-rank
//! `burn` primitive), not a `#[mimas] struct`, so this module is only the native
//! surface: constructors, shape/linalg/activation methods, and `to_list`/`item`
//! to get values back out. All actual burn contact is in `vm::tensor`; every
//! fallible op surfaces as `Raised` (`t.reshape(..)!` style) rather than a panic.
//!
//! Elementwise `+ - * / %` on tensors (and tensor-scalar broadcast) come from
//! `Val::bin`, not natives, so `w * 0.5` and `a + b` just work in mimas code.

use macros::native;
use vm::{
    Ctx, Val,
    api::Api,
    conversion::{Raisable, TensorTy},
    tensor as bt,
};

pub(crate) fn install<'gc>(api: &mut Api<'_, 'gc>) {
    // before any `add`/`add_method` below -- parameter/return types resolve
    // `Tensor` through the registry eagerly.
    api.add_adt::<TensorTy>();
    {
        let mut m = api.module("std::tensor");
        m.add(tensor);
        m.add(zeros);
        m.add(ones);
        m.add(full);
        m.add(randn);
        m.add(eye);
        m.add(arange);
        m.add(cat);
        m.add(stack);
    }
    api.add_method(shape);
    api.add_method(rank);
    api.add_method(numel);
    api.add_method(reshape);
    api.add_method(t);
    api.add_method(swap_dims);
    api.add_method(permute);
    api.add_method(unsqueeze);
    api.add_method(squeeze);
    api.add_method(flatten);
    api.add_method(matmul);
    api.add_method(dot);
    api.add_method(sum);
    api.add_method(mean);
    api.add_method(sum_dim);
    api.add_method(mean_dim);
    api.add_method(max);
    api.add_method(min);
    api.add_method(argmax);
    api.add_method(relu);
    api.add_method(sigmoid);
    api.add_method(gelu);
    api.add_method(tanh);
    api.add_method(exp);
    api.add_method(log);
    api.add_method(sqrt);
    api.add_method(abs);
    api.add_method(neg);
    api.add_method(sin);
    api.add_method(cos);
    api.add_method(powf);
    api.add_method(softmax);
    api.add_method(log_softmax);
    api.add_method(to_list);
    api.add_method(item);
    // masks/indexing (Uiua-flavored: comparisons themselves are the pervasive
    // `==`/`!=`/`<`/`>`/`<=`/`>=` operators, not methods)
    api.add_method(slice);
    api.add_method(select);
    api.add_method(rows);
    api.add_method(unfold);
    api.add_method(argsort);
    api.add_method(topk);
    api.add_method(mask_fill);
    api.add_method(mask_where);
    api.add_method(nonzero);
    api.add_method(all);
    api.add_method(any);
    // scan / order / tiling
    api.add_method(cumsum);
    api.add_method(cumprod);
    api.add_method(reverse);
    api.add_method(repeat);
    api.add_method(expand);
    api.add_method(sort);
    api.add_method(floor);
    api.add_method(ceil);
    api.add_method(round);
    api.add_method(sign);
    api.add_method(erf);
}

/// Walk a nested mimas list into `(flat, dims)` for `from_flat`. Every row must
/// agree on width at every depth (a rectangular nest); ints promote to f32.
fn nested(v: Val) -> Result<(Vec<f32>, Vec<usize>), String> {
    let mut flat = Vec::new();
    let mut dims: Vec<usize> = Vec::new();
    fn walk<'gc>(
        v: Val<'gc>,
        depth: usize,
        dims: &mut Vec<usize>,
        flat: &mut Vec<f32>,
    ) -> Result<(), String> {
        if v.is_seq() {
            let n = v.seq_len().unwrap();
            match dims.get_mut(depth) {
                Some(d) if *d == n => {}
                Some(d) => {
                    return Err(format!(
                        "ragged array: dim {depth} has length {n}, expected {d}"
                    ));
                }
                None => dims.push(n),
            }
            for i in 0..n {
                walk(v.seq_get(i).unwrap(), depth + 1, dims, flat)?;
            }
            Ok(())
        } else if let Some(f) = v.as_float().or_else(|| v.as_int().map(|i| i as f64)) {
            if dims.len() != depth {
                return Err("ragged array: scalar at a different depth than siblings".into());
            }
            flat.push(f as f32);
            Ok(())
        } else {
            Err("tensor() takes nested lists of numbers".into())
        }
    }
    walk(v, 0, &mut dims, &mut flat)?;
    Ok((flat, dims))
}

/// Rebuild a nested mimas list of `Val::Float`s from `(flat, dims)`.
fn unnest<'gc>(ctx: Ctx<'gc>, flat: &[f32], dims: &[usize]) -> Val<'gc> {
    fn go<'gc>(ctx: Ctx<'gc>, flat: &mut std::slice::Iter<'_, f32>, dims: &[usize]) -> Val<'gc> {
        if dims.is_empty() {
            return Val::Float(*flat.next().unwrap() as f64);
        }
        let items: Vec<Val<'gc>> = (0..dims[0]).map(|_| go(ctx, flat, &dims[1..])).collect();
        ctx.array_val(items)
    }
    let mut it = flat.iter();
    go(ctx, &mut it, dims)
}

fn wrap<'gc>(ctx: Ctx<'gc>, r: Result<bt::Prim, String>) -> Raisable<vm::Tensor<'gc>> {
    r.map(|p| ctx.new_tensor(p)).into()
}

// -- constructors -------------------------------------------------------------

/// `tensor([[1, 2], [3, 4]])` -- a rank-N tensor from nested lists (ints promote
/// to f32). `tensor(1.5)` is a rank-0 scalar tensor.
#[native]
fn tensor<'gc>(ctx: Ctx<'gc>, data: Val<'gc>) -> Raisable<vm::Tensor<'gc>> {
    let r = nested(data).and_then(|(flat, dims)| bt::from_flat(flat, &dims));
    wrap(ctx, r)
}

/// `zeros([4, 8])` / `zeros([2, 3, 4])` -- an all-zeros tensor of the shape.
#[native]
fn zeros<'gc>(ctx: Ctx<'gc>, shape: Vec<i64>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, dims_of(&shape).and_then(|d| bt::zeros(&d)))
}

/// `ones([4, 8])`.
#[native]
fn ones<'gc>(ctx: Ctx<'gc>, shape: Vec<i64>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, dims_of(&shape).and_then(|d| bt::ones(&d)))
}

/// `full([4, 8], 0.5)` -- every element set to `value`.
#[native]
fn full<'gc>(ctx: Ctx<'gc>, shape: Vec<i64>, value: f64) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, dims_of(&shape).and_then(|d| bt::full(&d, value)))
}

/// `randn([4, 8])` -- standard-normal samples.
#[native]
fn randn<'gc>(ctx: Ctx<'gc>, shape: Vec<i64>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, dims_of(&shape).and_then(|d| bt::randn(&d)))
}

/// `eye(4)` -- the rank-2 identity matrix.
#[native]
fn eye<'gc>(ctx: Ctx<'gc>, n: i64) -> Raisable<vm::Tensor<'gc>> {
    let r = if n <= 0 {
        Err(format!("eye({n}) needs a positive size"))
    } else {
        bt::eye(n as usize)
    };
    wrap(ctx, r)
}

/// `arange(6)` -- `[0, 1, 2, 3, 4, 5]` as a rank-1 float tensor.
#[native]
fn arange<'gc>(ctx: Ctx<'gc>, n: i64) -> Raisable<vm::Tensor<'gc>> {
    let r = if n < 0 {
        Err(format!("arange({n}) needs a non-negative length"))
    } else {
        bt::from_flat((0..n).map(|i| i as f32).collect(), &[n as usize])
    };
    wrap(ctx, r)
}

/// `cat([a, b], 0)` -- concatenate along an existing dim.
#[native]
fn cat<'gc>(ctx: Ctx<'gc>, tensors: Vec<vm::Tensor<'gc>>, dim: i64) -> Raisable<vm::Tensor<'gc>> {
    let prims = tensors.iter().map(|t| t.inner()).collect();
    wrap(ctx, bt::cat(prims, dim.max(0) as usize))
}

/// `stack([a, b], 0)` -- like `cat` on a *new* dim: each tensor is unsqueezed at
/// `dim` first, so stacking two `[3]` vectors gives `[2, 3]`.
#[native]
fn stack<'gc>(ctx: Ctx<'gc>, tensors: Vec<vm::Tensor<'gc>>, dim: i64) -> Raisable<vm::Tensor<'gc>> {
    let d = dim.max(0) as usize;
    let r = tensors
        .iter()
        .map(|t| bt::unsqueeze(t.inner(), d))
        .collect::<Result<Vec<_>, _>>()
        .and_then(|ps| bt::cat(ps, d));
    wrap(ctx, r)
}

fn dims_of(shape: &[i64]) -> Result<Vec<usize>, String> {
    shape
        .iter()
        .map(|&d| {
            if d < 0 {
                Err(format!("negative dim {d}"))
            } else {
                Ok(d as usize)
            }
        })
        .collect()
}

// -- metadata -------------------------------------------------------------------

/// `t.shape()` -> `[int]` -- the dims, e.g. `[784, 128]`. `where` clauses compare
/// these: `where x.shape() == [784]` checks at the fn boundary.
#[native]
fn shape(t: vm::Tensor) -> Vec<i64> {
    bt::dims(&t.inner()).iter().map(|&d| d as i64).collect()
}

/// `t.rank()` -> `int` -- the number of dims.
#[native]
fn rank(t: vm::Tensor) -> i64 {
    bt::rank(&t.inner()) as i64
}

/// `t.numel()` -> `int` -- the total element count.
#[native]
fn numel(t: vm::Tensor) -> i64 {
    bt::numel(&t.inner()) as i64
}

// -- shape ops --------------------------------------------------------------------

/// `t.reshape([4, 8])` -- same elements, new shape; sizes must multiply out.
#[native]
fn reshape<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, shape: Vec<i64>) -> Raisable<vm::Tensor<'gc>> {
    wrap(
        ctx,
        dims_of(&shape).and_then(|d| bt::reshape(t.inner(), &d)),
    )
}

/// `t.t()` -- matrix transpose (swaps the last two dims).
#[native]
fn t<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::t(t.inner()))
}

/// `t.swap_dims(0, 1)`.
#[native]
fn swap_dims<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, a: i64, b: i64) -> Raisable<vm::Tensor<'gc>> {
    wrap(
        ctx,
        bt::swap_dims(t.inner(), a.max(0) as usize, b.max(0) as usize),
    )
}

/// `t.permute([2, 0, 1])` -- rearrange dims into the given order.
#[native]
fn permute<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, axes: Vec<i64>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, dims_of(&axes).and_then(|a| bt::permute(t.inner(), &a)))
}

/// `t.unsqueeze(0)` -- insert a size-1 dim (row vector -> `[1, n]`).
#[native]
fn unsqueeze<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, dim: i64) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::unsqueeze(t.inner(), dim.max(0) as usize))
}

/// `t.squeeze(0)` -- drop a size-1 dim.
#[native]
fn squeeze<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, dim: i64) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::squeeze(t.inner(), dim.max(0) as usize))
}

/// `t.flatten()` -- collapse to rank 1.
#[native]
fn flatten<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::flatten(t.inner()))
}

// -- linalg ------------------------------------------------------------------------

/// `a.matmul(b)` -- matrix multiply (last two dims contract; leading dims broadcast).
/// `x.matmul(w)` for `x: [.., in]` and `w: [in, out]` gives `[.., out]`.
#[native]
fn matmul<'gc>(ctx: Ctx<'gc>, a: vm::Tensor<'gc>, b: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::matmul(a.inner(), b.inner()))
}

/// `a.dot(b)` -- 1-D inner product -> scalar tensor.
#[native]
fn dot<'gc>(ctx: Ctx<'gc>, a: vm::Tensor<'gc>, b: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::dot(a.inner(), b.inner()))
}

// -- reductions ----------------------------------------------------------------------

/// `t.sum()` -- total over every element (scalar tensor).
#[native]
fn sum<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::sum(t.inner()))
}

/// `t.mean()`.
#[native]
fn mean<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::mean(t.inner()))
}

/// `t.sum_dim(1)` -- reduce along `dim`; the dim stays at size 1.
#[native]
fn sum_dim<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, dim: i64) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::sum_dim(t.inner(), dim.max(0) as usize))
}

/// `t.mean_dim(1)`.
#[native]
fn mean_dim<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, dim: i64) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::mean_dim(t.inner(), dim.max(0) as usize))
}

/// `t.max()` / `t.min()` -- scalar extrema.
#[native]
fn max<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::max(t.inner()))
}

#[native]
fn min<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::min(t.inner()))
}

/// `t.argmax(1)` -- indices of the per-slice maxima along `dim` -> `[int]`.
#[native]
fn argmax(t: vm::Tensor, dim: i64) -> Raisable<Vec<i64>> {
    bt::argmax(&t.inner(), dim.max(0) as usize)
        .map(|(idx, _)| idx)
        .into()
}

// -- activations / elementwise math -----------------------------------------------------

#[native]
fn relu<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::relu(t.inner()))
}

#[native]
fn sigmoid<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::sigmoid(t.inner()))
}

#[native]
fn gelu<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::gelu(t.inner()))
}

#[native]
fn tanh<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::tanh(t.inner()))
}

#[native]
fn exp<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::exp(t.inner()))
}

#[native]
fn log<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::log(t.inner()))
}

#[native]
fn sqrt<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::sqrt(t.inner()))
}

#[native]
fn abs<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::abs(t.inner()))
}

#[native]
fn neg<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::neg(t.inner()))
}

#[native]
fn sin<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::sin(t.inner()))
}

#[native]
fn cos<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::cos(t.inner()))
}

/// `t.powf(2)` -- elementwise power.
#[native]
fn powf<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, s: f64) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::powf(t.inner(), s))
}

/// `t.softmax()` / `t.softmax(1)` -- softmax along `dim` (last dim by default).
#[native]
fn softmax<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, dim: Option<i64>) -> Raisable<vm::Tensor<'gc>> {
    let d = dim.map(|d| d.max(0) as usize).unwrap_or_else(|| {
        let r = bt::rank(&t.inner());
        r.saturating_sub(1)
    });
    wrap(ctx, bt::softmax(t.inner(), d))
}

/// `t.log_softmax(dim)` -- `log(softmax)`, numerically stabler for losses.
#[native]
fn log_softmax<'gc>(
    ctx: Ctx<'gc>,
    t: vm::Tensor<'gc>,
    dim: Option<i64>,
) -> Raisable<vm::Tensor<'gc>> {
    let d = dim.map(|d| d.max(0) as usize).unwrap_or_else(|| {
        let r = bt::rank(&t.inner());
        r.saturating_sub(1)
    });
    wrap(ctx, bt::log_softmax(t.inner(), d))
}

// -- back to lists ------------------------------------------------------------------------

// -- masks / indexing / scan ----------------------------------------------------------
//
// Comparisons aren't methods: `==`/`!=`/`<`/`>`/`<=`/`>=` are pervasive
// operators on tensors (`t > 0` is a 0.0/1.0 float mask, `t == u` is
// elementwise). These natives are the verbs masks feed.

/// `t.slice(dim, lo, hi)` -- `t[..., lo..hi, ...]`.
#[native]
fn slice<'gc>(
    ctx: Ctx<'gc>,
    t: vm::Tensor<'gc>,
    dim: i64,
    lo: i64,
    hi: i64,
) -> Raisable<vm::Tensor<'gc>> {
    wrap(
        ctx,
        bt::slice(
            t.inner(),
            dim.max(0) as usize,
            lo.max(0) as usize,
            hi.max(0) as usize,
        ),
    )
}

/// `t.select(dim, [i, j, ..])` -- gather positions along `dim` by index.
#[native]
fn select<'gc>(
    ctx: Ctx<'gc>,
    t: vm::Tensor<'gc>,
    dim: i64,
    idxs: Vec<i64>,
) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::select(t.inner(), dim.max(0) as usize, idxs))
}

/// `t.rows([i, j, ..])` -- `select(0, ..)`; `w.rows(tokens)` is an embedding
/// lookup (and `eye(k)!.rows(labels)` is one-hot).
#[native]
fn rows<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, idxs: Vec<i64>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::select(t.inner(), 0, idxs))
}

/// `t.unfold(dim, size, step)` -- sliding windows along `dim` (Uiua `stencil`'s
/// raw form); `[.., n, ..]` becomes `[.., n_windows, size, ..]`. Chain two dims
/// and reshape for im2col-style conv.
#[native]
fn unfold<'gc>(
    ctx: Ctx<'gc>,
    t: vm::Tensor<'gc>,
    dim: i64,
    size: i64,
    step: i64,
) -> Raisable<vm::Tensor<'gc>> {
    wrap(
        ctx,
        bt::unfold(
            t.inner(),
            dim.max(0) as usize,
            size.max(0) as usize,
            step.max(0) as usize,
        ),
    )
}

/// `t.argsort(dim, desc?)` -- flat list of the per-slice sort indices.
#[native]
fn argsort(t: vm::Tensor, dim: i64, desc: Option<bool>) -> Raisable<Vec<i64>> {
    bt::argsort(&t.inner(), dim.max(0) as usize, desc.unwrap_or(false)).into()
}

/// `t.topk(k)` / `t.topk(k, dim)` -- `(values_tensor, flat_index_list)`.
#[native]
fn topk<'gc>(
    ctx: Ctx<'gc>,
    t: vm::Tensor<'gc>,
    k: i64,
    dim: Option<i64>,
) -> Raisable<(vm::Tensor<'gc>, Vec<i64>)> {
    let d = dim
        .map(|d| d.max(0) as usize)
        .unwrap_or_else(|| bt::rank(&t.inner()) - 1);
    bt::topk(t.inner(), d, k.max(0) as usize)
        .map(|(v, i)| (ctx.new_tensor(v), i))
        .into()
}

/// `t.mask_fill(mask, v)` -- set `v` wherever `mask != 0`.
#[native]
fn mask_fill<'gc>(
    ctx: Ctx<'gc>,
    t: vm::Tensor<'gc>,
    mask: vm::Tensor<'gc>,
    v: f64,
) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::mask_fill(t.inner(), mask.inner(), v))
}

/// `t.mask_where(mask, src)` -- take `src`'s value wherever `mask != 0`.
#[native]
fn mask_where<'gc>(
    ctx: Ctx<'gc>,
    t: vm::Tensor<'gc>,
    mask: vm::Tensor<'gc>,
    src: vm::Tensor<'gc>,
) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::mask_where(t.inner(), mask.inner(), src.inner()))
}

/// `t.nonzero()` -- coordinates of nonzero cells, `[[i, j, ..]]` row-major.
#[native]
fn nonzero(t: vm::Tensor) -> Raisable<Vec<Vec<i64>>> {
    bt::nonzero(&t.inner()).into()
}

/// `t.all()` / `t.any()` -- collapse a mask to a bool (nonzero = truthy).
#[native]
fn all(t: vm::Tensor) -> Raisable<bool> {
    bt::all(&t.inner()).into()
}

#[native]
fn any(t: vm::Tensor) -> Raisable<bool> {
    bt::any(&t.inner()).into()
}

/// `t.cumsum(dim)` / `t.cumprod(dim)` -- cumulative scan along `dim`.
#[native]
fn cumsum<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, dim: i64) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::cumsum(t.inner(), dim.max(0) as usize))
}

#[native]
fn cumprod<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, dim: i64) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::cumprod(t.inner(), dim.max(0) as usize))
}

/// `t.reverse(dim)` -- flip an axis.
#[native]
fn reverse<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, dim: i64) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::reverse(t.inner(), dim.max(0) as usize))
}

/// `t.repeat(dim, n)` -- tile `dim` `n` times.
#[native]
fn repeat<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, dim: i64, n: i64) -> Raisable<vm::Tensor<'gc>> {
    wrap(
        ctx,
        bt::repeat(t.inner(), dim.max(0) as usize, n.max(0) as usize),
    )
}

/// `t.expand(shape)` -- broadcast size-1 dims out to `shape`.
#[native]
fn expand<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>, shape: Vec<i64>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, dims_of(&shape).and_then(|d| bt::expand(t.inner(), &d)))
}

/// `t.sort(dim)` / `t.sort(dim, true)` -- values sorted along `dim`.
#[native]
fn sort<'gc>(
    ctx: Ctx<'gc>,
    t: vm::Tensor<'gc>,
    dim: i64,
    desc: Option<bool>,
) -> Raisable<vm::Tensor<'gc>> {
    wrap(
        ctx,
        bt::sort(t.inner(), dim.max(0) as usize, desc.unwrap_or(false)),
    )
}

#[native]
fn floor<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::floor(t.inner()))
}

#[native]
fn ceil<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::ceil(t.inner()))
}

#[native]
fn round<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::round(t.inner()))
}

#[native]
fn sign<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::sign(t.inner()))
}

#[native]
fn erf<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Raisable<vm::Tensor<'gc>> {
    wrap(ctx, bt::erf(t.inner()))
}

/// `t.to_list()` -- back to nested mimas lists of floats. `Result<_, RtErr>`
/// rather than `Raisable` because `Val`'s type is `Unknown`, which would eat
/// the `!` unwrap -- a read failure (essentially unreachable on Flex) throws
/// as a runtime error instead.
#[native]
fn to_list<'gc>(ctx: Ctx<'gc>, t: vm::Tensor<'gc>) -> Result<Val<'gc>, vm::RtErr> {
    let (flat, d) = bt::to_flat(&t.inner()).map_err(vm::RtErr::Custom)?;
    Ok(unnest(ctx, &flat, &d))
}

/// `t.item()` -- the single element of a numel-1 tensor as a `float`.
#[native]
fn item(t: vm::Tensor) -> Raisable<f64> {
    bt::item(&t.inner()).into()
}

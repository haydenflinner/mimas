//! `std::tensor` -- first-class `Tensor` values backed by burn's NdArray backend.
//! See `crates/library/src/std_lib/tensor.rs` and `crates/vm/src/tensor.rs`.
//!
//! `Tensor` is a real `Val` variant (`Captured::Other` to the inspector, like
//! `DataFrame`), so value assertions here go through `test_run_display!` and the
//! flat `tensor[dims]([elems])` preview, while `shape()`/`to_list()`/`item()`
//! results are ordinary lists and read through `test_run!`.
//!
//! The whole file is a no-op without the `tensor` feature: `std::tensor` isn't
//! registered, so every case would fail to resolve rather than being skipped.
#![cfg(feature = "tensor")]

#[macro_use]
mod test_runner;

const T: &str = "use std::tensor::*;";

// -- construction --------------------------------------------------------------

test_run_display!(
    construct_from_nested_lists,
    T,
    "tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])!" => "tensor[2, 3]([1, 2, 3, 4, 5, 6])",
    "tensor(1.5)!" => "tensor[]([1.5])",
    "tensor([1, 2, 3])!" => "tensor[3]([1, 2, 3])",
);

test_run_display!(
    zeros_ones_full_eye_arange,
    T,
    "zeros([2, 2])!" => "tensor[2, 2]([0, 0, 0, 0])",
    "ones([3])!" => "tensor[3]([1, 1, 1])",
    "full([2, 2], 0.5)!" => "tensor[2, 2]([0.5, 0.5, 0.5, 0.5])",
    "eye(3)!" => "tensor[3, 3]([1, 0, 0, 0, 1, 0, 0, 0, 1])",
    "arange(4)!" => "tensor[4]([0, 1, 2, 3])",
);

// -- metadata --------------------------------------------------------------------

test_run!(
    shape_rank_numel_read_back_as_lists_and_ints,
    T,
    "tensor([[1.0, 2.0], [3.0, 4.0]])!.shape()" => "[2, 2]",
    "tensor([[1.0, 2.0], [3.0, 4.0]])!.rank()" => "2",
    "tensor([[1.0, 2.0], [3.0, 4.0]])!.numel()" => "4",
    "randn([784, 128])!.shape()" => "[784, 128]",
);

// -- elementwise ops through Val::bin ------------------------------------------------

test_run_display!(
    arithmetic_broadcasts_elementwise,
    T,
    "tensor([1.0, 2.0])! + tensor([3.0, 4.0])!" => "tensor[2]([4, 6])",
    "tensor([1.0, 2.0])! * 2" => "tensor[2]([2, 4])",
    "2 * tensor([1.0, 2.0])!" => "tensor[2]([2, 4])",
    "ones([2])! + 0.5" => "tensor[2]([1.5, 1.5])",
    "tensor([4.0, 9.0])!.sqrt()!" => "tensor[2]([2, 3])",
);

// -- linalg ----------------------------------------------------------------------------

test_run_display!(
    matmul_contracts_the_inner_dim,
    T,
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    "tensor([[1.0, 2.0], [3.0, 4.0]])!.matmul(tensor([[5.0, 6.0], [7.0, 8.0]])!)!"
        => "tensor[2, 2]([19, 22, 43, 50])",
    "tensor([[1.0, 2.0]])!.matmul(eye(2)!)!" => "tensor[1, 2]([1, 2])",
);

test_run_display!(
    dot_is_a_scalar,
    T,
    "tensor([1.0, 2.0, 3.0])!.dot(tensor([4.0, 5.0, 6.0])!)!" => "tensor[1]([32])",
);

// -- shape ops ----------------------------------------------------------------------------

test_run_display!(
    reshape_transpose_flatten,
    T,
    "arange(6)!.reshape([2, 3])!" => "tensor[2, 3]([0, 1, 2, 3, 4, 5])",
    "tensor([[1.0, 2.0], [3.0, 4.0]])!.t()!" => "tensor[2, 2]([1, 3, 2, 4])",
    "arange(6)!.reshape([2, 3])!.flatten()!" => "tensor[6]([0, 1, 2, 3, 4, 5])",
    "tensor([1.0, 2.0])!.unsqueeze(0)!.shape()" => "tensor[1, 2]([1, 2])",
);

// -- reductions / activations ------------------------------------------------------------

test_run_display!(
    reductions_and_activations,
    T,
    "tensor([[1.0, 2.0], [3.0, 4.0]])!.sum()!" => "tensor[1]([10])",
    "tensor([1.0, 2.0, 3.0])!.mean()!" => "tensor[1]([2])",
    "tensor([-1.0, 0.0, 2.0])!.relu()!" => "tensor[3]([0, 0, 2])",
    "tensor([[1.0, 2.0], [3.0, 4.0]])!.sum_dim(0)!.shape()" => "tensor[1, 2]([4, 6])",
);

test_run!(
    argmax_reads_back_per_slice_indices,
    T,
    "tensor([[1.0, 5.0], [7.0, 2.0]])!.argmax(1)!" => "[1, 0]",
);

// -- back to lists --------------------------------------------------------------------------

test_run_display!(
    to_list_rebuilds_the_nest,
    T,
    "tensor([[1.5, 2.5], [3.5, 4.5]])!.to_list()!" => "[[1.5, 2.5], [3.5, 4.5]]",
);

test_run!(
    item_extracts_a_scalar_float,
    T,
    "tensor([[1.0, 2.0], [3.0, 4.0]])!.sum()!.item()!" => "10.0",
);

// -- the ML-shaped smoke test: an MLP layer, pure natives -------------------------------------

test_run!(
    mlp_layer_forward_pass,
    T,
    "randn([1, 784])!.matmul(randn([784, 128])!)!.relu()!.shape()" => "[1, 128]",
);

// -- `where` shape contracts --------------------------------------------------------------------

const FORWARD: &str = "use std::tensor::*;
    fn forward(x: Tensor, w: Tensor) -> Tensor
        where x.shape()[1] == w.shape()[0]
    { x.matmul(w)! }
    fn gate(x: Tensor) -> Tensor where x.rank() == 2 { x }";

test_run!(
    where_shape_contract_passes_when_dims_line_up,
    FORWARD,
    "forward(tensor([[1.0, 2.0]])!, eye(2)!)!.shape()" => "[1, 2]",
    "gate(eye(3)!)!.rank()" => "2",
);

test_fail!(
    where_shape_contract_fails_at_the_boundary,
    FORWARD,
    // x is [1, 3] but w is [2, 2] -- the contract reads the shape at entry.
    "forward(tensor([[1.0, 2.0, 3.0]])!, eye(2)!)",
    "gate(tensor([1.0, 2.0])!)",
);

// -- errors surface as Raised, never panics -------------------------------------------------------

test_fail!(
    bad_ops_raise_instead_of_panicking,
    T,
    // ragged input to the constructor
    "tensor([[1.0], [2.0, 3.0]])",
    // inner dims don't line up
    "tensor([[1.0, 2.0]])!.matmul(eye(3)!)",
    // rank-1 matmul
    "tensor([1.0, 2.0])!.matmul(tensor([3.0, 4.0])!)",
    // wrong element count
    "arange(6)!.reshape([4])",
    // squeeze a dim that isn't 1
    "arange(6)!.reshape([2, 3])!.squeeze(1)",
    // negative dim
    "zeros([-1, 2])",
    // permute that isn't a permutation
    "eye(2)!.permute([0, 0])",
);

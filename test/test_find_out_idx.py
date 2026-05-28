"""Tests for hessian.find_out_idx.

Pin down current behavior so a faster rewrite can be validated against it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from folx.api import (
    JAC_DIM,
    FunctionFlags,
    FwdJacobian,
    FwdLaplArgs,
    FwdLaplArray,
)
from folx.hessian import find_out_idx

jax.config.update('jax_enable_x64', True)


def _make_args(masks, x_shapes=None):
    """Build a FwdLaplArgs from a list of x0_idx masks.

    If a mask entry is None, builds a dense (non-weak) jacobian instead.
    """
    arrays = []
    for i, mask in enumerate(masks):
        if mask is None:
            assert x_shapes is not None
            shape = x_shapes[i]
            data = jnp.zeros((3, *shape), dtype=jnp.float64)
            x = jnp.zeros(shape, dtype=jnp.float64)
            lapl = jnp.zeros(shape, dtype=jnp.float64)
            jac = FwdJacobian(data=data, x0_idx=None)
        else:
            x0_idx = np.asarray(mask, dtype=np.int32)
            data = jnp.zeros(x0_idx.shape, dtype=jnp.float64)
            x_shape = x_shapes[i] if x_shapes is not None else x0_idx.shape[1:]
            x = jnp.zeros(x_shape, dtype=jnp.float64)
            lapl = jnp.zeros(x_shape, dtype=jnp.float64)
            jac = FwdJacobian(data=data, x0_idx=x0_idx)
        arrays.append(FwdLaplArray(x=x, jacobian=jac, laplacian=lapl))
    return FwdLaplArgs(arrays=tuple(arrays))


def _assert_idx_equal(actual, expected):
    """Compare two index arrays."""
    assert actual is not None
    expected = np.asarray(expected, dtype=np.int64)
    assert actual.shape == expected.shape, (
        f'shape mismatch: {actual.shape} vs {expected.shape}'
    )
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_no_weak_jac_returns_none_dense():
    args = _make_args([None], x_shapes=[(4,)])
    out, dense = find_out_idx(args, ((),), FunctionFlags.GENERAL, threshold=10)
    assert out is None
    assert dense is True


def test_single_jac_already_unique():
    m1 = np.array([[0, 1, 2], [3, 4, -1]], dtype=np.int32)
    args = _make_args([m1])
    out, dense = find_out_idx(args, ((),), FunctionFlags.GENERAL, threshold=100)
    _assert_idx_equal(out, [[0, 1, 2], [3, 4, -1]])
    assert dense is False


def test_two_jacs_union():
    m1 = np.array([[0, 1, 2], [3, 4, -1]], dtype=np.int32)
    m2 = np.array([[0, 1, 5], [6, 4, -1]], dtype=np.int32)
    args = _make_args([m1, m2])
    out, dense = find_out_idx(args, ((), ()), FunctionFlags.GENERAL, threshold=100)
    # Per column unique non-neg sorted:
    # col 0: {0, 3} ∪ {0, 6} = {0, 3, 6}
    # col 1: {1, 4} ∪ {1, 4} = {1, 4}
    # col 2: {2}    ∪ {5}     = {2, 5}
    _assert_idx_equal(out, [[0, 1, 2], [3, 4, 5], [6, -1, -1]])
    assert dense is False


def test_two_jacs_intersect():
    m1 = np.array([[0, 1, 2], [3, 4, -1]], dtype=np.int32)
    m2 = np.array([[0, 1, 5], [6, 4, -1]], dtype=np.int32)
    args = _make_args([m1, m2])
    out, dense = find_out_idx(args, ((), ()), FunctionFlags.LINEAR_IN_ONE, threshold=100)
    # Per column intersect:
    # col 0: {0, 3} ∩ {0, 6} = {0}
    # col 1: {1, 4} ∩ {1, 4} = {1, 4}
    # col 2: {2}    ∩ {5}     = {}
    _assert_idx_equal(out, [[0, 1, -1], [-1, 4, -1]])
    assert dense is False


def test_extra_broadcast_dim():
    m1 = np.array(
        [[[0, 1, 2], [10, 11, 12]], [[3, 4, -1], [13, 14, -1]]], dtype=np.int32
    )
    args = _make_args([m1])
    out, dense = find_out_idx(args, ((),), FunctionFlags.GENERAL, threshold=100)
    assert out is not None
    assert out.shape == (2, 2, 3)
    _assert_idx_equal(out, m1)
    assert dense is False


def test_broadcast_with_size_one_axis():
    # Two masks broadcasted: (2, 3, 1) and (2, 1, 3) → output (..., 3, 3)
    m1 = np.array([[[0], [1], [2]], [[3], [4], [-1]]], dtype=np.int32)
    m2 = np.array([[[5, 6, 7]], [[-1, 8, 9]]], dtype=np.int32)
    args = _make_args([m1, m2])
    out, dense = find_out_idx(args, ((), ()), FunctionFlags.GENERAL, threshold=100)
    assert out is not None
    # Sanity: shape should be (max_count, 3, 3)
    assert out.shape[1:] == (3, 3)
    # Verify a few positions:
    # position (0, 0): m1 contributes {0, 3}, m2 contributes {5} → {0, 3, 5}
    # position (2, 2): m1 contributes {2}, m2 contributes {7, 9} → {2, 7, 9}
    pos_00 = sorted(int(v) for v in out[:, 0, 0] if v >= 0)
    pos_22 = sorted(int(v) for v in out[:, 2, 2] if v >= 0)
    assert pos_00 == [0, 3, 5]
    assert pos_22 == [2, 7, 9]
    assert dense is False


def test_threshold_triggers_dense():
    m1 = np.array([[0, 1], [2, 3]], dtype=np.int32)
    args = _make_args([m1])
    out, dense = find_out_idx(args, ((),), FunctionFlags.GENERAL, threshold=1)
    assert out is not None
    # K_out=2 > threshold=1 → dense=True
    assert out.shape[JAC_DIM] == 2
    assert dense is True


def test_dense_when_at_max_size():
    # Force out shape to reach max_size: each column has only one unique input,
    # but max_size counts unique indices across the whole mask (4 here).
    m1 = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.int32)
    args = _make_args([m1])
    out, dense = find_out_idx(args, ((),), FunctionFlags.GENERAL, threshold=100)
    assert out is not None
    # Each column has 4 unique → K_out=4 == max_size=4 → dense=True
    assert out.shape[JAC_DIM] == 4
    assert dense is True


def test_all_invalid_mask():
    m1 = -np.ones((2, 3), dtype=np.int32)
    args = _make_args([m1])
    out, dense = find_out_idx(args, ((),), FunctionFlags.GENERAL, threshold=100)
    assert out is not None
    assert out.shape == (0, 3)
    # max_size = 0; idx.shape[JAC_DIM]=0 >= 0 → dense
    assert dense is True


def test_duplicate_indices_same_column():
    # Same index appears twice in a column; unique should collapse it.
    m1 = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32)
    args = _make_args([m1])
    out, dense = find_out_idx(args, ((),), FunctionFlags.GENERAL, threshold=100)
    _assert_idx_equal(out, [[0, 1, 2]])
    assert dense is False


def test_with_vmap_axes():
    # x has shape (B=4, D=3); in_axes = (0,) vmaps over B.
    # mask shape (K=2, B=4, D=3); kept axes (JAC=0, B_shifted=1).
    # Inner unique flattens K*B=8 per D position.
    m = np.arange(2 * 4 * 3, dtype=np.int32).reshape(2, 4, 3)
    args = _make_args([m], x_shapes=[(4, 3)])
    out, dense = find_out_idx(args, ((0,),), FunctionFlags.GENERAL, threshold=100)
    assert out is not None
    # Each D position has 8 unique values from K*B flattening.
    assert out.shape == (8, 3)
    # Column 0: indices 0,3,6,9,12,15,18,21 (sorted)
    np.testing.assert_array_equal(
        np.asarray(out[:, 0]), np.array([0, 3, 6, 9, 12, 15, 18, 21])
    )
    assert dense is False


def test_three_jacs_union():
    # Three sparse jacobians with shape (K=2, S=2).
    m1 = np.array([[0, 1], [2, -1]], dtype=np.int32)
    m2 = np.array([[1, 3], [4, -1]], dtype=np.int32)
    m3 = np.array([[5, 1], [-1, -1]], dtype=np.int32)
    args = _make_args([m1, m2, m3])
    out, dense = find_out_idx(
        args, ((), (), ()), FunctionFlags.GENERAL, threshold=100
    )
    # col 0: {0,2} ∪ {1,4} ∪ {5} = {0,1,2,4,5}
    # col 1: {1}   ∪ {3}   ∪ {1} = {1, 3}
    expected = np.full((5, 2), -1, dtype=np.int64)
    expected[:5, 0] = [0, 1, 2, 4, 5]
    expected[:2, 1] = [1, 3]
    _assert_idx_equal(out, expected)
    # max_size = max(3, 3, 2) = 3; K_out=5 >= 3 → dense=True
    assert dense is True


def test_three_jacs_intersect():
    m1 = np.array([[0, 1], [2, -1]], dtype=np.int32)
    m2 = np.array([[1, 3], [0, -1]], dtype=np.int32)
    m3 = np.array([[5, 1], [0, -1]], dtype=np.int32)
    args = _make_args([m1, m2, m3])
    out, dense = find_out_idx(
        args, ((), (), ()), FunctionFlags.LINEAR_IN_ONE, threshold=100
    )
    # col 0: {0,2} ∩ {0,1,3} ∩ {0,1,5} = {0}
    # col 1: {1}   ∩ {3}     ∩ {1}     = {}
    expected = np.array([[0, -1]], dtype=np.int64)
    _assert_idx_equal(out, expected)
    assert dense is False


def test_two_jacs_different_K():
    # K_1 = 2, K_2 = 3
    m1 = np.array([[0, 1], [2, 3]], dtype=np.int32)
    m2 = np.array([[10, 11], [12, 13], [-1, -1]], dtype=np.int32)
    args = _make_args([m1, m2])
    out, dense = find_out_idx(args, ((), ()), FunctionFlags.GENERAL, threshold=100)
    # col 0: {0,2} ∪ {10,12} = {0,2,10,12}
    # col 1: {1,3} ∪ {11,13} = {1,3,11,13}
    expected = np.array([[0, 1], [2, 3], [10, 11], [12, 13]], dtype=np.int64)
    _assert_idx_equal(out, expected)
    # max_size = max(4, 4) = 4; K_out=4 >= 4 → dense=True
    assert dense is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

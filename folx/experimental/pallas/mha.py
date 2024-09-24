from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax._src.pallas.pallas_call import pallas_call
from jax._src.pallas.primitives import dot as pl_dot
from jax._src.pallas.primitives import load as pl_load
from jax._src.pallas.primitives import program_id
from jax._src.state.indexing import dslice as pl_dslice

from .utils import (
    compute_q_and_kv_block_len,
    create_grid,
    get_mask_block_spec,
    get_value_or_laplacian_block_spec,
)


def mha(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    input_mask: jax.Array,
    kernel: Literal["pallas", "reference"] = "pallas",
    interpret: bool = False,
    q_block_len: int | None = None,
    num_warps: int = 2,
    num_stages: int = 2,
) -> jax.Array:
    r"""Pallas implementation of masked multi-head attention."""
    del input_mask  # Only used in the forward Laplacian
    batch_len, seq_len, num_heads, head_len = q.shape
    q_block_len, kv_block_len = compute_q_and_kv_block_len(seq_len, q_block_len)

    if kernel == "pallas":
        kernel_fn = pallas_call(
            partial(mha_kernel, q_block_len=q_block_len),
            grid=create_grid(batch_len, seq_len, num_heads, q_block_len),
            in_specs=[
                get_value_or_laplacian_block_spec(seq_len, head_len, q_block_len),
                get_value_or_laplacian_block_spec(seq_len, head_len, kv_block_len),
                get_value_or_laplacian_block_spec(seq_len, head_len, kv_block_len),
                get_mask_block_spec(seq_len, q_block_len),
            ],
            out_specs=get_value_or_laplacian_block_spec(seq_len, head_len, q_block_len),
            out_shape=jax.ShapeDtypeStruct(
                shape=(batch_len, seq_len, num_heads, head_len), dtype=q.dtype
            ),
            compiler_params=dict(triton=dict(num_warps=num_warps, num_stages=num_stages)),
            debug=False,
            interpret=interpret,
            name="mha",
        )
    elif kernel == "reference":
        kernel_fn = reference_mha_kernel
    else:
        raise ValueError(f"Unknown multi-head attention kernel: {kernel}")
    o = kernel_fn(q, k, v, mask)
    return o


def reference_mha_kernel(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    interpret: bool = False,
) -> jax.Array:
    r"""Reference jax implementation of the multi-head attention distance kernel."""
    del interpret  # Only used with the pallas kernel
    # [batch_len, seq_len, num_heads, seq_len]
    square_mask = mask[:, None, None, :] * mask[:, :, None, None]
    s = jnp.einsum("Biha,Bjha->Bihj", q, k)
    s = jnp.where(square_mask, s, -1e20)
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum("Bihj,Bjha->Biha", p, v)
    return o


def mha_kernel(
    q_ref,  # Inputs
    k_ref,
    v_ref,
    mask_ref,
    o_ref,  # Outputs
    q_block_len: int | None,
):
    r"""The pallas implementation of the multi-head attention kernel.

    Here pallas grid has already removed the batch and head dimensions.

    Args:
        q_ref: Queries, shape ``(sequence_length, head_dim)``
        k_ref: Keys, shape ``(sequence_length, head_dim)``
        v_ref: Values, shape ``(sequence_length, head_dim)``
        mask_ref: Mask of the q, k, v values, shape ``(sequence_length,)``
        o_ref: Output, shape ``(sequence_length, head_dim)``
        q_block_len: pallas block length
    """
    q_idx = 0 if q_block_len is None else program_id(1)
    q_block_len = q_block_len or q_ref.shape[0]
    kv_mask = mask_ref[:]
    q_slice = pl_dslice(q_idx * q_block_len, q_block_len)
    q_mask = pl_load(mask_ref, (q_slice,))
    square_mask = q_mask[:, None] * kv_mask[None, :]
    # Forward pass
    q = jnp.where(q_mask[:, None], q_ref[:, :], 0.0)
    k = jnp.where(kv_mask[:, None], k_ref[:, :], 0.0)
    v = jnp.where(kv_mask[:, None], v_ref[:, :], 0.0)
    s = jnp.where(square_mask, pl_dot(q, k, trans_b=True), -1e20)
    p = jax.nn.softmax(s, axis=1)
    o = pl_dot(p, v)
    o_ref[:, :] = o
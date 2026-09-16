# SPDX-License-Identifier: Apache-2.0
"""Experimental variants of the VSA-H3 block-sparse attention kernel (route A).

Each variant is a copy of ``_attn_fwd_sparse`` from
``sglang/multimodal_gen/runtime/layers/attention/backends/vsa_h3_kernels.py``
with one or more changes toggled by constexpr flags, so the benchmark can
isolate the effect of every change:

* ``PREFETCH_INDEX`` - load the next kv-tile index and its valid size one
  iteration ahead, so the TMA load of K for iteration i+1 does not wait on a
  dependent scalar load at the loop head (NCU: 13.5% of samples stall
  ``long_scoreboard`` at the loop head).
* ``FOLD_SCALE`` - pre-multiply q by ``sm_scale * log2(e)`` in fp32 before
  the loop and round back to bf16, removing the 64x64 scalar multiply per
  iteration. Changes rounding (q is re-rounded), so the benchmark checks the
  error against the fp32 reference separately.
* ``SKIP_FULL_MASK`` - apply the pad-column mask only when the kv tile is
  ragged (``block_size < BLOCK_N``), a runtime scalar branch inside the loop.

Unverified until run on a GPU: everything here is validated by
``bench_vsa_h3_attn.py --check``.
"""

import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK = 64


@triton.jit
def _attn_fwd_sparse_v1(
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    sm_scale,
    q2k_index,
    q2k_num,
    max_kv_blks,
    variable_block_sizes,
    H,
    N_CTX_Q,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PREFETCH_INDEX: tl.constexpr,
    FOLD_SCALE: tl.constexpr,
    SKIP_FULL_MASK: tl.constexpr,
):
    q_blk = tl.program_id(0)
    off_hz = tl.program_id(1)
    b = off_hz // H
    h = off_hz % H
    q_tiles = N_CTX_Q // BLOCK_M
    meta_base = off_hz * q_tiles + q_blk

    kv_blocks = tl.load(q2k_num + meta_base)
    kv_ptr = q2k_index + meta_base.to(tl.int64) * max_kv_blks

    qk_scale = sm_scale * 1.44269504  # 1/ln2
    q = desc_q.load([b, h, q_blk * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])
    if FOLD_SCALE:
        q = (q.to(tl.float32) * qk_scale).to(tl.bfloat16)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if PREFETCH_INDEX:
        # Row 0 of the list always exists (max_kv_blks >= 1); when kv_blocks
        # is 0 the loop body never runs and the value is unused.
        kv_idx = tl.load(kv_ptr).to(tl.int32)
        block_size = tl.load(variable_block_sizes + kv_idx)

    for i in range(0, kv_blocks):
        if PREFETCH_INDEX:
            has_next = (i + 1) < kv_blocks
            kv_idx_next = tl.load(kv_ptr + i + 1, mask=has_next, other=0).to(tl.int32)
            block_size_next = tl.load(variable_block_sizes + kv_idx_next)
        else:
            kv_idx = tl.load(kv_ptr + i).to(tl.int32)
            block_size = tl.load(variable_block_sizes + kv_idx)

        k = desc_k.load([b, h, kv_idx * BLOCK_N, 0]).reshape([BLOCK_N, HEAD_DIM])
        qk = tl.dot(q, tl.trans(k))
        if SKIP_FULL_MASK:
            if block_size < BLOCK_N:
                mask = tl.arange(0, BLOCK_N) < block_size
                qk = tl.where(mask[None, :], qk, -float("inf"))
        else:
            mask = tl.arange(0, BLOCK_N) < block_size
            qk = tl.where(mask[None, :], qk, -float("inf"))

        if FOLD_SCALE:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = desc_v.load([b, h, kv_idx * BLOCK_N, 0]).reshape([BLOCK_N, HEAD_DIM])
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

        if PREFETCH_INDEX:
            kv_idx = kv_idx_next
            block_size = block_size_next

    acc = acc / l_i[:, None]
    desc_o.store(
        [b, h, q_blk * BLOCK_M, 0],
        acc.to(desc_o.dtype).reshape([1, 1, BLOCK_M, HEAD_DIM]),
    )


def launch_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    out: torch.Tensor,
    *,
    num_warps: int,
    num_stages: int,
    prefetch_index: bool = False,
    fold_scale: bool = False,
    skip_full_mask: bool = False,
    maxnreg: int | None = None,
):
    """Same contract as ``vsa_h3_block_sparse_attn_forward`` (q/k/v/out are
    contiguous [B, H, S_pad, D] bf16; lists are [B, H, n_tiles, max_kv] /
    [B, H, n_tiles] int32). Returns the compiled kernel handle."""
    batch, heads, seq_q, head_dim = q.shape
    block = [1, 1, BLOCK, head_dim]
    desc_q, desc_k, desc_v, desc_o = (
        TensorDescriptor.from_tensor(t, block_shape=block) for t in (q, k, v, out)
    )
    grid = (seq_q // BLOCK, batch * heads, 1)
    extra = {}
    if maxnreg is not None:
        extra["maxnreg"] = maxnreg
    return _attn_fwd_sparse_v1[grid](
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        1.0 / math.sqrt(head_dim),
        q2k_index,
        q2k_num,
        q2k_index.shape[-1],
        variable_block_sizes,
        heads,
        seq_q,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        PREFETCH_INDEX=prefetch_index,
        FOLD_SCALE=fold_scale,
        SKIP_FULL_MASK=skip_full_mask,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra,
    )

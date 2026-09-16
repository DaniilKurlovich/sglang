# SPDX-License-Identifier: Apache-2.0
"""Standalone copy of the VSA-H3 tile geometry / top-k list construction.

Mirrors ``sglang/multimodal_gen/runtime/layers/attention/backends/
video_sparse_attn_h3.py`` (``_h3_tile_geometry``, ``_compute_topk``,
``_topk_tile_lists``, ``_Workspace``, ``_select_kv_lists``) and the helpers it
takes from ``video_sparse_attn.py``, so the benchmark runs in the minimal
docker image (docker/Dockerfile.vsa_h3_bench) that ships only
``vsa_h3_kernels.py`` under a stub ``sglang`` package tree. Keep in sync with
upstream; ``verify_against_sglang`` cross-checks the geometry when the real
package is importable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

VSA_H3_TILE_SHAPE = (4, 4, 4)
VSA_H3_TILE_ELEMS = math.prod(VSA_H3_TILE_SHAPE)  # 64


# --- video_sparse_attn.py helpers (verbatim semantics) ---------------------- #
def get_tile_partition_indices(dit_seq_shape, tile_size, device) -> torch.Tensor:
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    ls = []
    for t in range(math.ceil(T / ts)):
        for h in range(math.ceil(H / hs)):
            for w in range(math.ceil(W / ws)):
                ls.append(
                    indices[
                        t * ts : min(t * ts + ts, T),
                        h * hs : min(h * hs + hs, H),
                        w * ws : min(w * ws + ws, W),
                    ].flatten()
                )
    return torch.cat(ls, dim=0)


def construct_variable_block_sizes(dit_seq_shape, num_tiles, device) -> torch.Tensor:
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = VSA_H3_TILE_SHAPE
    n_t, n_h, n_w = num_tiles

    def _sizes(dim_len: int, tile: int, n: int) -> torch.Tensor:
        sizes = torch.full((n,), tile, dtype=torch.int, device=device)
        remainder = dim_len - (n - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    return (
        _sizes(t, ts_t, n_t)[:, None, None]
        * _sizes(h, ts_h, n_h)[None, :, None]
        * _sizes(w, ts_w, n_w)[None, None, :]
    ).reshape(-1)


def get_non_pad_index(variable_block_sizes: torch.Tensor, max_block_size: int):
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts_pad = torch.arange(n_win, device=device) * max_block_size
    index_pad = starts_pad[:, None] + torch.arange(max_block_size, device=device)[None, :]
    index_mask = (
        torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    )
    return index_pad[index_mask]


# --- video_sparse_attn_h3.py ------------------------------------------------ #
@dataclass
class Meta:
    VSA_sparsity: float
    total_seq_length: int
    num_prefix_tiles: int
    num_video_tiles: int
    exempt: bool
    variable_block_sizes: torch.Tensor  # int32 [n_tiles]
    pack_index: torch.Tensor  # int32 [S_pad]: padded pos -> packed row or -1
    unpack_index: torch.Tensor  # int32 [used]: packed row -> padded pos

    @property
    def num_tiles(self) -> int:
        return self.num_prefix_tiles + self.num_video_tiles


def h3_tile_geometry(prefix_segments, dit_seq_shape, device):
    prefix_len = sum(prefix_segments)
    prefix_sizes: list[int] = []
    for segment in prefix_segments:
        full, remainder = divmod(segment, VSA_H3_TILE_ELEMS)
        prefix_sizes.extend([VSA_H3_TILE_ELEMS] * full)
        if remainder:
            prefix_sizes.append(remainder)
    num_prefix_tiles = len(prefix_sizes)

    num_tiles = tuple(math.ceil(d / t) for d, t in zip(dit_seq_shape, VSA_H3_TILE_SHAPE))
    video_sizes = construct_variable_block_sizes(dit_seq_shape, num_tiles, device)
    num_video_tiles = int(video_sizes.numel())

    video_indices = get_tile_partition_indices(dit_seq_shape, VSA_H3_TILE_SHAPE, device) + prefix_len
    tile_partition_indices = torch.cat(
        [torch.arange(prefix_len, device=device, dtype=torch.long), video_indices]
    )
    variable_block_sizes = torch.cat(
        [torch.tensor(prefix_sizes, dtype=torch.long, device=device), video_sizes.to(torch.long)]
    )
    non_pad_index = get_non_pad_index(variable_block_sizes, VSA_H3_TILE_ELEMS)

    total = prefix_len + math.prod(dit_seq_shape)
    if int(variable_block_sizes.sum()) != total or non_pad_index.numel() != total:
        raise ValueError("VSA-H3 tile geometry mismatch")

    seq_pad = variable_block_sizes.numel() * VSA_H3_TILE_ELEMS
    pack_index = torch.full((seq_pad,), -1, dtype=torch.int32, device=device)
    pack_index[non_pad_index] = tile_partition_indices.to(torch.int32)
    unpack_index = torch.empty(total, dtype=torch.int32, device=device)
    unpack_index[tile_partition_indices] = non_pad_index.to(torch.int32)
    return variable_block_sizes.to(torch.int32), pack_index, unpack_index, num_prefix_tiles, num_video_tiles


def build_meta(raw_latent_shape, patch_size, VSA_sparsity, prefix_segments, device, exempt=True) -> Meta:
    dit_seq_shape = tuple(r // p for r, p in zip(raw_latent_shape, patch_size))
    prefix_segments = tuple(int(s) for s in prefix_segments if s > 0)
    vbs, pack_index, unpack_index, n_prefix, n_video = h3_tile_geometry(
        prefix_segments, dit_seq_shape, device
    )
    return Meta(
        VSA_sparsity=float(VSA_sparsity),
        total_seq_length=sum(prefix_segments) + math.prod(dit_seq_shape),
        num_prefix_tiles=n_prefix,
        num_video_tiles=n_video,
        exempt=exempt,
        variable_block_sizes=vbs,
        pack_index=pack_index,
        unpack_index=unpack_index,
    )


def compute_topk(sparsity: float, num_video_tiles: int) -> int:
    keep = math.ceil((1.0 - sparsity) * num_video_tiles)
    return max(1, min(keep, num_video_tiles))


def topk_tile_lists(scores, num_prefix_tiles, num_video_tiles, sparsity, exempt) -> torch.Tensor:
    prefix = num_prefix_tiles
    keep = compute_topk(sparsity, num_video_tiles)
    video_rows = scores[:, prefix:, :]
    if exempt or prefix == 0:
        picked = video_rows[:, :, prefix:].topk(keep, dim=-1).indices + prefix
        picked = picked.sort(dim=-1).values
        if prefix == 0:
            return picked.to(torch.int32)
        prefix_cols = torch.arange(prefix, device=scores.device).expand(*picked.shape[:-1], prefix)
        return torch.cat([prefix_cols, picked], dim=-1).to(torch.int32)
    keep_total = min(keep + prefix, scores.shape[-1])
    return video_rows.topk(keep_total, dim=-1).indices.sort(dim=-1).values.to(torch.int32)


class Workspace:
    """Per-geometry scratch, same layout as the backend's ``_Workspace``."""

    def __init__(self, meta: Meta, heads: int, head_dim: int, dtype, device) -> None:
        n_tiles = meta.num_tiles
        seq_pad = n_tiles * VSA_H3_TILE_ELEMS
        self.tiled = torch.empty((3, heads, seq_pad, head_dim), dtype=dtype, device=device)
        self.pooled = torch.empty((3, heads, n_tiles, head_dim), dtype=torch.float32, device=device)
        self.out_tiled = torch.empty((heads, seq_pad, head_dim), dtype=dtype, device=device)
        all_tiles = torch.arange(n_tiles, dtype=torch.int32, device=device)
        self.dense_index = all_tiles.repeat(heads, n_tiles, 1)
        self.dense_num = torch.full((heads, n_tiles), n_tiles, dtype=torch.int32, device=device)
        self.q2k_index = self.dense_index.clone()
        self.q2k_num = self.dense_num.clone()

    def sparse_lists(self, video_lists: torch.Tensor, num_prefix_tiles: int):
        width = video_lists.shape[-1]
        self.q2k_index[:, num_prefix_tiles:, :width] = video_lists
        self.q2k_num[:, num_prefix_tiles:] = width
        return self.q2k_index, self.q2k_num


def select_kv_lists(ws: Workspace, meta: Meta, scores, sparsity: float):
    keep = compute_topk(sparsity, meta.num_video_tiles)
    if sparsity <= 0.0 or keep >= meta.num_video_tiles:
        return ws.dense_index, ws.dense_num
    return ws.sparse_lists(
        topk_tile_lists(scores, meta.num_prefix_tiles, meta.num_video_tiles, sparsity, meta.exempt),
        meta.num_prefix_tiles,
    )


def verify_against_sglang(meta: Meta, raw_latent_shape, patch_size, prefix_segments, device) -> str:
    """Cross-check the shim against the real builder when sglang is installed."""
    try:
        from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn_h3 import (
            VideoSparseAttentionH3MetadataBuilder,
        )
    except Exception as e:  # stub package tree in the docker image
        return f"skipped ({type(e).__name__})"
    real = VideoSparseAttentionH3MetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=raw_latent_shape,
        patch_size=patch_size,
        VSA_sparsity=meta.VSA_sparsity,
        prefix_segments=prefix_segments,
        device=device,
        exempt=meta.exempt,
    )
    same = (
        real.num_prefix_tiles == meta.num_prefix_tiles
        and real.num_video_tiles == meta.num_video_tiles
        and torch.equal(real.variable_block_sizes, meta.variable_block_sizes)
        and torch.equal(real.pack_index, meta.pack_index)
        and torch.equal(real.unpack_index, meta.unpack_index)
    )
    return "match" if same else "MISMATCH with sglang builder"

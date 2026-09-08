# SPDX-License-Identifier: Apache-2.0
"""Synthetic benchmark for the VSA-H3 block-sparse attention Triton kernel.

Generates random q/k/v plus a random top-k block index list and measures the
achieved TFLOPS of ``vsa_h3_block_sparse_attn_forward``.

Executed FLOPs are counted per *selected* 64x64 block pair (the kernel always
computes full BLOCK_M x BLOCK_N tiles; ragged tails are masked after the dot):

    flops = 4 * BLOCK * BLOCK * head_dim * sum(q2k_num)

(2*B*M*N*D for Q@K^T plus 2*B*M*N*D for P@V.)

Example:
    python benchmark/vsa_h3/bench_vsa_h3_kernel.py \
        --seq-lens 8192 32768 131072 --topk 32 64 128

Production reference point (MiniMax-H3, 4x H200, tp=2/ulysses=2, VSA 0.9),
measured in the serving nsys trace (h3+sparse.sqlite), steady state:

    B=1 H=14 D=128 seq=42752 (668 tiles) topk~=67 -> 4.05 ms ~= 324 TFLOPS

i.e. ~33% of H200 bf16 peak; dense FA3 on the same shape reaches ~83%.
Reproduce with: --heads 14 --seq-lens 42752 --topk 67
"""

import argparse

import torch
import triton

from sglang.multimodal_gen.runtime.layers.attention.backends.vsa_h3_kernels import (
    VSA_H3_KERNEL_BLOCK,
    vsa_h3_block_sparse_attn_forward,
)


def make_inputs(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    topk: int,
    ragged: bool,
    device: str = "cuda",
):
    block = VSA_H3_KERNEL_BLOCK
    if seq_len % block:
        raise ValueError(f"seq_len must be a multiple of {block}, got {seq_len}")
    n_tiles = seq_len // block
    n_kv_blocks = n_tiles
    if topk > n_kv_blocks:
        raise ValueError(f"topk={topk} exceeds n_kv_blocks={n_kv_blocks}")

    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

    # Random distinct key-block indices per query block, sorted ascending so
    # each q-block attends to `topk` distinct kv blocks. The ascending sort
    # matches production (`_topk_tile_lists` sorts picked indices) and keeps
    # K/V read locality comparable to serving.
    rand = torch.rand(batch, heads, n_tiles, n_kv_blocks, device=device)
    q2k_index = (
        rand.argsort(dim=-1)[..., :topk].sort(dim=-1).values.to(torch.int32).contiguous()
    )
    q2k_num = torch.full(
        (batch, heads, n_tiles), topk, dtype=torch.int32, device=device
    )

    if ragged:
        variable_block_sizes = torch.randint(
            1, block + 1, (n_kv_blocks,), dtype=torch.int32, device=device
        )
    else:
        variable_block_sizes = torch.full(
            (n_kv_blocks,), block, dtype=torch.int32, device=device
        )

    return q, k, v, q2k_index, q2k_num, variable_block_sizes


def bench_one(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    topk: int,
    ragged: bool,
    warmup: int,
    rep: int,
):
    q, k, v, q2k_index, q2k_num, variable_block_sizes = make_inputs(
        batch, heads, seq_len, head_dim, topk, ragged
    )
    fn = lambda: vsa_h3_block_sparse_attn_forward(
        q, k, v, q2k_index, q2k_num, variable_block_sizes
    )
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    block = VSA_H3_KERNEL_BLOCK
    selected_blocks = int(q2k_num.sum().item())
    flops = 4 * block * block * head_dim * selected_blocks
    tflops = flops / (ms * 1e-3) / 1e12

    # Bytes touched: q/out once + k/v once per selected block pair.
    nbytes = (
        2 * q.numel() * q.element_size()
        + 2 * selected_blocks * block * head_dim * k.element_size()
    )
    gbps = nbytes / (ms * 1e-3) / 1e9

    return ms, tflops, gbps, selected_blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[8192, 32768, 131072])
    parser.add_argument("--topk", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument(
        "--ragged",
        action="store_true",
        help="randomize variable_block_sizes in [1, 64] instead of all-full blocks",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    print(
        f"VSA-H3 block-sparse attn: B={args.batch} H={args.heads} "
        f"D={args.head_dim} block={VSA_H3_KERNEL_BLOCK} ragged={args.ragged}"
    )
    header = f"{'seq':>8} {'topk':>6} {'sel_blks':>10} {'ms':>10} {'TFLOPS':>9} {'GB/s':>9}"
    print(header)
    print("-" * len(header))
    for seq_len in args.seq_lens:
        for topk in args.topk:
            if topk > seq_len // VSA_H3_KERNEL_BLOCK:
                continue
            ms, tflops, gbps, selected = bench_one(
                args.batch,
                args.heads,
                seq_len,
                args.head_dim,
                topk,
                args.ragged,
                args.warmup,
                args.rep,
            )
            print(f"{seq_len:>8} {topk:>6} {selected:>10} {ms:>10.3f} {tflops:>9.1f} {gbps:>9.1f}")


if __name__ == "__main__":
    main()

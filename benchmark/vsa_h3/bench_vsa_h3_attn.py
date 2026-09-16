# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for the VSA-H3 block-sparse attention kernel on the real
MiniMax-H3 tile geometry (route A of plan/minimax_perf.md, item 1).

What it does
------------
1. Builds the per-step VSA-H3 metadata like the denoising stage does
   (``vsa_h3_geometry.py`` is a standalone copy of the backend's builder,
   cross-checked against the real one when sglang is importable) for a clip:
   frames x height x width -> latent -> (1, 2, 2) patches -> (4, 4, 4) tiles,
   plus segment-pure prefix tiles for text / condition / audio rows.
2. Synthesizes packed q/k/v with a tunable amount of spatial locality so the
   top-k lists look like real ones (nearby tiles score high), or loads real
   tensors dumped from a server (``--load``).
3. Runs the same pack -> pooled scores -> top-k -> kernel path as
   ``VideoSparseAttentionH3Impl.forward_varlen`` and times only the attention
   kernel for every selected variant, reporting ms, achieved TFLOPS, compiled
   registers / spills / smem, and the max error against an fp32 per-tile
   reference and against the production kernel.

Usage (GPU box, repo root, sglang importable)
---------------------------------------------
  python benchmark/vsa_h3/bench_vsa_h3_attn.py                     # prod + variants, default geometry
  python benchmark/vsa_h3/bench_vsa_h3_attn.py --sweep --warps 4,8 --stages 2,3,4,5
  python benchmark/vsa_h3/bench_vsa_h3_attn.py --variants prod,v1:prefetch --check-tiles 256
  python benchmark/vsa_h3/bench_vsa_h3_attn.py --dump /tmp/vsa_inputs.pt
  ncu --set full -k regex:_attn_fwd_sparse -c 2 \
      python benchmark/vsa_h3/bench_vsa_h3_attn.py --variants prod,v1:prefetch --ncu

Variant names
-------------
  prod                 production launcher (its own warps/stages choice)
  base                 production kernel, explicit --warps/--stages (for --sweep)
  v1[:flag[+flag...]]  experimental kernel from vsa_h3_variants.py; flags:
                       prefetch, fold, skipmask   (e.g. v1:prefetch+fold)

Geometry defaults reproduce the profiled run (minimax_reports/h3+sparse.sqlite):
4 x H200, Ulysses 4 -> 14 heads per rank, head_dim 128, 1344x768, 124 frames,
sparsity 0.9, exempt mode. The trace records only kernel grids, so the tile
layout is recovered from them: ``_attn_fwd_sparse`` / ``_pack_tiles_kernel``
grid.x = 668 tiles and ``_untile_kernel`` grid.x = 590 -> 37697..37760 packed
rows. With a 24 x 42 latent plane (768/16/2 x 1344/16/2 -> 6 x 11 tiles) the
only consistent solution is latent_t = 37 (10 temporal tiles, 660 video tiles),
8 prefix tiles and 449..464 prefix rows, i.e. ~456 text tokens and no audio
rows inside the joint attention. Hence ``--latent-t 37`` and ``--text-len 456``
defaults; pass ``--latent-t -1`` to derive latent_t from ``--frames`` instead,
or ``--load`` a dump of real tensors.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Only the kernel file is taken from sglang (the docker image ships it under a
# stub package tree); geometry / top-k / workspace come from the local shim.
from sglang.multimodal_gen.runtime.layers.attention.backends.vsa_h3_kernels import (  # noqa: E402
    _attn_fwd_sparse,
    vsa_h3_block_sparse_attn_forward,
    vsa_h3_pack_tiles,
)

import vsa_h3_variants  # noqa: E402
from vsa_h3_geometry import (  # noqa: E402
    VSA_H3_TILE_ELEMS,
    VSA_H3_TILE_SHAPE,
    Workspace,
    compute_topk,
    select_kv_lists,
    verify_against_sglang,
)
from vsa_h3_geometry import build_meta as build_h3_meta  # noqa: E402

BLOCK = VSA_H3_TILE_ELEMS  # 64
TRACED_Q_TILES = 668  # grid.x of _attn_fwd_sparse in minimax_reports/h3+sparse.sqlite
TRACED_UNTILE_GRID = 590  # grid.x of _untile_kernel there -> 37697..37760 packed rows


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def latent_shape(frames: int, height: int, width: int) -> tuple[int, int, int]:
    """MiniMax-H3 video VAE: temporal 4x (causal, first frame alone), spatial 16x."""
    latent_t = (frames - 1) // 4 + 1
    return latent_t, height // 16, width // 16


def video_tiles(dit_seq_shape: tuple[int, int, int]) -> int:
    return math.prod(
        math.ceil(d / t) for d, t in zip(dit_seq_shape, VSA_H3_TILE_SHAPE)
    )


def prefix_tiles(segments: tuple[int, ...]) -> int:
    return sum(math.ceil(s / BLOCK) for s in segments if s > 0)


def build_meta(args, device):
    raw = latent_shape(args.frames, args.height, args.width)
    if args.latent_t is not None and args.latent_t > 0:
        raw = (args.latent_t, raw[1], raw[2])
    patch = (1, 2, 2)
    dit_seq_shape = tuple(r // p for r, p in zip(raw, patch))
    n_video = video_tiles(dit_seq_shape)

    text_len, cond_rows, audio_rows = args.text_len, args.cond_rows, args.audio_rows
    if args.n_tiles is not None:
        need = args.n_tiles - n_video - prefix_tiles((text_len, cond_rows))
        if need < 0:
            raise SystemExit(
                f"--n-tiles {args.n_tiles} < video tiles {n_video} + text/cond tiles"
            )
        # fill the audio segment so its tile count closes the gap (last tile ragged
        # like a real audio segment would be)
        audio_rows = max(0, need * BLOCK - (BLOCK - args.audio_tail)) if need else 0
    segments = (text_len, cond_rows, audio_rows)
    meta = build_h3_meta(raw, patch, args.sparsity, segments, device, exempt=args.mode == "exempt")
    geometry_check = verify_against_sglang(meta, raw, patch, segments, device)
    info = dict(
        geometry_vs_sglang=geometry_check,
        latent=raw,
        dit_seq_shape=dit_seq_shape,
        prefix_segments=(text_len, cond_rows, audio_rows),
        num_prefix_tiles=meta.num_prefix_tiles,
        num_video_tiles=meta.num_video_tiles,
        num_tiles=meta.num_tiles,
        seq_pad=meta.num_tiles * BLOCK,
        used=meta.total_seq_length,
    )
    return meta, info


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def synth_packed_qkv(meta, info, heads, head_dim, locality, seed, device):
    """Packed [used, H, D] bf16 q/k/v. ``locality`` adds a smooth per-tile field
    shared by q and k so pooled scores favour spatially nearby video tiles."""
    gen = torch.Generator(device=device).manual_seed(seed)
    used = info["used"]

    def rand(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)

    q, k, v = rand(used, heads, head_dim), rand(used, heads, head_dim), rand(
        used, heads, head_dim
    )
    if locality > 0:
        nt = tuple(math.ceil(d / t) for d, t in zip(info["dit_seq_shape"], VSA_H3_TILE_SHAPE))
        field = rand(heads, head_dim, *nt)  # [H, D, Tt, Th, Tw]
        for _ in range(2):  # low-pass over the tile grid
            field = torch.nn.functional.avg_pool3d(
                field, kernel_size=3, stride=1, padding=1, count_include_pad=False
            )
        field = field / field.std() * locality
        video_field = field.flatten(2).permute(2, 0, 1)  # [n_video, H, D]
        prefix_field = rand(meta.num_prefix_tiles, heads, head_dim) * locality
        tile_field = torch.cat([prefix_field, video_field], 0)  # [n_tiles, H, D]

        padded_pos = torch.arange(meta.num_tiles * BLOCK, device=device)
        rows = meta.pack_index.long()
        valid = rows >= 0
        tile_of_row = torch.empty(used, dtype=torch.long, device=device)
        tile_of_row[rows[valid]] = padded_pos[valid] // BLOCK
        q = q + tile_field[tile_of_row]
        k = k + tile_field[tile_of_row]
    return tuple(t.to(torch.bfloat16) for t in (q, k, v))


def kernel_inputs(meta, q, k, v, sparsity):
    """Mirror forward_varlen: pack tiles, pooled scores, top-k lists."""
    ws = Workspace(meta, q.shape[1], q.shape[2], q.dtype, q.device)
    vsa_h3_pack_tiles(
        q, k, v, None, meta.pack_index, meta.variable_block_sizes, ws.tiled, ws.pooled
    )
    head_dim = q.shape[-1]
    q_pooled, k_pooled, _ = ws.pooled
    scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) * head_dim**-0.5
    q2k_index, q2k_num = select_kv_lists(ws, meta, scores, sparsity)
    return dict(
        q=ws.tiled[0:1].contiguous(),
        k=ws.tiled[1:2].contiguous(),
        v=ws.tiled[2:3].contiguous(),
        q2k_index=q2k_index[None].contiguous(),
        q2k_num=q2k_num[None].contiguous(),
        variable_block_sizes=meta.variable_block_sizes.contiguous(),
    )


def list_stats(inp, info):
    num = inp["q2k_num"][0]  # [H, n_tiles]
    idx = inp["q2k_index"][0]  # [H, n_tiles, max_kv]
    vbs = inp["variable_block_sizes"]
    n_tiles = num.shape[1]
    total_blocks = int(num.sum())
    heads = num.shape[0]
    flops = 4.0 * BLOCK * BLOCK * inp["q"].shape[-1] * total_blocks
    p = info["num_prefix_tiles"]
    video_num = num[:, p:]
    width = int(video_num[0, 0]) if video_num.numel() else 0
    # locality: mean |kv_tile - q_tile| over the video part of video rows
    q_tile = torch.arange(p, n_tiles, device=num.device)[None, :, None]
    vidx = idx[:, p:, :width].long()
    video_part = vidx >= p
    dist = (vidx - q_tile).abs().float()
    locality = float((dist * video_part).sum() / video_part.sum().clamp(min=1))
    ragged = int((vbs < BLOCK).sum())
    listed_ragged = float((vbs[idx.long()] < BLOCK).float().mean())
    return dict(
        heads=heads,
        max_kv=idx.shape[-1],
        keep=compute_topk(ARGS.sparsity, info["num_video_tiles"]),
        video_row_width=width,
        avg_blocks_per_qtile=total_blocks / (heads * n_tiles),
        total_blocks=total_blocks,
        tflop_per_call=flops / 1e12,
        mean_tile_distance=locality,
        ragged_tiles=ragged,
        listed_ragged_frac=listed_ragged,
    )


# --------------------------------------------------------------------------- #
# reference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reference_tiles(inp, sample):
    """fp32 attention of q-tile t over its listed kv tiles for (h, t) pairs."""
    q, k, v = inp["q"][0].float(), inp["k"][0].float(), inp["v"][0].float()
    idx, num, vbs = inp["q2k_index"][0], inp["q2k_num"][0], inp["variable_block_sizes"]
    head_dim = q.shape[-1]
    outs = []
    col = torch.arange(BLOCK, device=q.device)
    for h, t in sample:
        n = int(num[h, t])
        lst = idx[h, t, :n].long()
        rows = (lst[:, None] * BLOCK + col[None, :]).reshape(-1)
        k_sel, v_sel = k[h, rows], v[h, rows]  # [n*64, D]
        valid = (col[None, :] < vbs[lst][:, None]).reshape(-1)
        s = q[h, t * BLOCK : (t + 1) * BLOCK] @ k_sel.T / math.sqrt(head_dim)
        s = s.masked_fill(~valid[None, :], float("-inf"))
        outs.append(torch.softmax(s, dim=-1) @ v_sel)
    return outs


def pick_sample(inp, info, n, seed):
    gen = torch.Generator().manual_seed(seed)
    heads, n_tiles = inp["q2k_num"].shape[1:]
    p = info["num_prefix_tiles"]
    vbs = inp["variable_block_sizes"].cpu()
    ragged = torch.nonzero(vbs < BLOCK).flatten().tolist()
    sample = set()
    # always cover: a prefix (dense) row, a ragged tile row, first/last video tile
    for t in ([0] if p else []) + ragged[:4] + [p, n_tiles - 1]:
        sample.add((int(torch.randint(heads, (1,), generator=gen)), int(t)))
    while len(sample) < n:
        sample.add(
            (
                int(torch.randint(heads, (1,), generator=gen)),
                int(torch.randint(n_tiles, (1,), generator=gen)),
            )
        )
    return sorted(sample)


def max_err(out, refs, sample):
    err = 0.0
    for (h, t), ref in zip(sample, refs):
        tile = out[0, h, t * BLOCK : (t + 1) * BLOCK].float()
        err = max(err, float((tile - ref).abs().max()))
    return err


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def parse_variant(name: str):
    if name in ("prod", "base"):
        return name, {}
    if not name.startswith("v1"):
        raise SystemExit(f"unknown variant {name!r}")
    flags = name.split(":", 1)[1].split("+") if ":" in name else []
    allowed = {"prefetch": "prefetch_index", "fold": "fold_scale", "skipmask": "skip_full_mask"}
    bad = [f for f in flags if f not in allowed]
    if bad:
        raise SystemExit(f"unknown v1 flags {bad}; allowed {sorted(allowed)}")
    return "v1", {allowed[f]: True for f in flags}


def make_launcher(kind, flags, inp, out, num_warps, num_stages, maxnreg):
    q, k, v = inp["q"], inp["k"], inp["v"]
    idx, num, vbs = inp["q2k_index"], inp["q2k_num"], inp["variable_block_sizes"]
    if kind == "prod":

        def fn():
            return vsa_h3_block_sparse_attn_forward(q, k, v, idx, num, vbs, out=out)

        return fn
    if kind == "base":
        from triton.tools.tensor_descriptor import TensorDescriptor

        head_dim = q.shape[-1]
        block = [1, 1, BLOCK, head_dim]
        descs = [TensorDescriptor.from_tensor(t, block_shape=block) for t in (q, k, v, out)]
        grid = (q.shape[2] // BLOCK, q.shape[0] * q.shape[1], 1)
        extra = {"maxnreg": maxnreg} if maxnreg else {}

        def fn():
            return _attn_fwd_sparse[grid](
                *descs,
                1.0 / math.sqrt(head_dim),
                idx,
                num,
                idx.shape[-1],
                vbs,
                q.shape[1],
                q.shape[2],
                HEAD_DIM=head_dim,
                BLOCK_M=BLOCK,
                BLOCK_N=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
                **extra,
            )

        return fn

    def fn():
        return vsa_h3_variants.launch_v1(
            q, k, v, idx, num, vbs, out,
            num_warps=num_warps, num_stages=num_stages, maxnreg=maxnreg, **flags,
        )

    return fn


def compiled_info(handle):
    if handle is None:
        return "?", "?", "?"
    regs = getattr(handle, "n_regs", "?")
    spills = getattr(handle, "n_spills", "?")
    smem = getattr(getattr(handle, "metadata", None), "shared", "?")
    return regs, spills, smem


def time_fn(fn, warmup, rep):
    from triton.testing import do_bench

    # with quantiles given, do_bench returns [p50, p20, p80] in ms
    ms = do_bench(fn, warmup=warmup, rep=rep, quantiles=(0.5, 0.2, 0.8))
    return tuple(ms) if isinstance(ms, (list, tuple)) else (ms, ms, ms)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("geometry")
    g.add_argument("--frames", type=int, default=124)
    g.add_argument("--latent-t", type=int, default=37,
                   help="override latent frames (trace-consistent default 37); -1 = derive from --frames")
    g.add_argument("--height", type=int, default=768)
    g.add_argument("--width", type=int, default=1344)
    g.add_argument("--heads", type=int, default=14, help="heads per rank (56 / ulysses)")
    g.add_argument("--head-dim", type=int, default=128)
    g.add_argument("--text-len", type=int, default=456)
    g.add_argument("--cond-rows", type=int, default=0)
    g.add_argument("--audio-rows", type=int, default=0)
    g.add_argument("--n-tiles", type=int, default=None,
                   help="pad audio rows so the total tile count matches this value "
                        "(only for what-if runs; the default geometry already gives 668)")
    g.add_argument("--audio-tail", type=int, default=40, help="valid rows in the last (ragged) audio tile when padding")
    g.add_argument("--sparsity", type=float, default=0.9)
    g.add_argument("--mode", choices=("exempt", "compete"), default="exempt")
    g.add_argument("--locality", type=float, default=1.0, help="0 = iid random q/k (unrealistic, no L2 reuse structure)")
    g.add_argument("--seed", type=int, default=20260916)

    r = ap.add_argument_group("run")
    r.add_argument("--variants", default="prod,base,v1:prefetch,v1:prefetch+fold,v1:prefetch+fold+skipmask")
    r.add_argument("--warps", default="8", help="comma list for base/v1 (all combos with --stages when --sweep)")
    r.add_argument("--stages", default="3")
    r.add_argument("--maxnreg", type=int, default=None)
    r.add_argument("--sweep", action="store_true", help="run every warps x stages combo for base/v1 variants")
    r.add_argument("--warmup", type=int, default=25)
    r.add_argument("--rep", type=int, default=200)
    r.add_argument("--check-tiles", type=int, default=96, help="(h, tile) samples for the fp32 reference; 0 disables")
    r.add_argument("--tol", type=float, default=2e-2, help="max abs error vs fp32 reference (test_vsa_h3_attention uses 2e-2)")
    r.add_argument("--ncu", action="store_true", help="one warm-up + one timed launch per variant, no do_bench (for ncu/nsys)")
    r.add_argument("--dump", default=None, help="save kernel inputs (.pt) after synthesis")
    r.add_argument("--load", default=None, help="load kernel inputs (.pt) instead of synthesizing; keys: q,k,v,q2k_index,q2k_num,variable_block_sizes[,info]")
    return ap.parse_args()


def main():
    global ARGS
    ARGS = args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"device: {props.name} sm_{props.major}{props.minor} SMs={props.multi_processor_count}")

    if args.load:
        blob = torch.load(args.load, map_location=device)
        inp = {k: blob[k].to(device) for k in ("q", "k", "v", "q2k_index", "q2k_num", "variable_block_sizes")}
        info = blob.get("info") or {}
        n_tiles = inp["q2k_num"].shape[-1]
        info.setdefault("num_tiles", n_tiles)
        info.setdefault("num_prefix_tiles", int((inp["q2k_num"][0, 0] == n_tiles).sum()))
        info.setdefault("num_video_tiles", n_tiles - info["num_prefix_tiles"])
        print(f"loaded {args.load}: {info}")
    else:
        if args.n_tiles is not None and args.n_tiles < 0:
            args.n_tiles = None
        meta, info = build_meta(args, device)
        info["untile_grid_x"] = math.ceil(info["used"] / BLOCK)
        print("geometry:", info)
        if info["num_tiles"] != TRACED_Q_TILES or info["untile_grid_x"] != TRACED_UNTILE_GRID:
            print(f"  note: traced run had {TRACED_Q_TILES} tiles and untile grid {TRACED_UNTILE_GRID}")
        q, k, v = synth_packed_qkv(meta, info, args.heads, args.head_dim, args.locality, args.seed, device)
        inp = kernel_inputs(meta, q, k, v, args.sparsity)
        if args.dump:
            torch.save({**inp, "info": info}, args.dump)
            print(f"dumped kernel inputs to {args.dump}")

    stats = list_stats(inp, info)
    print("lists:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()})
    tflop = stats["tflop_per_call"]

    # reference output for the correctness check + production output for the diff
    out_prod = torch.empty_like(inp["q"])
    vsa_h3_block_sparse_attn_forward(
        inp["q"], inp["k"], inp["v"], inp["q2k_index"], inp["q2k_num"], inp["variable_block_sizes"], out=out_prod
    )
    torch.cuda.synchronize()
    sample, refs = [], []
    if args.check_tiles > 0:
        sample = pick_sample(inp, info, args.check_tiles, args.seed)
        refs = reference_tiles(inp, sample)
        print(f"prod vs fp32 reference on {len(sample)} tiles: max abs err {max_err(out_prod, refs, sample):.4e}")

    warps_list = [int(x) for x in args.warps.split(",")]
    stages_list = [int(x) for x in args.stages.split(",")]
    combos = [(w, s) for w in warps_list for s in stages_list] if args.sweep else [(warps_list[0], stages_list[0])]

    header = f"{'variant':<34} {'warps':>5} {'stg':>3} {'regs':>4} {'spill':>5} {'smem':>7} {'ms p50':>8} {'p20':>8} {'p80':>8} {'TFLOPS':>7} {'vs prod':>8} {'err ref':>9} {'ok':>3}"
    print(header)
    print("-" * len(header))
    results = []
    for name in args.variants.split(","):
        kind, flags = parse_variant(name)
        for num_warps, num_stages in combos if kind != "prod" else [(None, None)]:
            out = torch.empty_like(inp["q"])
            fn = make_launcher(kind, flags, inp, out, num_warps, num_stages, args.maxnreg)
            try:
                handle = fn()
                torch.cuda.synchronize()
            except Exception as e:  # compile or launch failure: report and continue
                print(f"{name:<34} {str(num_warps):>5} {str(num_stages):>3}  FAILED: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
                continue
            regs, spills, smem = compiled_info(handle)
            diff_prod = float((out.float() - out_prod.float()).abs().max())
            err = max_err(out, refs, sample) if sample else float("nan")
            ok = "yes" if (not sample or err < args.tol) else "NO"
            if args.ncu:
                torch.cuda.nvtx.range_push(f"vsa_h3::{name}::w{num_warps}s{num_stages}")
                fn()
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                p50 = p20 = p80 = float("nan")
            else:
                p50, p20, p80 = time_fn(fn, args.warmup, args.rep)
            tflops = tflop / (p50 / 1e3) if p50 == p50 else float("nan")
            print(
                f"{name:<34} {str(num_warps):>5} {str(num_stages):>3} {str(regs):>4} {str(spills):>5} {str(smem):>7} "
                f"{p50:8.3f} {p20:8.3f} {p80:8.3f} {tflops:7.1f} {diff_prod:8.2e} {err:9.2e} {ok:>3}"
            )
            results.append((name, num_warps, num_stages, p50, tflops, err))

    if results and not args.ncu:
        best = min((r for r in results if r[5] != r[5] or r[5] < args.tol), key=lambda r: r[3], default=None)
        if best:
            print(f"\nbest passing: {best[0]} warps={best[1]} stages={best[2]}  {best[3]:.3f} ms  {best[4]:.1f} TFLOPS")
    peak = {9: 989.0, 10: 2250.0}.get(props.major)
    if peak:
        print(f"(dense bf16 peak for this arch ~{peak:.0f} TFLOPS; FA3 dense on H100/H200 reaches ~800)")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Compare diffusion attention backends: operator speed/accuracy and end-to-end generation.

Two modes, both run on the current GPU:

``op``  times the attention backends on dense head_dim-128 inputs through the same
        ``AttentionImpl`` classes the server dispatches to, and checks accuracy against
        an FP32 SDPA reference. The FP8 row fails loudly if the backend fell back to
        cuDNN, so the number is always the FP8 kernel's.

            python benchmark/diffusion_fp8_attn/bench_fp8_attn.py op \
                --heads 56,14 --seqs 2048,4096,8192,16384,30272

``e2e`` runs ``sglang generate`` once per attention backend with the same seed and
        flags, reads the ``--perf-dump-path`` report, and compares every video against
        the first backend's output (aggregate and worst-frame SSIM / PSNR).

            python benchmark/diffusion_fp8_attn/bench_fp8_attn.py e2e \
                --backends fa,fp8_fa_sm120 --out-dir /tmp/h3-e2e

        The default generate flags are the single-RTX-4090 MiniMax-H3 recipe from
        docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx; override them with --generate-args.

Both modes print a Markdown table and write a JSON summary (``--json``).
"""

import argparse
import importlib.util
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

HEAD_DIM = 128
FP8_BACKEND = "fp8_fa_sm120"
CUDNN_BACKEND = "torch_cudnn_sdpa"
SAGE_BACKEND = "sage_attn"

# Video thresholds from the sglang-diffusion-performance skill (aggregate SSIM / PSNR dB).
VIDEO_SSIM_THRESHOLD = 0.92
VIDEO_PSNR_THRESHOLD_DB = 24.0

# 1x RTX 4090 MiniMax-H3 recipe (cookbook), minus prompt / backend / output flags.
DEFAULT_GENERATE_ARGS = (
    "--model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va "
    "--performance-mode memory --layerwise-offload-components dit,text_encoder "
    "--dit-offload-prefetch-size 1 --dit-layerwise-resident-layers 0 "
    "--enable-torch-compile false"
)
DEFAULT_PROMPT = "A cat walking on a sunny beach, gentle waves."
FALLBACK_LOG_PATTERN = re.compile(r"fp8_fa_sm120 attention: (.+?); using cuDNN SDPA")


# ----------------------------------------------------------------------------- op mode


def _parse_int_list(text):
    return [int(item) for item in text.split(",") if item]


def _make_impl(*, backend, num_heads):
    common = dict(
        num_heads=num_heads,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=HEAD_DIM**-0.5,
    )
    if backend == FP8_BACKEND:
        from sglang.multimodal_gen.runtime.layers.attention.backends.fp8_fa_sm120_attn import (
            FP8FlashAttentionSM120Impl,
        )

        return FP8FlashAttentionSM120Impl(**common)
    if backend == CUDNN_BACKEND:
        from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
            CudnnSDPAImpl,
        )

        return CudnnSDPAImpl(**common)
    if backend == SAGE_BACKEND:
        from sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn import (
            SageAttentionImpl,
        )

        return SageAttentionImpl(**common)
    raise ValueError(f"unknown backend {backend!r}")


def _available_op_backends(requested):
    backends = []
    for backend in requested:
        if (
            backend == SAGE_BACKEND
            and importlib.util.find_spec("sageattention") is None
        ):
            print(f"skip {SAGE_BACKEND}: the sageattention package is not installed")
            continue
        backends.append(backend)
    return backends


def _fused_qkv(*, sequence, num_heads, seed):
    """[1, S, H, D] views into one [S, 3*H*D] projection output, as the DiT passes them."""
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    width = num_heads * HEAD_DIM
    qkv = torch.randn(
        (sequence, 3 * width), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    views = [
        qkv[:, i * width : (i + 1) * width].view(1, sequence, num_heads, HEAD_DIM)
        for i in range(3)
    ]
    return views


def _reference(q, k, v):
    import torch

    def to_bhsd(t):
        return t[0].float().permute(1, 0, 2).unsqueeze(0)

    out = torch.nn.functional.scaled_dot_product_attention(
        to_bhsd(q), to_bhsd(k), to_bhsd(v)
    )
    return out.squeeze(0).permute(1, 0, 2).unsqueeze(0)


def _error_metrics(candidate, reference):
    import torch

    candidate = candidate.float()
    rel_rms = (candidate - reference).pow(2).mean().sqrt() / reference.pow(
        2
    ).mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(
        candidate.flatten(), reference.flatten(), dim=0
    )
    return float(rel_rms), float(cosine)


def _l2_flush_buffer():
    """5x L2, the margin sglang's marker.do_bench uses; small H3 shapes fit in L2 otherwise."""
    import torch

    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    return torch.empty(5 * l2_bytes // 4, dtype=torch.int32, device="cuda")


def _median_ms(fn, *, warmup, iters, flush):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        flush.zero_()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times)


def _bench_one_shape(*, backends, num_heads, sequence, args):
    q, k, v = _fused_qkv(sequence=sequence, num_heads=num_heads, seed=args.seed)
    reference = _reference(q, k, v) if sequence <= args.ref_max_seq else None
    flops = 4 * sequence * sequence * HEAD_DIM * num_heads
    rows = []
    for backend in backends:
        impl = _make_impl(backend=backend, num_heads=num_heads)
        if backend == FP8_BACKEND:
            reason = impl._fallback_reason(q[0], k[0], v[0])
            if reason is not None:
                raise RuntimeError(
                    f"{FP8_BACKEND} would fall back to cuDNN here: {reason}"
                )

        def run(impl=impl):
            return impl.forward(q, k, v, None)

        output = run()
        rel_rms, cosine = (
            _error_metrics(output, reference) if reference is not None else (None, None)
        )
        ms = _median_ms(run, warmup=args.warmup, iters=args.iters, flush=args.flush)
        rows.append(
            dict(
                backend=backend,
                heads=num_heads,
                seq=sequence,
                ms=ms,
                tflops=flops / ms / 1e9,
                rel_rms=rel_rms,
                cos=cosine,
            )
        )
    baseline = next((row for row in rows if row["backend"] == CUDNN_BACKEND), None)
    for row in rows:
        row["speedup_vs_cudnn"] = baseline["ms"] / row["ms"] if baseline else None
    return rows


def _breakeven(rows):
    """Smallest sequence per head count from which FP8 stays faster than cuDNN."""
    result = {}
    for heads in sorted({row["heads"] for row in rows}):
        fp8 = sorted(
            (
                row
                for row in rows
                if row["heads"] == heads and row["backend"] == FP8_BACKEND
            ),
            key=lambda row: row["seq"],
        )
        winning_from = None
        for row in fp8:
            if row["speedup_vs_cudnn"] is not None and row["speedup_vs_cudnn"] > 1.0:
                winning_from = winning_from or row["seq"]
            else:
                winning_from = None
        result[heads] = winning_from
    return result


def _fmt(value, spec):
    return "-" if value is None else format(value, spec)


def run_op(args):
    import torch

    props = torch.cuda.get_device_properties(0)
    device = f"{props.name} (sm_{props.major}{props.minor})"
    backends = _available_op_backends(args.backends.split(","))
    args.flush = _l2_flush_buffer()
    rows = []
    for heads in _parse_int_list(args.heads):
        for sequence in _parse_int_list(args.seqs):
            rows.extend(
                _bench_one_shape(
                    backends=backends, num_heads=heads, sequence=sequence, args=args
                )
            )
            torch.cuda.empty_cache()

    print(f"\nDevice: {device}, head_dim {HEAD_DIM}, batch 1, non-causal, cold L2\n")
    print("| backend | H | S | ms | TFLOPS | vs cuDNN | rel RMS | cos |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['backend']} | {row['heads']} | {row['seq']} | {row['ms']:.2f} | "
            f"{row['tflops']:.0f} | {_fmt(row['speedup_vs_cudnn'], '.2f')}x | "
            f"{_fmt(row['rel_rms'], '.4f')} | {_fmt(row['cos'], '.5f')} |"
        )
    breakeven = (
        _breakeven(rows)
        if FP8_BACKEND in backends and CUDNN_BACKEND in backends
        else {}
    )
    for heads, seq in breakeven.items():
        verdict = f"from S={seq}" if seq else "never in the measured range"
        print(f"\n{FP8_BACKEND} beats {CUDNN_BACKEND} at H={heads}: {verdict}")
    _write_json(
        args.json, dict(mode="op", device=device, rows=rows, breakeven=breakeven)
    )


# ---------------------------------------------------------------------------- e2e mode


def _generate_command(*, backend, args, out_dir):
    return [
        sys.executable,
        "-c",
        # Same entry point as the `sglang` console script, pinned to this interpreter.
        "from sglang.cli.main import main; main()",
        "generate",
        *shlex.split(args.generate_args),
        "--attention-backend",
        backend,
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
        "--save-output",
        "--output-path",
        str(out_dir),
        "--output-file-name",
        f"{backend}.mp4",
        "--perf-dump-path",
        str(out_dir / "perf.json"),
    ]


def _run_generate(*, backend, args):
    out_dir = Path(args.out_dir) / backend
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        *args.launcher,
        *_generate_command(backend=backend, args=args, out_dir=out_dir),
    ]
    log_path = out_dir / "generate.log"
    print(
        f"\n[{backend}] {shlex.join(command)}\n[{backend}] log: {log_path}", flush=True
    )
    started = time.monotonic()
    with open(log_path, "w") as log:
        returncode = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    wall_s = time.monotonic() - started
    log_text = log_path.read_text(errors="replace")
    problem = None
    if returncode != 0:
        problem = f"sglang generate exited with {returncode}"
    elif not (out_dir / "perf.json").exists():
        problem = "sglang generate wrote no perf report"
    if problem:
        tail = "\n".join(log_text.splitlines()[-40:])
        raise RuntimeError(f"[{backend}] {problem}; last log lines:\n{tail}")
    return out_dir, wall_s, log_text


def _perf_summary(*, perf_path, skip_steps):
    report = json.loads(perf_path.read_text())
    steps = [item["duration_ms"] for item in report.get("denoise_steps_ms", [])]
    steady = steps[skip_steps:] or steps
    stages = {item["name"]: item["duration_ms"] for item in report.get("steps", [])}
    return dict(
        total_s=report["total_duration_ms"] / 1000,
        denoise_s=sum(steps) / 1000,
        step_median_ms=statistics.median(steady) if steady else None,
        num_steps=len(steps),
        stages_ms=stages,
    )


def _find_video(out_dir, backend):
    expected = out_dir / f"{backend}.mp4"
    if expected.exists():
        return expected
    videos = sorted(out_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no .mp4 written under {out_dir}")
    return videos[0]


def _read_frames(path):
    import imageio.v3 as iio

    return iio.imread(path, plugin="pyav") if _has_pyav() else iio.imread(path)


def _has_pyav():
    return importlib.util.find_spec("av") is not None


def _video_quality(candidate_path, reference_path):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    candidate = _read_frames(candidate_path)
    reference = _read_frames(reference_path)
    if candidate.shape != reference.shape:
        raise ValueError(f"frame shapes differ: {candidate.shape} vs {reference.shape}")
    ssim, psnr = [], []
    for got, want in zip(candidate, reference):
        ssim.append(structural_similarity(want, got, channel_axis=-1, data_range=255))
        value = peak_signal_noise_ratio(want, got, data_range=255)
        psnr.append(100.0 if math.isinf(value) else value)
    return dict(
        ssim_mean=statistics.fmean(ssim),
        ssim_min=min(ssim),
        psnr_mean=statistics.fmean(psnr),
        psnr_min=min(psnr),
        frames=len(ssim),
    )


def _fallback_reasons(log_text):
    return sorted(set(FALLBACK_LOG_PATTERN.findall(log_text)))


def run_e2e(args):
    backends = args.backends.split(",")
    results = []
    for backend in backends:
        out_dir, wall_s, log_text = _run_generate(backend=backend, args=args)
        perf = _perf_summary(
            perf_path=out_dir / "perf.json", skip_steps=args.skip_steps
        )
        results.append(
            dict(
                backend=backend,
                wall_s=wall_s,
                video=str(_find_video(out_dir, backend)),
                fp8_fallbacks=_fallback_reasons(log_text)
                if backend == FP8_BACKEND
                else [],
                **perf,
            )
        )

    baseline = results[0]
    for row in results:
        row["speedup_denoise"] = baseline["denoise_s"] / row["denoise_s"]
        row["speedup_total"] = baseline["total_s"] / row["total_s"]
        row["quality"] = (
            None if row is baseline else _video_quality(row["video"], baseline["video"])
        )

    print(
        f"\nBaseline: {baseline['backend']}; quality is measured against its video.\n"
    )
    print(
        "| backend | denoise s | median step ms | total s | denoise speedup | total speedup "
        "| SSIM mean / min | PSNR mean / min dB | passes 0.92 / 24 dB |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in results:
        quality = row["quality"]
        if quality is None:
            ssim_text = psnr_text = verdict = "baseline"
        else:
            ssim_text = f"{quality['ssim_mean']:.4f} / {quality['ssim_min']:.4f}"
            psnr_text = f"{quality['psnr_mean']:.2f} / {quality['psnr_min']:.2f}"
            passes = (
                quality["ssim_mean"] >= VIDEO_SSIM_THRESHOLD
                and quality["psnr_mean"] >= VIDEO_PSNR_THRESHOLD_DB
            )
            verdict = "yes" if passes else "no"
        print(
            f"| {row['backend']} | {row['denoise_s']:.1f} | {_fmt(row['step_median_ms'], '.0f')} | "
            f"{row['total_s']:.1f} | {row['speedup_denoise']:.2f}x | {row['speedup_total']:.2f}x | "
            f"{ssim_text} | {psnr_text} | {verdict} |"
        )
    for row in results:
        if row["fp8_fallbacks"]:
            print(
                f"\n{FP8_BACKEND} fell back to cuDNN for some calls: {row['fp8_fallbacks']}"
            )
    _write_json(args.json, dict(mode="e2e", baseline=baseline["backend"], rows=results))


# --------------------------------------------------------------------------------- cli


def _write_json(path, payload):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nJSON summary: {path}")


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    op = sub.add_parser("op", help="operator-level speed and accuracy")
    op.add_argument(
        "--backends", default=f"{FP8_BACKEND},{CUDNN_BACKEND},{SAGE_BACKEND}"
    )
    op.add_argument(
        "--heads", default="56,14", help="H3 runs 56 heads at TP1, 14 at TP4"
    )
    op.add_argument("--seqs", default="2048,4096,8192,16384,20440,30272,60000")
    op.add_argument(
        "--ref-max-seq",
        type=int,
        default=32768,
        help="skip the FP32 reference above this length (it is slow)",
    )
    op.add_argument("--warmup", type=int, default=3)
    op.add_argument("--iters", type=int, default=10)
    op.add_argument("--seed", type=int, default=0)
    op.add_argument("--json", default="")

    e2e = sub.add_parser(
        "e2e", help="sglang generate per backend, time and video quality"
    )
    e2e.add_argument(
        "--backends",
        default=f"fa,{FP8_BACKEND}",
        help="the first one is the quality and speed baseline",
    )
    e2e.add_argument("--generate-args", default=DEFAULT_GENERATE_ARGS)
    e2e.add_argument("--prompt", default=DEFAULT_PROMPT)
    e2e.add_argument("--seed", type=int, default=42)
    e2e.add_argument(
        "--skip-steps",
        type=int,
        default=2,
        help="denoise steps left out of the median (first-shape compile)",
    )
    e2e.add_argument("--out-dir", default="outputs/fp8_attn_e2e")
    e2e.add_argument(
        "--launcher",
        type=shlex.split,
        default=[],
        help='prefix for the generate command, e.g. "env CUDA_VISIBLE_DEVICES=0"',
    )
    e2e.add_argument("--json", default="")
    return parser


def main():
    args = _build_parser().parse_args()
    if args.mode == "op":
        run_op(args)
    else:
        if not args.json:
            args.json = os.path.join(args.out_dir, "summary.json")
        run_e2e(args)


if __name__ == "__main__":
    main()

# fp8_fa_sm120 attention on a rented GPU

The image built by `.github/workflows/diffusion-fp8-attn.yml` contains SGLang from this
branch, the `fp8_fa_sm120` unit tests and `bench_fp8_attn.py`. It is meant for SM89
(RTX 4090, L4, L40S) and SM120 (RTX 5090, RTX PRO 6000) GPUs.

## 1. Build the image

Push this branch to your fork. Every push to a `diffusion/fp8-fa-sm89*` branch builds
the image and pushes two tags to GHCR; the run summary prints both:

```text
ghcr.io/<owner>/sglang-diffusion-fp8-attn:<commit sha, 12 chars>
ghcr.io/<owner>/sglang-diffusion-fp8-attn:diffusion-fp8-fa-sm89-build
```

GHCR packages are private by default. Either make the package public in its GitHub
settings, or add registry credentials in RunPod (a GitHub token with `read:packages`).

## 2. Start a RunPod pod

- GPU: RTX 4090, L40S or L4 for SM89; RTX 5090 or RTX PRO 6000 for SM120.
- Container image: one of the tags above.
- Leave the start command empty: the image runs `sleep infinity` so the pod stays up.
- Attach a volume at `/workspace` if you will run the end-to-end comparison; model
  weights are large.

Open a terminal in the pod. The checkout is in `/sgl-workspace/sglang`.

```bash
cd /sgl-workspace/sglang
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## 3. Unit tests

```bash
python -m pytest -v -rs python/sglang/multimodal_gen/test/unit/test_fp8_fa_sm120_attn.py
```

On SM89 expect 9 passed and 1 skipped: the bit-equality test needs the block-scaled
MMA, which only SM120 / SM121 have. On SM120 all 10 should pass.

## 4. Operator benchmark

Times `fp8_fa_sm120`, cuDNN SDPA and SageAttention (when installed) through the
server's attention classes, with a cold L2, and checks accuracy against FP32 SDPA.
H3 uses 56 heads at TP1 and 14 at TP4.

```bash
mkdir -p /workspace/results
python benchmark/diffusion_fp8_attn/bench_fp8_attn.py op \
  --heads 56,14 --seqs 2048,4096,8192,16384,20440,30272,60000 \
  --json /workspace/results/op.json | tee /workspace/results/op.md
```

The table ends with the sequence length from which FP8 beats cuDNN for each head count.
To compare with SageAttention, install it first (`pip install sageattention`).

## 5. End-to-end comparison

Runs `sglang generate` once per attention backend with the same seed and flags, then
compares each video with the first backend's output. The defaults are the single
RTX 4090 MiniMax-H3 recipe from `docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx`.

```bash
export HF_HOME=/workspace/hf
export HF_TOKEN=...   # only for gated repos
python benchmark/diffusion_fp8_attn/bench_fp8_attn.py e2e \
  --backends fa,fp8_fa_sm120 \
  --out-dir /workspace/results/e2e
```

Add `sage_attn` to `--backends` only after installing SageAttention. Use
`--generate-args "..."` to replace the model flags, for example a smaller model:

```bash
python benchmark/diffusion_fp8_attn/bench_fp8_attn.py e2e \
  --backends fa,fp8_fa_sm120 \
  --generate-args "--model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --num-inference-steps 10 --num-frames 17 --height 480 --width 832" \
  --out-dir /workspace/results/e2e-wan
```

The table reports denoise time, median step time, total time, speedup against the
first backend, and aggregate and worst-frame SSIM / PSNR. The pass column uses the
video thresholds from the diffusion performance guide: SSIM 0.92 and PSNR 24 dB.
Each backend's log and perf report are in `<out-dir>/<backend>/`.

Known issue: on an L4 the `fp8_fa_sm120` run of Wan2.1-T2V-1.3B failed while loading the
transformer weights (`Tensor.item() cannot be called on meta tensors`), before any
attention call; the `fa` run of the same command worked. Check
`<out-dir>/fp8_fa_sm120/generate.log` if you hit it.

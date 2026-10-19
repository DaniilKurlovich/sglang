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

Create the pod from a blank custom template, not from a RunPod PyTorch template:
a template keeps its own image and you end up in a different container.

- GPU: RTX 4090, L40S or L4 for SM89; RTX 5090 or RTX PRO 6000 for SM120.
- Container image: one of the tags above, owner name in lower case.
- Container start command: leave empty. The image's start script keeps the pod up.
- Expose TCP port 22 for SSH. RunPod passes the SSH key from your account settings as
  `PUBLIC_KEY`; the start script adds it and starts sshd.
- Attach a volume at `/workspace` if you will run the end-to-end comparison; model
  weights are large.

Connect with the SSH command RunPod shows for the pod. The login shell starts in the
checkout, `/sgl-workspace/sglang`, with the image's virtualenv active. Check that you
are in the right container:

```bash
echo $VIRTUAL_ENV          # /opt/venv
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
RTX 4090 MiniMax-H3 recipe from `docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx`, with a
`t2va` request at 1344x768 for 5 seconds (the cookbook's VBench shape). H3 rejects a
request without a task, so the script passes `task`, `conditions` and `target` through
`generate --config`; change them with `--request-json`.

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
  --generate-args "--model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --request-json '' \
  --out-dir /workspace/results/e2e-wan
```

The table reports denoise time, median step time, total time, speedup against the
first backend, and aggregate and worst-frame SSIM / PSNR. The pass column uses the
video thresholds from the diffusion performance guide: SSIM 0.92 and PSNR 24 dB.
Each backend's log and perf report are in `<out-dir>/<backend>/`.

The H3 weights for the `fl2va` partition are about 144 GB; keep `HF_HOME` on the
`/workspace` volume.

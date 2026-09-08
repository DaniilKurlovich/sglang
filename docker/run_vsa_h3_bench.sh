#!/usr/bin/env bash
# Entrypoint for the VSA-H3 benchmark image (docker/Dockerfile.vsa_h3_bench).
#
# Profiles the block-sparse Triton attention kernel with Nsight Compute and
# writes the report to ${REPORT_DIR}/${REPORT_NAME}.ncu-rep.
#
# Any extra arguments are passed through to the benchmark script, e.g.:
#   docker run ... vsa-h3-bench --seq-lens 8192 --topk 32 --ragged
set -euo pipefail

REPORT_DIR="${REPORT_DIR:-/workspace/reports}"
REPORT_NAME="${REPORT_NAME:-vsa_h3_full}"

# /workspace is often a volume mounted at runtime (e.g. RunPod), which hides
# anything created in the image at build time. Create the report dir here, or
# ncu fails with a misleading "Unable to write to file ... locked" error.
mkdir -p "${REPORT_DIR}"

exec ncu --set full --force-overwrite \
    -k "regex:_attn_fwd_sparse" --launch-count 8 \
    -o "${REPORT_DIR}/${REPORT_NAME}" \
    python benchmark/vsa_h3/bench_vsa_h3_kernel.py "$@"

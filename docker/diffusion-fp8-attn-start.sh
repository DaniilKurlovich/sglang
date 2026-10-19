#!/usr/bin/env bash
# Container start for docker/diffusion-fp8-attn.Dockerfile on rented GPUs (RunPod style):
# with PUBLIC_KEY set, accept that key and start sshd; then stay alive.
set -euo pipefail

if [[ -n "${PUBLIC_KEY:-}" ]]; then
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
    ssh-keygen -A
    mkdir -p /run/sshd
    /usr/sbin/sshd
    echo "sshd started"
fi

# SSH sessions do not inherit the container environment; pass the pod's variables on.
env | grep -E '^(HF_|HUGGING_FACE_|CUDA_VISIBLE_DEVICES=|RUNPOD_)' \
    | sed 's/^\([^=]*\)=\(.*\)$/export \1="\2"/' > /etc/profile.d/pod-env.sh || true

exec sleep infinity

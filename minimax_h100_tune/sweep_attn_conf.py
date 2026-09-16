# Sweep (num_warps, num_stages, maxnreg) for _attn_fwd_sparse on H100.
# Run on the GPU box:  python minimax_h100_tune/sweep_attn_conf.py
# Prints registers / spills / shared-mem per compiled config + do_bench timing.
import math

import torch
import triton
from triton.testing import do_bench
from triton.tools.tensor_descriptor import TensorDescriptor

from sglang.multimodal_gen.runtime.layers.attention.backends.vsa_h3_kernels import (
    _attn_fwd_sparse,
)

B, H, S, D = 1, 32, 16384, 128  # adjust to the real workload shape
BLOCK = 64
MAX_KV = 64  # top-k per q-tile; adjust to the real workload

device = "cuda"
q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
out = torch.empty_like(q)

n_tiles = S // BLOCK
n_kv = S // BLOCK
q2k_num = torch.full((B, H, n_tiles), MAX_KV, dtype=torch.int32, device=device)
q2k_index = (
    torch.arange(n_kv, dtype=torch.int32, device=device)
    .expand(B, H, n_tiles, n_kv)[:, :, :, :MAX_KV]
    .contiguous()
)
variable_block_sizes = torch.full((n_kv,), BLOCK, dtype=torch.int32, device=device)

block = [1, 1, BLOCK, D]
desc_q, desc_k, desc_v, desc_o = (
    TensorDescriptor.from_tensor(t, block_shape=block) for t in (q, k, v, out)
)
grid = (n_tiles, B * H, 1)


def launch(warps, stages, maxnreg=None):
    kwargs = dict(num_warps=warps, num_stages=stages)
    if maxnreg:
        kwargs["maxnreg"] = maxnreg
    return _attn_fwd_sparse[grid](
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        1.0 / math.sqrt(D),
        q2k_index,
        q2k_num,
        MAX_KV,
        variable_block_sizes,
        H,
        S,
        HEAD_DIM=D,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        **kwargs,
    )


print(f"shape B={B} H={H} S={S} D={D} max_kv={MAX_KV}  (grid {grid})")
print(f"{'warps':>5} {'stages':>6} {'maxnreg':>7} {'regs':>5} {'spills':>6} "
      f"{'smem':>7} {'ms':>9}")
for warps, stages, maxnreg in [
    (4, 2, None),
    (8, 2, None),
    (8, 3, None),
    (8, 2, 128),
    (16, 2, None),
    (16, 3, None),
]:
    try:
        compiled = launch(warps, stages, maxnreg)
        ms = do_bench(lambda: launch(warps, stages, maxnreg), warmup=25, rep=100)
        regs = getattr(compiled, "n_regs", "?")
        spills = getattr(compiled, "n_spills", "?")
        smem = getattr(compiled.metadata, "shared", "?")
        print(f"{warps:>5} {stages:>6} {str(maxnreg):>7} {regs:>5} {spills:>6} "
              f"{smem:>7} {ms:>9.3f}")
    except Exception as e:
        print(f"{warps:>5} {stages:>6} {str(maxnreg):>7}  FAILED: {e}")

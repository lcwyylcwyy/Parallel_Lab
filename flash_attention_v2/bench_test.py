import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import os
import numpy as np
from flash_attn import flash_attn_func
os.environ['TORCH_CUDA_ARCH_LIST'] = 'Ampere'
os.environ['MAX_JOBS'] = '10'
# extra_compile_args['nvcc'] = ['-g', '-G']
# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['fattn.cpp', 'flash.cu'], verbose=True, extra_cuda_cflags=['-O3'])
torch.manual_seed(0)

# set CUDA seeds
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)  # multi-GPU

# make sure the random number's repeatability
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch import Tensor

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 1024
n_head = 12
seq_len = 192
head_embd = 64

softmax_scale = 1.0 / math.sqrt(head_embd)
# batch_size = 1
# n_head = 1
# seq_len = 192
# head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd)
k = torch.randn(batch_size, n_head, seq_len, head_embd)
v = torch.randn(batch_size, n_head, seq_len, head_embd)

print('=== profiling manual attention ===')

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def spda(q, k, v):
    return F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), scale=1.0 / math.sqrt(k.size(-1)))

def fa2_python(Q:Tensor, K:Tensor, V:Tensor) -> Tensor:
    # tr, tc = 2, 2
    NEG_INF = -1e10
    EPSILON = 1e-10
    softmax_scale = 1.0 / math.sqrt(K.size(-1))
    # q_split_size = int(np.ceil(Q.shape[-2] / tr))
    # kv_split_size = int(np.ceil(K.shape[-2] / tc))
    br = 32 # br
    bc = 32 # bc
    tr = int(np.ceil(Q.shape[-2] / br))
    tc = int(np.ceil(K.shape[-2] / bc))
    Q_BLOCKS = list(torch.split(Q, br, dim=2))
    K_BLOCKS = list(torch.split(K, bc, dim=2))
    V_BLOCKS = list(torch.split(V, bc, dim=2))

    Ov2 = torch.zeros_like(Q)
    Ov2_BLOCKS = list(torch.split(Ov2, br, dim=2))

    for i in range(tr):
        Qi = Q_BLOCKS[i]
        Oi_v2 = Ov2_BLOCKS[i]
        mi_v2 = torch.full((Q.shape[0], Q.shape[1], br, 1), -torch.inf)
        li_v2 = torch.zeros((Q.shape[0], Q.shape[1], br, 1))
        for j in range(tc):
            Kj, Vj = K_BLOCKS[j], V_BLOCKS[j]
            S_ij = Qi @ Kj.transpose(-2, -1) * softmax_scale

            mi_new = torch.maximum(mi_v2, torch.max(S_ij, dim=-1, keepdim=True)[0])
            P_ij = torch.exp(S_ij - mi_new)
            li_v2 = torch.exp(mi_v2 - mi_new) * li_v2 + torch.sum(P_ij, dim=-1, keepdim=True) + EPSILON

            P_ij_Vj = P_ij @ Vj
            Oi_v2 = torch.exp(mi_v2 - mi_new) * Oi_v2 + P_ij_Vj
            mi_v2 = mi_new

        Ov2_BLOCKS[i] = Oi_v2 / li_v2
    return torch.cat(Ov2_BLOCKS, dim=2)

fa2_output = fa2_python(q, k, v)
# print("fa2_output output")
# print(fa2_output)
print('=== profiling spda attention ===')
with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    spda_output = spda(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# print("spda output")
# print(spda_output)

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    manual_result = manual_attn(q, k, v)
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
# print('manual_result')
# print(manual_result)
print('manual_result vs spda values sanity check:', torch.allclose(manual_result, spda_output.to('cpu'), rtol=0, atol=1e-02))
print('fa2_output values sanity check:', torch.allclose(fa2_output, spda_output.to('cpu'), rtol=0, atol=1e-02))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    minimal_result = minimal_attn.forward(q.cuda(), k.cuda(), v.cuda())
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
# print('minimal_result')
# print(minimal_result)
print('attn values sanity check:', torch.allclose(spda_output, minimal_result, rtol=0, atol=1e-02))

# flash_output = flash_attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), dropout_p=0.0,
#                                softmax_scale=softmax_scale, causal=False,deterministic=True)
# print(flash_output.to(torch.float32))

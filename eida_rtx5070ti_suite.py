#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy-Invariant Differential Attention (EIDA) - Production Module & Benchmark Suite
=====================================================================================
Resolves:
  1. Entropic Dilution in Long-Context Extrapolation via Log-Length Temperature Scaling.
  2. Common-Mode Context & INT4 KV-Cache Quantization Noise via Differential Cancellation.
  3. O(N^2) VRAM Bottleneck via Distributive Flash-SDPA Decomposition in O(N) Memory.

Hardware Target: NVIDIA GeForce RTX 5070 Ti (bfloat16 Tensor Cores) / CUDA & CPU Fallback.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization per head.
    Stabilizes variance after differential subtraction.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps).to(x.dtype)
        return self.weight.to(x.dtype) * x_normed


def fake_quantize_kv_int4(tensor: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Simulates asymmetric 4-bit group quantization (INT4 KV-Cache) with systematic
    rounding noise (as seen in vLLM / llama.cpp Q4_0 / Q4_K_M).
    """
    orig_shape = tensor.shape
    last_dim = orig_shape[-1]
    if last_dim % group_size != 0:
        group_size = last_dim

    x = tensor.reshape(-1, group_size).float()
    min_val = x.amin(dim=-1, keepdim=True)
    max_val = x.amax(dim=-1, keepdim=True)
    scale = (max_val - min_val).clamp(min=1e-6) / 15.0

    q = torch.round((x - min_val) / scale).clamp(0, 15)
    dequant = q * scale + min_val
    return dequant.reshape(orig_shape).to(tensor.dtype)


class EntropyInvariantDiffAttention(nn.Module):
    """
    Production-ready implementation of EIDA compatible with:
      - Grouped-Query Attention (GQA)
      - FlashAttention-2 / cuDNN SDPA in O(N) memory
      - Entropy-invariant scaling gamma(N) = max(1, ln(N)/ln(N_base))
      - Learnable per-head differential lambda reparameterized for stability.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        n_base: int = 512,
        layer_idx: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.full_head_dim = embed_dim // num_heads
        if self.full_head_dim % 2 != 0:
            raise ValueError("full_head_dim must be even to split into differential branches (Q1, Q2).")
        self.sub_head_dim = self.full_head_dim // 2

        self.n_base = max(n_base, 2)
        self.log_n_base = math.log(self.n_base)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.full_head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.full_head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * max(layer_idx - 1, 0))
        self.lambda_q1 = nn.Parameter(torch.zeros(self.num_heads, self.sub_head_dim).normal_(mean=0.0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.num_heads, self.sub_head_dim).normal_(mean=0.0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.num_heads, self.sub_head_dim).normal_(mean=0.0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.num_heads, self.sub_head_dim).normal_(mean=0.0, std=0.1))

        self.subln = RMSNorm(self.full_head_dim, eps=1e-5)

    def compute_lambda(self) -> torch.Tensor:
        dot1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        dot2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        lam = torch.exp(dot1) - torch.exp(dot2) + self.lambda_init
        return lam.view(1, self.num_heads, 1, 1)

    def compute_entropic_scale(self, seq_len_k: int) -> float:
        gamma_n = max(1.0, math.log(max(seq_len_k, 2)) / self.log_n_base)
        return gamma_n / math.sqrt(self.sub_head_dim)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
        quantize_kv_int4: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, 2, self.sub_head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, 2, self.sub_head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.full_head_dim)

        if quantize_kv_int4:
            k = fake_quantize_kv_int4(k)
            v = fake_quantize_kv_int4(v)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        q1 = q[:, :, :, 0, :].transpose(1, 2).contiguous()
        q2 = q[:, :, :, 1, :].transpose(1, 2).contiguous()
        k1 = k[:, :, :, 0, :].transpose(1, 2).contiguous()
        k2 = k[:, :, :, 1, :].transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        scale = self.compute_entropic_scale(seq_len)

        out1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=is_causal, scale=scale)
        out2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=is_causal, scale=scale)

        lam = self.compute_lambda().to(out1.dtype)
        attn_out = out1 - lam * out2

        attn_out = self.subln(attn_out) * (1.0 - self.lambda_init)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.out_proj(attn_out)


class StandardAttentionBaseline(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.subln = RMSNorm(self.head_dim, eps=1e-5)

    def forward(self, x: torch.Tensor, is_causal: bool = False, quantize_kv_int4: bool = False) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)

        if quantize_kv_int4:
            k = fake_quantize_kv_int4(k)
            v = fake_quantize_kv_int4(v)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = self.subln(out)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.out_proj(out)


def evaluar_recuperacion_aguja_en_pajar(
    context_lengths: List[int],
    sub_head_dim: int = 64,
    n_base: int = 64,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
) -> List[Dict[str, any]]:
    torch.manual_seed(42)
    resultados = []
    full_dim = sub_head_dim * 2

    u_signal = F.normalize(torch.randn(1, 1, sub_head_dim, device=device, dtype=torch.float32), dim=-1)
    u_common = F.normalize(torch.randn(1, 1, sub_head_dim, device=device, dtype=torch.float32), dim=-1)
    u_common = F.normalize(u_common - (u_common * u_signal).sum(dim=-1, keepdim=True) * u_signal, dim=-1)

    for N in context_lengths:
        for usar_int4 in [False, True]:
            etiqueta_kv = "INT4-KV" if usar_int4 else "BF16-KV"

            q1 = (4.2 * u_signal + 3.0 * u_common).to(dtype)
            q2 = (0.2 * u_signal + 3.0 * u_common).to(dtype)

            ruido_1 = 0.35 * torch.randn(1, N, sub_head_dim, device=device, dtype=torch.float32)
            ruido_2 = 0.35 * torch.randn(1, N, sub_head_dim, device=device, dtype=torch.float32)

            k1 = 3.0 * u_common.expand(1, N, sub_head_dim) + ruido_1
            k2 = 3.0 * u_common.expand(1, N, sub_head_dim) + ruido_2

            k1[:, 0:1, :] += 4.2 * u_signal

            k1 = k1.to(dtype)
            k2 = k2.to(dtype)

            if usar_int4:
                k1 = fake_quantize_kv_int4(k1, group_size=32)
                k2 = fake_quantize_kv_int4(k2, group_size=32)

            q_std = torch.cat([q1, q2], dim=-1).float()
            k_std = torch.cat([k1, k2], dim=-1).float()
            logits_std = torch.matmul(q_std, k_std.transpose(-2, -1)) / math.sqrt(full_dim)
            probs_std = F.softmax(logits_std, dim=-1).squeeze()

            prob_aguja_std = float(probs_std[0].item())
            ruido_std = float(probs_std[1:].sum().item())
            entropia_std = float(-(probs_std * torch.log(probs_std.clamp(min=1e-12))).sum().item())

            gamma_n = max(1.0, math.log(N) / math.log(n_base))
            escala_eida = gamma_n / math.sqrt(sub_head_dim)

            w1 = F.softmax(torch.matmul(q1.float(), k1.float().transpose(-2, -1)) * escala_eida, dim=-1).squeeze()
            w2 = F.softmax(torch.matmul(q2.float(), k2.float().transpose(-2, -1)) * escala_eida, dim=-1).squeeze()

            pesos_diff = torch.clamp(w1 - 0.85 * w2, min=0.0)
            pesos_diff = pesos_diff / pesos_diff.sum().clamp(min=1e-12)

            prob_aguja_eida = float(pesos_diff[0].item())
            ruido_eida = float(pesos_diff[1:].sum().item())
            entropia_eida = float(-(pesos_diff * torch.log(pesos_diff.clamp(min=1e-12))).sum().item())

            snr_std = prob_aguja_std / max(ruido_std, 1e-9)
            snr_eida = prob_aguja_eida / max(ruido_eida, 1e-9)
            ganancia_db = 10.0 * math.log10(max(snr_eida / max(snr_std, 1e-12), 1e-12))

            resultados.append({
                "N": N,
                "kv_mode": etiqueta_kv,
                "aguja_std_%": round(prob_aguja_std * 100.0, 2),
                "aguja_eida_%": round(prob_aguja_eida * 100.0, 2),
                "H_std": round(entropia_std, 3),
                "H_eida": round(entropia_eida, 3),
                "snr_gain_db": round(ganancia_db, 2),
            })

    return resultados


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    n_list_calidad = [64, 512, 2048, 8192, 32768]
    resultados_calidad = evaluar_recuperacion_aguja_en_pajar(n_list_calidad, device=device, dtype=dtype)
    for r in resultados_calidad:
        print(r)

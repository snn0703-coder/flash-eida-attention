#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import pytest
import torch
import torch.nn.functional as F

from eida_rtx5070ti_suite import (
    EntropyInvariantDiffAttention,
    evaluar_recuperacion_aguja_en_pajar,
)


def explicit_n2_differential_attention(q1, q2, k1, k2, v, lam, scale, is_causal=False):
    seq_len = q1.size(-2)
    scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
    scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q1.device, dtype=torch.bool),
            diagonal=1,
        )
        scores1 = scores1.masked_fill(causal_mask, float("-inf"))
        scores2 = scores2.masked_fill(causal_mask, float("-inf"))

    attn1 = F.softmax(scores1, dim=-1)
    attn2 = F.softmax(scores2, dim=-1)
    diff_attn_matrix = attn1 - lam * attn2
    return torch.matmul(diff_attn_matrix, v)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("seq_len", [64, 256])
def test_distributive_flash_equivalence(is_causal: bool, seq_len: int):
    torch.manual_seed(101)
    bsz, embed_dim, num_heads = 2, 128, 4
    mod = EntropyInvariantDiffAttention(embed_dim=embed_dim, num_heads=num_heads, n_base=128, layer_idx=2).float()
    x = torch.randn(bsz, seq_len, embed_dim, dtype=torch.float32)

    q = mod.q_proj(x).view(bsz, seq_len, num_heads, 2, mod.sub_head_dim)
    k = mod.k_proj(x).view(bsz, seq_len, num_heads, 2, mod.sub_head_dim)
    v = mod.v_proj(x).view(bsz, seq_len, num_heads, mod.full_head_dim)

    q1 = q[:, :, :, 0, :].transpose(1, 2).contiguous()
    q2 = q[:, :, :, 1, :].transpose(1, 2).contiguous()
    k1 = k[:, :, :, 0, :].transpose(1, 2).contiguous()
    k2 = k[:, :, :, 1, :].transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()

    scale = mod.compute_entropic_scale(seq_len)
    lam = mod.compute_lambda()

    ref_raw = explicit_n2_differential_attention(q1, q2, k1, k2, v_t, lam=lam, scale=scale, is_causal=is_causal)
    ref_normed = mod.subln(ref_raw) * (1.0 - mod.lambda_init)
    ref_out = mod.out_proj(ref_normed.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim))

    flash_out = mod(x, is_causal=is_causal)
    max_diff = (ref_out - flash_out).abs().max().item()
    assert max_diff < 1e-5

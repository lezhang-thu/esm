# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from esm.rotary_embedding import RotaryEmbedding
from esm.lora import LoRALinear


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        use_rotary_embeddings: bool = False,
        use_lora: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim
               ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        if use_lora:
            # https://github.com/pytorch/torchtune/blob/main/recipes/configs/qwen2/0.5B_lora_single_device.yaml
            # lora_rank: 32
            # lora_alpha: 64
            # lora_dropout: 0.0
            lora_rank = 32
            lora_alpha = 64
            lora_dropout = 0.0
            self.k_proj = LoRALinear(
                self.kdim,
                embed_dim,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                use_bias=True,
                quantize_base=False,
            )

            self.v_proj = LoRALinear(
                self.vdim,
                embed_dim,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                use_bias=True,
                quantize_base=False,
            )
            self.q_proj = LoRALinear(
                embed_dim,
                embed_dim,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                use_bias=True,
                quantize_base=False,
            )
        else:
            self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.rot_emb = None
        if use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads,
                                self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads,
                                self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads,
                                self.head_dim).transpose(0, 1)

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.rot_emb:
            q, k = self.rot_emb(q, k)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(
            attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                                             src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                                             src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = F.softmax(attn_weights,
                                       dim=-1,
                                       dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(
            attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len,
                src_len).type_as(attn).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

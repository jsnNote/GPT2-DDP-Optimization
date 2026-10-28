import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
import sys
import os

parent_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_path)
from utils.RoPE import rope


class ScaledDotProductAttention(nn.Module):
    def __init__(self, max_seq_len=1024):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer(
            "look_ahead_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
        )

    def forward(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1))
        if mask:
            scores = scores.masked_fill(self.look_ahead_mask == 0, -1e9)

        attention_weights = self.softmax(scores / math.sqrt(d_k))
        output = torch.matmul(attention_weights, V)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)
        self.W_O = nn.Linear(d_model, d_model, bias=True)

        self.attention = ScaledDotProductAttention(max_seq_len)

    def forward(self, Q, K, V, mask=False):
        batch_size = Q.size(0)
        q_heads = self.W_Q(Q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k_heads = self.W_K(K).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v_heads = self.W_V(V).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        q_heads = rope(q_heads)
        k_heads = rope(k_heads)

        out_heads = self.attention(q_heads, k_heads, v_heads, mask)

        concat_out = (
            out_heads.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        return self.W_O(concat_out)


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.gelu(self.linear1(x))
        x = self.dropout(x)
        pe = self.linear2(x)
        return pe


class TransformerBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, max_seq_len, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mmha = MultiHeadAttention(d_model, h, max_seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PointWiseFeedForward(d_model, d_ff, dropout_rate)

    def forward(self, x, mask=False):
        x_norm = self.norm1(x)
        attention_output = self.mmha(x_norm, x_norm, x_norm, mask)
        x = (
            attention_output + x
        )  # 계산한 값을 새로운 메모리에 저장하고 그것을 x 변수가 참조함
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)

        return ffn_output + x


class GPT2_RoPE(nn.Module):
    def __init__(
        self, vocab_size, d_model, h, d_ff, num_layers, max_seq_len, dropout_rate=0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, h, d_ff, max_seq_len, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.gradient_checkpointing = False
        self.training = True

    def forward(self, x, mask=False):
        x = self.wte(x)
        for layer in self.layers:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:

                x = layer(x, mask)

        x_norm = self.ln_f(x)
        logits = self.lm_head(x_norm)
        return logits

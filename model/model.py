import torch
import torch.nn as nn
import math
import sys
import os

parent_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_path)
from utils.RoPE import rope


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        max_len = Q.shape[-2]
        scores = torch.matmul(Q, K.transpose(-2, -1))
        if mask:
            look_ahead_mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
            scores = scores.masked_fill(look_ahead_mask == 0, -1e9)

        attention_weights = self.softmax(scores / math.sqrt(d_k))
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)
        self.W_O = nn.Linear(d_model, d_model, bias=True)

        self.attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=False):
        batch_size = Q.size(0)
        q_heads = self.W_Q(Q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k_heads = self.W_K(K).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v_heads = self.W_V(V).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        q_heads = rope(q_heads)
        k_heads = rope(k_heads)

        out_heads, _ = self.attention(q_heads, k_heads, v_heads, mask)

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
    def __init__(self, d_model, h, d_ff, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mmha = MultiHeadAttention(d_model, h)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PointWiseFeedForward(d_model, d_ff, dropout_rate)

    def forward(self, x, mask=False):
        x_norm = self.norm1(x)
        attention_output = self.mmha(x_norm, x_norm, x_norm, mask)
        x = attention_output + x
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        z = ffn_output + x

        return z


class GPT2_RoPE(nn.Module):
    def __init__(self, vocab_size, d_model, h, d_ff, num_layers, dropout_rate=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, h, d_ff, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, x, mask=False):
        x = self.wte(x)
        for layer in self.layers:
            x = layer(x, mask)
        x_norm = self.ln_f(x)
        logits = self.lm_head(x_norm)
        return logits


dummy_trg = torch.randint(0, 50257, size=(2, 10))

model = GPT2_RoPE(vocab_size=50257, d_model=1024, h=16, d_ff=(1024 * 4), num_layers=24)
# print(torch.numeal(model.parameters()))
with torch.no_grad():
    output = model(dummy_trg, mask=True)

print(output.size())

# print(model)
print(f"parameters: {sum([p.numel() for p in model.parameters()])/1e6:.2f}M")
# print([(name, p.numel()) for name, p in model.named_parameters()])


"""
from torchinfo import summary

summary(model, input_size=(2, 10, 1024))
"""

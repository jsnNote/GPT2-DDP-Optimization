import torch


def rope(x):
  seq_len = x.shape[-2]
  d_k = x.shape[-1]
  freqs = 10000 ** (-torch.arange(0, d_k // 2, device=x.device) * 2 / d_k)
  angles = torch.arange(seq_len, device=x.device)[:, None] * freqs[None, :] 
  sin, cos = angles.sin(), angles.cos()

  x1, x2 = x[..., ::2], x[..., 1::2]
  output = torch.empty_like(x, dtype=torch.float32, device=x.device)

  output[..., ::2] = x1 * cos - x2 * sin
  output[..., 1::2] = x1 * sin + x2 * cos

  return output
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    def __init__(self, x, seq_len=1024):
        self.x = (
            x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.int64)
        )  # 임베딩에서는 int16 지원 안 함
        self.num_samples = (len(self.x) - 1) // seq_len
        self.src = self.x[: self.num_samples * seq_len].reshape(
            self.num_samples, seq_len
        )  # (샘플수, seq_len)

        self.trg = self.x[1 : self.num_samples * seq_len + 1].reshape(
            self.num_samples, seq_len
        )

    def __getitem__(self, idx):
        x = self.src[idx]
        y = self.trg[idx]
        return x, y

    def __len__(self):
        return self.num_samples


# class TokenEmbedding(torch.nn.Module):
#     def __init__(self, vocab_size, d_model):
#         super().__init__()
#         self.emb = torch.nn.Embedding(vocab_size, d_model)

#     def forward(self, id):
#         return self.emb(id)


def get_dataloader(batch_size, seq_len=1024, split="train"):
    assert split in ["train", "val"], "split 인자는 train, val 값만을 가질 수 있음."
    path = os.path.join(os.path.dirname(__file__), f"{split}.bin")
    load_tokens = np.memmap(path, dtype=np.uint16, mode="r").astype(
        np.int64
    )  # 저장시 uint16으로 용량줄이고 로드할 때는 임베딩을 위해 int64로 변환
    dataset = Dataset(load_tokens, seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
    )

    return dataloader

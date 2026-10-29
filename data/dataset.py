import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class Dataset(Dataset):
    def __init__(self, x, seq_len=1024):
        self.num_samples = (len(x) - 1) // seq_len
        self.src = x[: self.num_samples * seq_len].reshape(
            self.num_samples, seq_len
        )  # (샘플수, seq_len)

        self.trg = x[1 : self.num_samples * seq_len + 1].reshape(
            self.num_samples, seq_len
        )

    def __getitem__(self, idx):
        # 임베딩에서는 int16 지원 안 함 ->  int64로 변환
        # astype은 새로운 메모리 할당이 발생하기 때문에
        x = self.src[idx].astype(np.int64)
        y = self.trg[idx].astype(np.int64)
        return x, y

    def __len__(self):
        return self.num_samples


def get_dataloader(
    batch_size, world_size, rank, seq_len=1024, split="train", shuffle=True, seed=42
):
    assert split in ["train", "val"], "split 인자는 train, val 값만을 가질 수 있음."
    path = os.path.join(os.path.dirname(__file__), f"{split}.bin")
    load_tokens = np.memmap(path, dtype=np.uint16, mode="r")
    dataset = Dataset(load_tokens, seq_len)
    sampler = DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=True
    )

    return sampler, dataloader

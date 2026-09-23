from datasets import load_dataset, Dataset
import numpy as np
import os
import sys
from tqdm import tqdm

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
from utils.tokenizer_utils import encode_batch, process_batch

if __name__ == "__main__":
    num_proc = 8
    num_threads = 2

    dataset = load_dataset("Skylion007/openwebtext", split="train", num_proc=num_proc)
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=42, shuffle=False)
    split_dataset["val"] = split_dataset.pop("test")

    # 최적화
    # nanoGPT의 경우에는 문서 단위로 처리 후 shard를 통해서 배치 단위로 읽어서 처리
    # 문서 단위로 저장하면 row 수가 증가해서 쓰기 작업이 많아짐
    # 배치 단위로 저장하면 shard 없이도 배치 단위로 읽어서 처리 가능

    tokenized = split_dataset.map(
        process_batch,
        remove_columns=["text"],
        num_proc=num_proc,
        batched=True,
        fn_kwargs={"encode": encode_batch, "num_threads": num_threads},
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(
            dset["len"], dtype=np.uint64
        )  # 토큰 수가 90억개 이므로 타입은 uint64로 해야 함
        path = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(arr_len,))

        idx = 0
        for batch in tqdm(dset, desc=f"writing {split}.bin"):
            ids = batch["ids"]
            n = batch["len"]
            arr[idx : idx + n] = ids
            idx += n
        arr.flush()  # 데이터 반영

        print(f"{split}: {arr_len} tokens saved")
        assert idx == arr_len, f"토큰 개수 불일치: idx={idx}, arr_len={arr_len}"

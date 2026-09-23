from datasets import load_dataset, Dataset
import tiktoken
import os
import sys
import time

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
from utils.tokenizer_utils import encode_batch, process_batch, count_tokens, BASE
from data.load_dumy import load_dummy

base = BASE
pattern = base._pat_str
vocab = base._mergeable_ranks
special_tokens = base._special_tokens

if __name__ == "__main__":
    test_mode = "--dummy" in sys.argv
    if test_mode:
        sample_docs = load_dummy()
        data_dict = {"text": sample_docs}
        dataset = Dataset.from_dict(data_dict)
    else:
        dataset = load_dataset(
            "Skylion007/openwebtext", split="train", num_proc=(os.cpu_count() // 2)
        ).select(range(200_000))

    def bench(dataset, name, encode_func, num_proc, num_threads):
        t = time.perf_counter()
        tokenized = dataset.map(
            process_batch,
            remove_columns=["text"],
            num_proc=num_proc,
            batched=True,
            fn_kwargs={"encode": encode_func, "num_threads": num_threads},
            load_from_cache_file=False,
        )
        n = count_tokens(dataset)
        duration = time.perf_counter() - t
        print(
            f"[{name}(num_proc: {num_proc}, num_threads: {num_threads})]: {len(dataset)/duration:.2f} docs/s, {n/duration/1e6:.2f}M tokens/s"
        )
        return tokenized["ids"]

    # 동작 테스트
    tokenized = dataset.map(
        process_batch,
        remove_columns=["text"],
        num_proc=(1 if test_mode else os.cpu_count() // 2),
        batched=True,
        fn_kwargs={"encode": encode_batch, "num_threads": 2},
        load_from_cache_file=False,
    )
    print(count_tokens(tokenized))

    # # 처리량 테스트
    """
    for proc, thr in [(8, 2), (8, 8), (16, 8)]:
        bench(dataset, "tiktoken", base.encode_ordinary_batch, proc, thr)

        bench(dataset, "tokendagger", encode_batch, proc, thr)
    bench(dataset, "tokendagger", encode_batch, 8, 2)
    """
    # 출력 테스트(결과: True)
    """
    text = dataset.select(range(2000))
    a = bench(text, "tiktoken", base.encode_ordinary_batch, 8, 2)

    b = bench(text, "tokendagger", encode_batch, 8, 2)
    print("동일한 출력: ", {a == b})
    """
    # 병목 테스트
    """
    sets = {
        "seq": dataset.train_test_split(test_size=0.0005, seed=42, shuffle=False)[
            "train"
        ].select(range(50_000)),
        "shuf": dataset.train_test_split(test_size=0.0005, seed=42, shuffle=True)[
            "train"
        ].select(range(50_000)),
    }
    for key in sets.keys():
        for proc, thr in [(8, 2), (16, 8)]:
            bench(sets[key], "tokendagger", encode_batch, proc, thr)
    """

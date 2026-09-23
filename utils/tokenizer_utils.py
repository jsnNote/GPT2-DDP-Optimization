import tiktoken
import tokendagger
from datasets import Dataset, DatasetDict

BASE = tiktoken.get_encoding("gpt2")
PARTTERN = BASE._pat_str
VOCAB = BASE._mergeable_ranks
SPECIAL_TOKENS = BASE._special_tokens
EOT_TOKEN = BASE.eot_token
_enc = tokendagger.Encoding(
    name="gpt2", pat_str=PARTTERN, mergeable_ranks=VOCAB, special_tokens=SPECIAL_TOKENS
)


def process_batch(batch, encode, num_threads):
    encoded_ids = encode(batch["text"], num_threads=num_threads)
    all_ids = [ids + [EOT_TOKEN] for ids in encoded_ids]
    merged_ids = [token for ids in all_ids for token in ids]

    return {
        "ids": [merged_ids],
        "len": [len(merged_ids)],
    }


def encode_batch(texts, **kw):
    global _enc
    if _enc is None:
        _enc = tokendagger.Encoding(
            name="gpt2",
            pat_str=PARTTERN,
            megeable_ranks=VOCAB,
            special_tokens=SPECIAL_TOKENS,
        )

    return _enc.encode_batch(texts, **kw)


def count_tokens(dataset, tokenized=False):
    if tokenized:
        dataset = dataset.map(
            process_batch,
            remove_columns=["text"],
            num_proc=8,
            batched=True,
            fn_kwargs={"encode": encode_batch, "num_threads": 2},
            load_from_cache_file=False,
        )
    assert isinstance(dataset, DatasetDict) or isinstance(
        dataset, Dataset
    ), "dataset의 타입 확인 필요(조건: Dataset, DatasetDict)"
    n = 0
    # dataset의 타입은 Dataset, DatasetDict 밖에 없음.
    if isinstance(dataset, DatasetDict):

        for _, data in dataset.items():
            n += sum(data["len"])

    elif isinstance(dataset, Dataset):
        n += sum(dataset["len"])

    return n


if __name__ == "__main__":
    data_dict = {"text": ["hello"]}
    dataset = Dataset.from_dict(data_dict)
    print(count_tokens(dataset, tokenized=True))
    print(encode_batch(["hello"]))

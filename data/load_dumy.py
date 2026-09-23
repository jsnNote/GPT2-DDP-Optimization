import os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dummy.text")


def load_dummy():
    sample_docs = []

    # claude를 이용해서 생성한 더미 데이터
    with open(_PATH, "r", encoding="utf-8") as f:
        for line in f:
            sample_docs.append(line.strip("\n"))
    return sample_docs

import re

import datasets
from datasets import load_dataset


def get_dataset() -> datasets.DatasetDict:
    ds = load_dataset("amarfurt/art-nlp-hs26")

    cols = ["observation_1", "observation_2", "hypothesis_1", "hypothesis_2"]

    def has_empty_field(x):
        return any(not re.search(r"\w", x[c]) for c in cols)

    # Train set contains empty hypotheses
    ds["train"] = ds["train"].filter(lambda x: not has_empty_field(x))

    return ds

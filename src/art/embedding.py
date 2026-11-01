import logging

import datasets
import gensim
import numpy as np
from datasets import List, Value
from datasets.fingerprint import Hasher

from art.preprocess import to_tokens

log = logging.getLogger(__name__)


def add_embed(
    ds: datasets.DatasetDict,
    wv: gensim.models.KeyedVectors,
    input_mode: str = "split_h",
    mean_cat: bool = True,
    vocab: dict[str, int] | None = None,
) -> datasets.DatasetDict:
    if not mean_cat and vocab is None:
        raise ValueError("vocab is required when mean_cat=False")

    def to_mean_embed(item):
        embed = {}
        for part in ["observation_1", "observation_2", "hypothesis_1", "hypothesis_2"]:
            token, oov, oov_rate = to_tokens(item[part], wv)

            if oov_rate and oov_rate >= 0.5:
                log.debug(
                    f"High oov_rate for '{item[part]}': oov: {oov}, oov_rate: {oov_rate}"
                )

            embed[part] = wv.get_mean_vector(token, ignore_missing=True)

        if input_mode == "hyp_only":
            return {
                "embed": [
                    embed["hypothesis_1"],
                    embed["hypothesis_2"],
                ]
            }
        return {
            "embed": [
                np.concatenate(
                    [
                        embed["observation_1"],
                        embed["hypothesis_1"],
                        embed["observation_2"],
                    ]
                ),
                np.concatenate(
                    [
                        embed["observation_1"],
                        embed["hypothesis_2"],
                        embed["observation_2"],
                    ]
                ),
            ]
        }

    def to_embed_idx(item):
        embed = {}
        for part in ["observation_1", "observation_2", "hypothesis_1", "hypothesis_2"]:
            tokens, oov, oov_rate = to_tokens(item[part], wv)

            if oov_rate and oov_rate >= 0.5:
                log.debug(
                    f"High oov_rate for '{item[part]}': oov: {oov}, oov_rate: {oov_rate}"
                )

            embed[part] = [vocab[t] for t in tokens]

        if input_mode == "hyp_only":
            return {
                "embed": [
                    embed["hypothesis_1"],
                    embed["hypothesis_2"],
                ]
            }
        return {
            "embed": [
                embed["observation_1"] + embed["hypothesis_1"] + embed["observation_2"],
                embed["observation_1"] + embed["hypothesis_2"] + embed["observation_2"],
            ]
        }

    if mean_cat:
        for split in ds:
            ds[split] = ds[split].map(
                to_mean_embed,
                new_fingerprint=f"{ds[split]._fingerprint}-w2v-mean-{input_mode}",
            )
    else:
        vocab_hash = Hasher.hash(vocab)
        for split in ds:
            features = ds[split].features.copy()
            features["embed"] = List(List(Value("int64")))
            ds[split] = ds[split].map(
                to_embed_idx,
                features=features,
                new_fingerprint=Hasher.hash(
                    (
                        ds[split]._fingerprint,
                        "w2v-idx",
                        input_mode,
                        vocab_hash,
                        features,
                    )
                ),
            )

    return ds

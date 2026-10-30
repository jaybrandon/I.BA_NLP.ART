import logging

import datasets
import gensim
import numpy as np

from art.preprocess import to_tokens

log = logging.getLogger(__name__)


def add_embed(
    ds: datasets.DatasetDict,
    wv: gensim.models.KeyedVectors,
    mean_cat: bool = True,
    split_h: bool = True,
) -> datasets.DatasetDict:
    def _to_mean_embed(item):
        embed = {}
        for part in ["observation_1", "observation_2", "hypothesis_1", "hypothesis_2"]:
            token, oov, oov_rate = to_tokens(item[part], wv)

            if oov_rate and oov_rate >= 0.5:
                log.warning(
                    f"High oov_rate for '{item[part]}': oov: {oov}, oov_rate: {oov_rate}"
                )

            embed[part] = wv.get_mean_vector(token, ignore_missing=False)
        if split_h:
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
        return {
            "embed": np.concatenate(
                [
                    embed["observation_1"],
                    embed["hypothesis_1"],
                    embed["hypothesis_2"],
                    embed["observation_2"],
                ]
            )
        }

    if mean_cat:
        for split in ds:
            ds[split] = ds[split].map(
                _to_mean_embed,
                new_fingerprint=f"{ds[split]._fingerprint}-w2v-mean-{split_h}",
                load_from_cache_file=False,
            )

    return ds

import logging
import re

import nltk
import torch
from datasets import DatasetDict

nltk.download("punkt_tab")
log = logging.getLogger(__name__)

COLS = ["observation_1", "observation_2", "hypothesis_1", "hypothesis_2"]


def build_vocab(ds: DatasetDict, wv) -> tuple[dict[str, int], torch.Tensor]:
    """Build a token -> index vocab over all splits and text columns and the matching w2v matrix."""
    vocab = {"<pad>": 0, "<unk>": 1}
    for split in ds.values():
        for col in COLS:
            for text in split[col]:
                tokens, _, _ = to_tokens(text, wv)
                for t in tokens:
                    vocab.setdefault(t, len(vocab))

    # <pad> and <unk> stay zero vectors
    embeddings = torch.zeros(len(vocab), wv.vector_size)
    for t, i in vocab.items():
        if t in wv:
            embeddings[i] = torch.from_numpy(wv[t].copy()).float()

    return vocab, embeddings


def to_tokens(text: str, wv) -> tuple[list[str], int, float | None]:
    """Preprocess and convert text to tokens for w2v.

    returns: ``out``, ``oov``, ``oov_rate``
    """

    tokens = [t for t in nltk.word_tokenize(text) if not re.fullmatch(r"\W+", t)]

    out = []
    oov = 0
    for t in tokens:
        lt = lookup_token(t, wv)
        if lt:
            out.append(lt)
        else:
            oov += 1
            out.append("<unk>")
            log.debug(f"token: {t}, not in wv")

    if len(tokens) == 0:
        log.warning(f"No valid tokens in {text}")
        return out, oov, None

    return out, oov, oov / len(tokens)


def lookup_token(t: str, wv):
    if t in wv:
        return t
    t2 = re.sub(r"\d", "#", t)

    if t2 in wv:
        return t2

    t3 = t.lower()
    return t3 if t3 in wv else None

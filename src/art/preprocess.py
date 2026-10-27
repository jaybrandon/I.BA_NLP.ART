import logging
import re

import nltk

nltk.download("punkt_tab")
log = logging.getLogger(__name__)


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

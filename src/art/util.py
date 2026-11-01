import os
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

SEED = 42


def set_seeds(seed: int = SEED):
    """
    Set seed for various random generators.

    RandomGenerators affected: ``HASHSEED``, ``random``, ``torch``, ``torch.cuda``,
    ``numpy.random``
    :param seed: Integer seed to set random generators to
    :raises ValueError: If the seed is not an integer
    """
    if not isinstance(seed, int):
        raise ValueError(f"Expect seed to be an integer, but got {type(seed)}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_env(key: str, default=None) -> str:
    env = os.environ.get(key)
    if not env:
        if default:
            return default
        raise ValueError(f"{key} environment variable is not set")
    return env


def get_device() -> str:
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"


def collate_fn(batch):  # batch: list of dicts, each with "ids": [seq_h1, seq_h2]
    seqs = [torch.tensor(s) for item in batch for s in item["ids"]]
    lengths = torch.tensor([len(s) for s in seqs])  # (2B,)
    ids = pad_sequence(seqs, batch_first=True, padding_value=0)  # (2B, S)
    B = len(batch)
    return (
        (ids.reshape(B, 2, -1), lengths.reshape(B, 2)),
        torch.tensor([item["label"] for item in batch]),
    )

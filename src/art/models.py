import gensim.downloader as api
import torch
from gensim.models.keyedvectors import KeyedVectors
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from art.config import W2V_MODEL, W2V_MODEL_PATH


def load_w2v() -> KeyedVectors:
    if W2V_MODEL_PATH.exists():
        wv = KeyedVectors.load(str(W2V_MODEL_PATH), mmap="r")
    else:
        wv = api.load(W2V_MODEL)
        W2V_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        wv.save(str(W2V_MODEL_PATH))
    return wv


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout=0.0):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        s1 = self.dropout(self.relu(self.fc1(x)))
        output = self.fc2(s1)
        return output.squeeze(-1)


class RNNClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embeddings: torch.Tensor,
        rnn_type: str,
        mlp_hidden_dim: int,
        bidirectional: bool = False,
        rnn_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings, freeze=True, padding_idx=0
        )

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embeddings.shape[1],
                hidden_dim,
                rnn_layers,
                dropout=dropout if rnn_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                embeddings.shape[1],
                hidden_dim,
                rnn_layers,
                dropout=dropout if rnn_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True,
            )

        self.mlp = MLPClassifier(
            2 * hidden_dim if bidirectional else hidden_dim,
            1,
            mlp_hidden_dim,
            dropout=dropout,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.rnn.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, inputs):
        x, lengths = inputs  # x: (B, 2, S) token ids
        B, C, S = x.shape

        x = x.reshape(B * C, S)  # (2B, S)
        lengths = lengths.reshape(B * C).clamp(min=1).cpu()

        e = self.embeddings(x)  # (2B, S, 300)
        p = pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(p)
        if isinstance(h_n, tuple):  # LSTM returns (h_n, c_n)
            h_n = h_n[0]

        if self.rnn.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (2B, 2H)
        else:
            h = h_n[-1]  # (2B, H)

        return self.mlp(h).reshape(B, C)  # (B, 2) logits

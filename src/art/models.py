import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors
from torch import nn

from art.config import W2V_MODEL, W2V_MODEL_PATH


def load_w2v() -> KeyedVectors:
    if W2V_MODEL_PATH.exists():
        wv = KeyedVectors.load(str(W2V_MODEL_PATH), mmap="r")
    else:
        wv = api.load(W2V_MODEL)
        W2V_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        wv.save(str(W2V_MODEL_PATH))
    return wv


class EmbeddingClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        s1 = self.relu(self.fc1(x))
        output = self.fc2(s1)
        return output.squeeze(-1)

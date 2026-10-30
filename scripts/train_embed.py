import gensim.downloader as api
import torch
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader

from art.config import CHECKPOINTS_DIR
from art.data import get_dataset
from art.embedding import add_embed
from art.models import EmbeddingClassifier
from art.training import fit

wv = api.load("word2vec-google-news-300")

ds = get_dataset()

ds = add_embed(ds, wv).with_format("torch", ["embed", "label"])

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = EmbeddingClassifier(3 * 300, 1, 6 * 900)
model.to(device)

train_loader = DataLoader(
    ds["train"],
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    ds["validation"],
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

optimizer = AdamW(model.parameters())

best_metrics = fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    cross_entropy,
    10,
    3,
    CHECKPOINTS_DIR / "best.pt",
    lambda x: print(x),
    device,
)

print(best_metrics)

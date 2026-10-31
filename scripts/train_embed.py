import argparse
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb
from art.config import CHECKPOINTS_DIR, ROOT
from art.data import get_dataset
from art.embedding import add_embed
from art.models import EmbeddingClassifier, load_w2v
from art.training import fit
from art.util import get_env, set_seeds

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()

with args.config.open() as f:
    cfg = yaml.safe_load(f)


load_dotenv()
set_seeds()

wv = load_w2v()

print("Loading dataset...")

ds = get_dataset()

ds = add_embed(ds, wv, split_h=cfg["input"] == "split_h").with_format(
    "torch", ["embed", "label"]
)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

run_name = f"{cfg['model']}-w2v-{cfg['input']}-h{cfg['hidden_dim']}"

with wandb.init(
    entity=get_env("WANDB_ENTITY"),
    project=get_env("WANDB_PROJECT"),
    name=run_name,
    group=cfg["model"],
    tags=[cfg["model"], cfg["input"]],
    config=cfg,
) as run:
    config = run.config

    model = EmbeddingClassifier(3 * 300, 1, config.hidden_dim)
    model.to(device)

    train_loader = DataLoader(
        ds["train"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        ds["validation"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint = CHECKPOINTS_DIR / f"{run_name}-{run.id}.pt"
    best_metrics = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        cross_entropy,
        config.epochs,
        config.patience,
        checkpoint,
        run.log,
        device,
    )

    run.summary.update({f"best_{k}": v for k, v in best_metrics.items()})
    run.summary.update({"checkpoint": checkpoint.relative_to(ROOT)})

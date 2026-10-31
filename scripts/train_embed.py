import argparse
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb
from art.config import CHECKPOINTS_DIR, ROOT, W2V_MODEL
from art.data import get_dataset
from art.embedding import add_embed
from art.models import EmbeddingClassifier, load_w2v
from art.training import fit
from art.util import get_env, set_seeds

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--sweep", type=Path)
parser.add_argument("--count", type=int, default=20)
args = parser.parse_args()

with args.config.open() as f:
    cfg = yaml.safe_load(f)
cfg["embedding"] = W2V_MODEL

load_dotenv()

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


def main():
    with wandb.init(
        entity=get_env("WANDB_ENTITY"),
        project=get_env("WANDB_PROJECT"),
        config=cfg,
    ) as run:
        config = run.config

        run.define_metric("val_accuracy", summary="max")
        run.define_metric("*", step_metric="epoch")

        run.name = (
            f"{config.model}-w2v-{config.input}-h{config.hidden_dim}"
            f"-lr{config.learning_rate:g}-wd{config.weight_decay:g}"
            f"-do{config.dropout:g}"
        )
        run.tags = (config.model, config.input)

        if config.input != "split_h":
            raise ValueError(f"Unsupported input format: {config.input}")

        set_seeds(config.seed)

        wv = load_w2v()

        print("Loading dataset...")

        ds = get_dataset()
        ds = add_embed(ds, wv).with_format("torch", ["embed", "label"])

        in_dim = ds["train"][0]["embed"].shape[-1]

        model = EmbeddingClassifier(in_dim, 1, config.hidden_dim, config.dropout)
        model.to(device)
        run.summary["n_params"] = sum(p.numel() for p in model.parameters())

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
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        checkpoint = CHECKPOINTS_DIR / f"{run.name}-{run.id}.pt"
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
        run.summary.update({"checkpoint": str(checkpoint.relative_to(ROOT))})


if args.sweep:
    sweep_cfg = yaml.safe_load(args.sweep.open())
    sweep_id = wandb.sweep(
        sweep_cfg, entity=get_env("WANDB_ENTITY"), project=get_env("WANDB_PROJECT")
    )
    wandb.agent(sweep_id, function=main, count=args.count)
else:
    main()

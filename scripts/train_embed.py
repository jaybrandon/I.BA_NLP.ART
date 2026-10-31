import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb
from art.config import CHECKPOINTS_DIR, ROOT, W2V_MODEL
from art.data import get_dataset, to_tensor_ds
from art.embedding import add_embed
from art.models import EmbeddingClassifier, load_w2v
from art.training import fit
from art.util import get_device, get_env, set_seeds

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--sweep", type=Path)
parser.add_argument("--count", type=int, default=20)
args = parser.parse_args()

with args.config.open() as f:
    cfg = yaml.safe_load(f)
cfg["embedding"] = W2V_MODEL

load_dotenv()

device = get_device()
print(f"Using {device} device")

wv = load_w2v()

print("Loading dataset...")

ds = get_dataset()
ds = add_embed(ds, wv, cfg["input"]).with_format("torch", ["embed", "label"])

train_ds = to_tensor_ds(ds["train"])
val_ds = to_tensor_ds(ds["validation"])

in_dim = train_ds.tensors[0].shape[-1]


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
            f"-do{config.dropout:g}-bs{config.batch_size}"
        )
        run.tags = (config.model, config.input)

        set_seeds(config.seed)

        model = EmbeddingClassifier(in_dim, 1, config.hidden_dim, config.dropout)
        model.to(device)
        run.summary["n_params"] = sum(p.numel() for p in model.parameters())

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

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

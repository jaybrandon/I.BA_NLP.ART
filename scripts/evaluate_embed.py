import argparse
from pathlib import Path

import torch
from datasets import DatasetDict
from dotenv import load_dotenv
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

import wandb
from art.config import CHECKPOINTS_DIR
from art.data import get_dataset, to_tensor_ds
from art.embedding import add_embed
from art.models import MLPClassifier, load_w2v
from art.training import evaluate
from art.util import get_device, get_env

parser = argparse.ArgumentParser()
target = parser.add_mutually_exclusive_group(required=True)
target.add_argument("--run-id")
target.add_argument("--checkpoint", type=Path)
args = parser.parse_args()

# checkpoints are saved as {run.name}-{run.id}.pt
if args.checkpoint:
    checkpoint = args.checkpoint
    run_id = checkpoint.stem.rsplit("-", 1)[-1]
else:
    run_id = args.run_id
    matches = list(CHECKPOINTS_DIR.glob(f"*-{run_id}.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one checkpoint for run {run_id}, found {len(matches)}"
        )
    checkpoint = matches[0]

load_dotenv()

# update the finished run through the API instead of resuming it, so its state is kept
run = wandb.Api().run(f"{get_env('WANDB_ENTITY')}/{get_env('WANDB_PROJECT')}/{run_id}")
if run.state == "running":
    raise RuntimeError(f"Run {run_id} is still running")
config = run.config

device = get_device()
print(f"Using {device} device")
print(f"Evaluating {checkpoint.name}")

wv = load_w2v()

print("Loading dataset...")

ds = get_dataset()
ds = add_embed(DatasetDict(test=ds["test"]), wv, config["input"]).with_format(
    "torch", ["embed", "label"]
)

test_ds = to_tensor_ds(ds["test"])

in_dim = test_ds.tensors[0].shape[-1]

model = MLPClassifier(in_dim, 1, config["hidden_dim"], config.get("dropout", 0.0))
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.to(device)

test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

test_loss, test_accuracy = evaluate(test_loader, model, cross_entropy, device)
print(f"test_loss: {test_loss:.4f}, test_accuracy: {test_accuracy:.2f}")

run.summary.update({"test_loss": test_loss, "test_accuracy": test_accuracy})
print(f"Logged test metrics to {run.url}")

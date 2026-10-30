from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    epochs: int,
    early_stopping: int,
    checkpoint_path: Path,
    log_fn,
    device,
) -> dict:

    best_metrics = {"val_accuracy": -np.inf, "val_loss": np.inf}
    patience = early_stopping

    for i in trange(epochs, desc="epochs"):
        train_loss, train_accuracy = train(
            train_loader, model, loss_fn, optimizer, device
        )

        val_loss, val_accuracy = evaluate(val_loader, model, loss_fn, device)

        metrics = {
            "epoch": i + 1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        }

        log_fn(metrics)

        if val_accuracy > best_metrics["val_accuracy"] or (
            val_accuracy == best_metrics["val_accuracy"]
            and val_loss < best_metrics["val_loss"]
        ):
            best_metrics = metrics
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            patience = early_stopping
        else:
            patience -= 1

        if patience <= 0:
            break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return best_metrics


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    optimizer.zero_grad()

    model.train()

    train_loss, correct = 0, 0

    prog = tqdm(dataloader, "train", num_batches, leave=False)

    for batch in prog:
        X, y = batch["embed"].to(device), batch["label"].to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        correct += (pred.argmax(-1) == y).type(torch.float).sum().item()

        prog.set_postfix(loss=f"{loss.item():.4f}")

    train_loss /= num_batches
    accuracy = 100 * correct / size

    return train_loss, accuracy


def evaluate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    val_loss, correct = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, "eval", num_batches, leave=False):
            X, y = batch["embed"].to(device), batch["label"].to(device)

            pred = model(X)
            val_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(-1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    accuracy = 100 * correct / size
    return val_loss, accuracy

# I.BA_NLP.ART

Abductive reasoning (ART / αNLI) with word embeddings and RNNs. Given two
observations, the model picks the more plausible of two hypotheses.

- Dataset: [`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26) (Hugging Face)
- Embeddings: `word2vec-google-news-300` (gensim)
- Models: MLP on mean word2vec embeddings, and uni/bidirectional LSTM/GRU classifiers
- Tracking: Weights & Biases

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync              # install dependencies (incl. dev group) into .venv
uv run pre-commit install   # optional: ruff, yamllint, nb-clean hooks
```

### Environment

Create a `.env` file in the project root:

```sh
HF_TOKEN=...         # Hugging Face token for downloading the dataset
WANDB_ENTITY=...     # W&B user or team
WANDB_PROJECT=...    # W&B project name
```

Log in to W&B once with `uv run wandb login` (or set `WANDB_API_KEY`).

On first run, the word2vec model (~1.6 GB) is downloaded and cached in
`models/w2v.kv`. NLTK `punkt_tab` is downloaded automatically.

## Usage

Run all scripts from the project root. Configs live in [configs/](configs/).

### Train

```sh
# MLP on averaged embeddings
uv run scripts/train_embed.py --config configs/mlp.yaml

# RNN (bilstm, bigru, unilstm, unigru)
uv run scripts/train_rnn.py --config configs/bilstm.yaml
```

The best checkpoint (by validation accuracy) is saved to
`models/checkpoints/{run_name}-{run_id}.pt`.

### Hyperparameter sweep

The sweep file defines the search space; the `--config` file supplies the
remaining fixed values.

```sh
uv run scripts/train_embed.py --config configs/mlp.yaml --sweep configs/sweep_mlp.yaml --count 20
uv run scripts/train_rnn.py --config configs/bigru.yaml --sweep configs/sweep_rnn.yaml --count 20
```

### Evaluate

Evaluates a finished run on the test split and writes `test_loss` /
`test_accuracy` to the run's W&B summary. Pass either the W&B run ID or the
checkpoint path.

```sh
uv run scripts/evaluate_embed.py --run-id <run_id>
uv run scripts/evaluate_rnn.py --checkpoint models/checkpoints/<name>-<run_id>.pt
```

## Project layout

```
configs/     training and sweep configs (YAML)
docs/        slides and charts
notebooks/   data exploration, OOV analysis, experiments
scripts/     train / evaluate entry points
src/art/     package: data loading, preprocessing, embeddings, models, training loop
```

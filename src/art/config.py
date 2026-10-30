from pathlib import Path

ROOT = Path(__file__).parents[2]
MODELS_DIR = ROOT / "models"

W2V_MODEL_PATH = MODELS_DIR / "w2v.kv"
W2V_MODEL = "word2vec-google-news-300"

CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

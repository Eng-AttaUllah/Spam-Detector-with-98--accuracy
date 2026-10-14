"""Data loading and preparation for the SMS spam dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Default location of the dataset, relative to the project root.
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "spam or ham dataset.csv"
)

# Map the raw string labels to integers: ham = 0, spam = 1.
LABEL_MAP = {"ham": 0, "spam": 1}
# Reverse mapping, used when turning predictions back into words.
LABEL_NAMES = {0: "ham", 1: "spam"}


def load_dataset(path=DEFAULT_DATASET_PATH):
    """Load the spam/ham CSV into a tidy DataFrame with `label` and `message`.

    The raw file uses `v1` for the label and `v2` for the message text, and
    carries three unnamed empty trailing columns that we drop.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path, encoding="latin-1")

    # Keep only the two meaningful columns and rename them.
    data = data[["v1", "v2"]].copy()
    data.columns = ["label", "message"]

    # Normalise labels to lower case and drop anything unparseable.
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[data["label"].isin(LABEL_MAP)]

    # Drop duplicate and empty messages so they don't skew evaluation.
    data = data.dropna(subset=["message"])
    data["message"] = data["message"].astype(str).str.strip()
    data = data[data["message"] != ""]
    data = data.drop_duplicates(subset=["message"]).reset_index(drop=True)

    return data


def split_features_labels(data):
    """Return `(X, y)` message texts and integer labels from a DataFrame."""
    X = data["message"]
    y = data["label"].map(LABEL_MAP)
    return X, y


def train_test(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

"""Tests for the spam detector package."""

from pathlib import Path

import pytest

from spam_detector.data import (
    DEFAULT_DATASET_PATH,
    LABEL_MAP,
    load_dataset,
    split_features_labels,
)
from spam_detector.predict import predict_messages, predict_one
from spam_detector.train import (
    build_pipeline,
    find_threshold,
    load_model,
    save_model,
)

# ------------------------------------------------------------------- data


def test_dataset_file_exists():
    assert DEFAULT_DATASET_PATH.exists(), "Dataset file is missing from the project."


def test_load_dataset_columns_and_labels():
    data = load_dataset()
    assert list(data.columns) == ["label", "message"]
    assert set(data["label"].unique()) <= {"ham", "spam"}
    assert len(data) > 0
    # No empty messages after cleaning.
    assert (data["message"].str.len() > 0).all()


def test_split_features_labels_encoding():
    data = load_dataset()
    X, y = split_features_labels(data)
    assert len(X) == len(y)
    assert set(y.unique()) == {LABEL_MAP["ham"], LABEL_MAP["spam"]}


# ------------------------------------------------------------------ model


@pytest.fixture(scope="module")
def tiny_pipeline():
    """Train a small pipeline on a varied set of obvious examples.

    Includes ham messages that reuse some spammy words (free, win) so the
    model learns context rather than latching onto a single token.
    """
    messages = [
        # spam
        "Congratulations! You have won a free prize. Click now!",
        "WINNER!! You have been selected for a free gift card.",
        "Free entry in a competition to win cash. Text WIN to 80085.",
        "Urgent: claim your free cash reward now, call this number",
        "You are a lucky winner! Redeem your prize immediately",
        "Buy cheap meds online, best prices, no prescription needed",
        # ham
        "Hey, are we still meeting for lunch tomorrow?",
        "Can you pick up some milk on your way home?",
        "Meeting moved to 3pm. See you there.",
        "Are you free for dinner tonight?",
        "I will be home in ten minutes, need anything?",
        "Great game last night, we really won that one",
    ]
    labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    pipeline = build_pipeline("logreg")
    pipeline.fit(messages, labels)
    return pipeline


def test_pipeline_predicts_obvious_examples(tiny_pipeline):
    spam = predict_one(
        "You won a free prize! Claim your cash reward now!", pipeline=tiny_pipeline
    )
    ham = predict_one("See you at the meeting tomorrow morning.", pipeline=tiny_pipeline)
    assert spam["is_spam"] is True
    assert ham["is_spam"] is False


def test_predict_messages_returns_one_result_per_input(tiny_pipeline):
    inputs = ["Free money now", "See you at work", "Claim your prize"]
    results = predict_messages(inputs, pipeline=tiny_pipeline)
    assert len(results) == len(inputs)
    for r in results:
        assert r["label"] in ("spam", "ham")
        assert isinstance(r["is_spam"], bool)
        assert 0.0 <= r["probability"] <= 1.0


def test_predict_one_handles_single_string(tiny_pipeline):
    result = predict_one("hello there", pipeline=tiny_pipeline)
    assert isinstance(result, dict)
    assert result["label"] in ("spam", "ham")


def test_threshold_changes_outcome(tiny_pipeline):
    msg = "Maybe you won something, not sure"
    low = predict_one(msg, pipeline=tiny_pipeline, threshold=0.0)
    high = predict_one(msg, pipeline=tiny_pipeline, threshold=1.0)
    assert low["is_spam"] is True
    assert high["is_spam"] is False


# ------------------------------------------------------------ persistence


def test_save_and_load_roundtrip(tmp_path, tiny_pipeline):
    path = tmp_path / "model.joblib"
    save_model(tiny_pipeline, path)
    assert path.exists()

    loaded = load_model(path)
    original = predict_one("Free gift waiting for you", pipeline=tiny_pipeline)
    restored = predict_one("Free gift waiting for you", pipeline=loaded)
    assert original["is_spam"] == restored["is_spam"]
    assert original["probability"] == pytest.approx(restored["probability"])


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does_not_exist.joblib")


# -------------------------------------------------------------- thresholds


def test_find_threshold_returns_valid_values(tiny_pipeline):
    messages = ["win a prize free", "ok see you tomorrow", "claim cash now",
                "lunch at noon", "free entry winner", "call me later"]
    labels = [1, 0, 1, 0, 1, 0]
    info = find_threshold(tiny_pipeline, messages, labels)
    assert 0.0 <= info["threshold"] <= 1.0
    assert "precision" in info and "recall" in info


def test_build_pipeline_unknown_classifier_raises():
    with pytest.raises(ValueError):
        build_pipeline("not-a-model")

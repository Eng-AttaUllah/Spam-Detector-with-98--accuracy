"""Prediction helpers built on top of a saved model pipeline."""

import numpy as np

from .data import LABEL_NAMES
from .train import DEFAULT_MODEL_PATH, load_model


def predict_messages(messages, pipeline=None, model_path=DEFAULT_MODEL_PATH,
                     threshold=0.5):
    """Classify one or more messages.

    Parameters
    ----------
    messages : str or list[str]
        The message text (or texts) to classify.
    pipeline : fitted Pipeline, optional
        If omitted, the model is loaded from `model_path`.
    model_path : str or Path
        Where to load the model from when `pipeline` is not supplied.
    threshold : float
        Decision threshold applied to the spam probability. A message is
        flagged as spam when its probability is >= threshold.

    Returns
    -------
    list[dict]
        One dict per input message with `label`, `is_spam`, and `probability`.
    """
    if pipeline is None:
        pipeline = load_model(model_path)

    if isinstance(messages, str):
        messages = [messages]

    texts = [str(m) for m in messages]

    proba = pipeline.predict_proba(texts)[:, 1]
    is_spam = proba >= threshold

    results = []
    for text, p, spam in zip(texts, proba, is_spam):
        results.append(
            {
                "message": text,
                "label": LABEL_NAMES[int(spam)],
                "is_spam": bool(spam),
                "probability": float(p),
            }
        )
    return results


def predict_one(message, pipeline=None, model_path=DEFAULT_MODEL_PATH,
                threshold=0.5):
    """Classify a single message and return the result dict."""
    return predict_messages(
        message, pipeline=pipeline, model_path=model_path, threshold=threshold
    )[0]

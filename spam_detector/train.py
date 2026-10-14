"""Model building, training, evaluation, and persistence."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Default location for the persisted model.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "spam_model.joblib"


def build_pipeline(classifier="logreg"):
    """Build a TF-IDF + classifier pipeline.

    `classifier` selects the model:
      - "logreg": Logistic Regression with balanced class weights (default)
      - "nb":     Multinomial Naive Bayes
      - "svc":    Linear SVM
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    if classifier == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    elif classifier == "nb":
        clf = MultinomialNB()
    elif classifier == "svc":
        clf = LinearSVC(class_weight="balanced")
    else:
        raise ValueError(
            f"Unknown classifier {classifier!r}; choose from 'logreg', 'nb', 'svc'."
        )

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def _decision_scores(pipeline, X):
    """Return scores suitable for ROC-AUC / precision-recall.

    LogisticRegression and MultinomialNB expose `predict_proba`; LinearSVC
    only exposes `decision_function`. We prefer probabilities when available.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def cross_validate_pipeline(pipeline, X, y, cv=5, random_state=42):
    """Stratified k-fold cross-validation.

    Returns a dict with mean/std for accuracy, F1, and (where possible) ROC-AUC.
    """
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    try:
        results = cross_validate(
            pipeline, X, y, cv=skf, scoring=scoring, return_train_score=False
        )
        use_auc = True
    except Exception:
        # Some estimators cannot produce probabilities for roc_auc.
        scoring = {k: v for k, v in scoring.items() if k != "roc_auc"}
        results = cross_validate(
            pipeline, X, y, cv=skf, scoring=scoring, return_train_score=False
        )
        use_auc = False

    summary = {
        "accuracy_mean": float(np.mean(results["test_accuracy"])),
        "accuracy_std": float(np.std(results["test_accuracy"])),
        "f1_mean": float(np.mean(results["test_f1"])),
        "f1_std": float(np.std(results["test_f1"])),
        "cv": cv,
    }
    if use_auc:
        summary["roc_auc_mean"] = float(np.mean(results["test_roc_auc"]))
        summary["roc_auc_std"] = float(np.std(results["test_roc_auc"]))
    return summary


def compare_models(X, y, cv=5, random_state=42):
    """Cross-validate each candidate classifier and return a ranked dict."""
    report = {}
    for name in ("logreg", "nb", "svc"):
        pipeline = build_pipeline(name)
        try:
            summary = cross_validate_pipeline(
                pipeline, X, y, cv=cv, random_state=random_state
            )
        except Exception as exc:  # pragma: no cover - defensive
            summary = {"error": str(exc)}
        report[name] = summary
    return report


def tune_pipeline(X, y, cv=5, random_state=42):
    """Grid-search the best hyperparameters for Logistic Regression.

    Returns the fitted GridSearchCV object.
    """
    pipeline = build_pipeline("logreg")

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "clf__C": [0.5, 1.0, 2.0],
    }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=skf,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    return search


def find_threshold(pipeline, X, y, target_recall=None):
    """Suggest a decision threshold.

    By default returns the threshold that maximises F1. If `target_recall` is
    given (e.g. 0.98), returns the highest threshold that still meets that
    recall, trading some precision for fewer missed spam messages.
    """
    scores = _decision_scores(pipeline, X)
    precision, recall, thresholds = precision_recall_curve(y, scores)

    # precision/recall arrays are one element longer than thresholds.
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-9
    )

    if target_recall is None:
        idx = int(np.argmax(f1))
        return {
            "threshold": float(thresholds[idx]),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "mode": "max_f1",
        }

    # Highest threshold whose recall still meets the target.
    eligible = np.where(recall[:-1] >= target_recall)[0]
    if len(eligible) == 0:
        idx = int(np.argmax(recall[:-1]))
    else:
        idx = int(eligible[np.argmax(thresholds[eligible])])
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(f1[idx]),
        "mode": f"target_recall>={target_recall}",
    }


def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """Train the tuned pipeline on a train split and evaluate on the hold-out.

    Returns a dict containing the fitted pipeline, metrics, and threshold info.
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    search = tune_pipeline(X_train, y_train, random_state=random_state)
    pipeline = search.best_estimator_

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Ham", "Spam"], zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "best_params": search.best_params_,
    }

    try:
        scores = _decision_scores(pipeline, X_test)
        metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
    except Exception:
        metrics["roc_auc"] = None

    threshold_info = find_threshold(pipeline, X_train, y_train)

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "threshold": threshold_info,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_model(pipeline, path=DEFAULT_MODEL_PATH):
    """Persist the fitted pipeline to disk with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path=DEFAULT_MODEL_PATH):
    """Load a previously saved pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train one first with `python train.py`."
        )
    return joblib.load(path)

#!/usr/bin/env python3
"""Train the spam detector and save the resulting model.

Usage:
    python train.py                      # train with defaults
    python train.py --cv 10              # 10-fold cross-validation
    python train.py --model-path models/custom.joblib
"""

import argparse

from spam_detector.data import load_dataset, split_features_labels
from spam_detector.train import (
    DEFAULT_MODEL_PATH,
    compare_models,
    cross_validate_pipeline,
    save_model,
    train_and_evaluate,
    build_pipeline,
)


def main():
    parser = argparse.ArgumentParser(description="Train the SMS spam detector.")
    parser.add_argument(
        "--cv", type=int, default=5, help="Number of cross-validation folds."
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Hold-out test fraction."
    )
    parser.add_argument(
        "--model-path", default=str(DEFAULT_MODEL_PATH), help="Where to save the model."
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip grid-search tuning and train with default hyperparameters.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Cross-validate every candidate classifier before training.",
    )
    args = parser.parse_args()

    print("Loading dataset...")
    data = load_dataset()
    X, y = split_features_labels(data)
    print(f"  {len(data)} messages ({int(y.sum())} spam, {int((1 - y).sum())} ham)")

    if args.compare:
        print(f"\nComparing classifiers ({args.cv}-fold CV)...")
        for name, summary in compare_models(X, y, cv=args.cv).items():
            if "error" in summary:
                print(f"  {name:6s} -> error: {summary['error']}")
            else:
                auc = (
                    f", ROC-AUC {summary['roc_auc_mean']:.4f}"
                    if "roc_auc_mean" in summary
                    else ""
                )
                print(
                    f"  {name:6s} -> accuracy {summary['accuracy_mean']:.4f} "
                    f"+/- {summary['accuracy_std']:.4f}, "
                    f"F1 {summary['f1_mean']:.4f}{auc}"
                )

    if args.no_tune:
        print(f"\nTraining default LogisticRegression ({args.cv}-fold CV)...")
        summary = cross_validate_pipeline(
            build_pipeline("logreg"), X, y, cv=args.cv
        )
        print(
            f"  CV accuracy {summary['accuracy_mean']:.4f} "
            f"+/- {summary['accuracy_std']:.4f}, "
            f"F1 {summary['f1_mean']:.4f}"
        )
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        pipeline = build_pipeline("logreg")
        pipeline.fit(X_train, y_train)
        from spam_detector.train import find_threshold

        threshold_info = find_threshold(pipeline, X_train, y_train)
        from sklearn.metrics import accuracy_score, classification_report, f1_score

        y_pred = pipeline.predict(X_test)
        print(f"\nHold-out accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Hold-out F1:      {f1_score(y_test, y_pred):.4f}")
        print("\n" + classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    else:
        print("\nTuning and training...")
        result = train_and_evaluate(
            X, y, test_size=args.test_size
        )
        pipeline = result["pipeline"]
        metrics = result["metrics"]
        threshold_info = result["threshold"]

        print(f"  Best params: {metrics['best_params']}")
        print(f"\nHold-out accuracy: {metrics['accuracy']:.4f}")
        print(f"Hold-out F1:      {metrics['f1']:.4f}")
        if metrics.get("roc_auc") is not None:
            print(f"Hold-out ROC-AUC: {metrics['roc_auc']:.4f}")
        print("\nClassification report:")
        print(metrics["classification_report"])
        print("Confusion matrix:")
        print(metrics["confusion_matrix"])

    print(
        f"\nSuggested decision threshold: {threshold_info['threshold']:.4f} "
        f"(precision {threshold_info['precision']:.3f}, "
        f"recall {threshold_info['recall']:.3f}, F1 {threshold_info['f1']:.3f})"
    )

    path = save_model(pipeline, args.model_path)
    print(f"\nModel saved to: {path}")


if __name__ == "__main__":
    main()

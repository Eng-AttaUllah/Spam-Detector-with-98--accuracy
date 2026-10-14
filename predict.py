#!/usr/bin/env python3
"""Classify SMS messages using the trained spam detector.

Usage:
    python predict.py "Congratulations! You have won a free prize."
    python predict.py --file messages.txt
    python predict.py --threshold 0.7 "Free entry to win..."
"""

import argparse
import sys

from spam_detector.predict import predict_messages
from spam_detector.train import DEFAULT_MODEL_PATH, load_model


def main():
    parser = argparse.ArgumentParser(description="Classify messages as spam or ham.")
    parser.add_argument(
        "messages", nargs="*", help="Message(s) to classify."
    )
    parser.add_argument(
        "--file",
        help="Read messages from a file, one per line, instead of the command line.",
    )
    parser.add_argument(
        "--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to the saved model."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Spam probability threshold (default 0.5).",
    )
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            messages = [line.rstrip("\n") for line in fh if line.strip()]
    elif args.messages:
        messages = args.messages
    else:
        parser.error("Provide at least one message or use --file.")

    pipeline = load_model(args.model_path)
    results = predict_messages(
        messages, pipeline=pipeline, threshold=args.threshold
    )

    for r in results:
        verdict = "SPAM" if r["is_spam"] else "HAM"
        print(f"{verdict:4s}  (spam prob {r['probability']:.3f})  {r['message']}")


if __name__ == "__main__":
    main()

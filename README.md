# SMS Spam Detector

A machine-learning SMS spam classifier built with scikit-learn. It vectorizes
message text with TF-IDF and classifies each message as **spam** or **ham**
using Logistic Regression (with class weighting to handle the imbalance between
ham and spam).

The project is structured as a reusable Python package with a training script,
a prediction CLI, an interactive Streamlit app, and a pytest test suite.

## Features

- **TF-IDF + Logistic Regression pipeline** — the vectorizer and model are
  bundled into a single scikit-learn `Pipeline`, so they always travel together.
- **Cross-validation** — stratified k-fold CV reports accuracy, F1, and ROC-AUC.
- **Model comparison** — optionally cross-validates Logistic Regression,
  Multinomial Naive Bayes, and Linear SVM side by side.
- **Hyperparameter tuning** — `GridSearchCV` over n-gram range, `min_df`, and
  regularization strength.
- **Threshold tuning** — suggests a decision threshold that maximises F1, or
  meets a target recall to catch more spam.
- **Model persistence** — the trained pipeline is saved with `joblib` so
  prediction doesn't require retraining.
- **Interactive app** — a Streamlit UI for single and bulk classification,
  with an adjustable sensitivity threshold.
- **Tests** — a pytest suite covering data loading, prediction, persistence,
  and thresholds.

## Project structure

```
.
├── spam_detector/          # Reusable package
│   ├── __init__.py
│   ├── data.py             # Load & clean the dataset
│   ├── train.py            # Build, tune, evaluate, save/load the model
│   └── predict.py          # Classify messages with a saved model
├── app.py                  # Streamlit interactive app
├── train.py                # CLI: train and save the model
├── predict.py              # CLI: classify messages
├── tests/
│   └── test_spam_detector.py
├── models/                 # Saved model(s) (gitignored, created on train)
├── spam-detector.ipynb     # Original exploration notebook (kept for reference)
├── spam or ham dataset.csv # SMS Spam Collection dataset
├── requirements.txt
└── conftest.py
```

## Setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
python train.py
```

This loads the dataset, runs cross-validation, tunes hyperparameters, prints
hold-out metrics (accuracy, F1, ROC-AUC, classification report, confusion
matrix), suggests a decision threshold, and saves the model to
`models/spam_model.joblib`.

Useful options:

```bash
python train.py --cv 10            # 10-fold cross-validation
python train.py --compare          # compare LogReg, Naive Bayes, and Linear SVM
python train.py --no-tune          # skip grid search, train with defaults
python train.py --model-path models/custom.joblib
```

### Classify messages

```bash
python predict.py "Congratulations! You have won a free prize. Click now!"
python predict.py "Are we still meeting for lunch?"
python predict.py --threshold 0.7 "Free entry to win a cash prize"
python predict.py --file messages.txt
```

### Run the interactive app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (default `http://localhost:8501`).
The sidebar includes a threshold slider to adjust how aggressively the model
flags spam.

### Run the tests

```bash
pytest
```

## How it works

1. **Load** — the raw CSV (`v1` = label, `v2` = message) is cleaned: columns
   renamed, labels mapped to `0`/`1`, duplicates and empty messages dropped.
2. **Vectorize** — `TfidfVectorizer` with English stop words and unigrams +
   bigrams converts text into numerical features.
3. **Classify** — a `LogisticRegression` with `class_weight="balanced"` trains
   on the resulting features.
4. **Evaluate** — stratified cross-validation plus a held-out test set report
   accuracy, precision, recall, F1, and ROC-AUC.
5. **Persist** — the full pipeline is saved with `joblib` for later reuse.

## Dataset

Uses the classic **SMS Spam Collection** dataset (~5,500 labeled SMS messages).

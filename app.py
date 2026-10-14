"""Interactive Streamlit app for the SMS spam detector.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from spam_detector.predict import predict_messages
from spam_detector.train import DEFAULT_MODEL_PATH, load_model

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="centered",
)

# ---------------------------------------------------------------- Load model
@st.cache_resource
def get_pipeline(model_path: str):
    """Load the trained pipeline once and cache it."""
    return load_model(model_path)


model_path = st.sidebar.text_input("Model path", str(DEFAULT_MODEL_PATH))

if not Path(model_path).exists():
    st.error(
        f"No trained model found at `{model_path}`.\n\n"
        "Train one first by running `python train.py` in the project directory."
    )
    st.stop()

try:
    pipeline = get_pipeline(model_path)
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

# ------------------------------------------------------------------- Sidebar
st.sidebar.title("🛡️ SMS Spam Detector")
st.sidebar.markdown(
    "A TF-IDF + Logistic Regression classifier trained on the classic "
    "SMS Spam Collection dataset."
)

threshold = st.sidebar.slider(
    "Spam probability threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Messages with a spam probability at or above this value are flagged as spam. "
    "Lower it to catch more spam (at the cost of more false alarms).",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tip:** lower the threshold (e.g. 0.35) to catch more spam, "
    "or raise it (e.g. 0.75) to reduce false positives."
)

# --------------------------------------------------------------------- Main
st.title("📩 Classify a message")

user_input = st.text_area(
    "Paste an SMS message below:",
    height=120,
    placeholder="Congratulations! You have won a free prize. Click now!",
)

col_classify, col_clear = st.columns([1, 1])
with col_classify:
    classify = st.button("Classify", type="primary")
with col_clear:
    st.button("Clear")

if classify:
    if not user_input.strip():
        st.warning("Please enter a message first.")
    else:
        result = predict_messages(
            user_input, pipeline=pipeline, threshold=threshold
        )[0]

        if result["is_spam"]:
            st.error("🚫 **SPAM**")
        else:
            st.success("✅ **HAM** (not spam)")

        prob = result["probability"]
        st.metric("Spam probability", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(
            f"Classified using a threshold of {threshold:.2f}. "
            "Adjust it in the sidebar to change sensitivity."
        )

# --------------------------------------------------------------- Batch mode
st.markdown("---")
st.subheader("Bulk check")
st.caption("Enter one message per line.")

bulk_input = st.text_area("Messages", height=150, label_visibility="collapsed")

if st.button("Classify all") and bulk_input.strip():
    lines = [ln for ln in bulk_input.splitlines() if ln.strip()]
    results = predict_messages(lines, pipeline=pipeline, threshold=threshold)
    df = pd.DataFrame(
        [
            {
                "Verdict": r["label"].upper(),
                "Spam probability": f"{r['probability']:.3f}",
                "Message": r["message"],
            }
            for r in results
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    spam_count = sum(r["is_spam"] for r in results)
    st.caption(f"{spam_count} of {len(results)} messages flagged as spam.")

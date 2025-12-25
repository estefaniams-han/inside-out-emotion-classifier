import joblib
import streamlit as st

LABELS_ORDER = ["anger", "disgust", "fear", "sadness", "joy", "surprise", "neutral"]
EMOJI = {
    "anger": "😡",
    "disgust": "🤢",
    "fear": "😱",
    "sadness": "😢",
    "joy": "😄",
    "surprise": "😲",
    "neutral": "😐",
}

st.set_page_config(page_title="Inside Out Emotion Classifier", page_icon="🎭")

@st.cache_resource
def load_model():
    return joblib.load("models/emotion_lr_tfidf_best.joblib")

model = load_model()

st.title("🎭 Inside Out Emotion Classifier")
st.write("Write a sentence and see which Inside Out emotion it corresponds to!")

text = st.text_area("Enter your text here:", height=140, placeholder="Type something...")

col1, col2 = st.columns(2)
with col1:
    analyze = st.button("Analyze Emotion")
with col2:
    st.caption("This model was trained with text data from Reddit")

if analyze:
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        pred = model.predict([text])[0]
        st.success(f"The predicted emotion is: **{pred} {EMOJI.get(pred, '')}**")

        # Probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            pairs = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)
            st.subheader("Probabilities:")
            for label, p in pairs:
                st.progress(float(p), text=f"{EMOJI.get(label, '')} {label}: {p:.2%}")
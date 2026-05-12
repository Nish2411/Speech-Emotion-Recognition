import pickle
import tempfile
from pathlib import Path
import os

import streamlit as st

from realtime_predict import predict_file
from utils import TARGET_SAMPLE_RATE


MODEL_PATH = Path("model/emotion_model.pkl")
EMOTION_STYLES = {
    "happy": {"emoji": "HAPPY", "accent": "#f59e0b"},
    "sad": {"emoji": "SAD", "accent": "#2563eb"},
    "angry": {"emoji": "ANGRY", "accent": "#ef4444"},
    "neutral": {"emoji": "NEUTRAL", "accent": "#0f766e"},
}


st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def save_uploaded_audio(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def format_probabilities(probabilities):
    if not probabilities:
        return []
    return sorted(
        ((str(label), float(score)) for label, score in probabilities.items()),
        key=lambda item: item[1],
        reverse=True,
    )


def render_confidence(probabilities):
    ordered = format_probabilities(probabilities)
    if not ordered:
        return

    st.markdown("### Confidence breakdown")
    for label, score in ordered:
        st.write(f"{label.title()} — {score * 100:.1f}%")
        st.progress(score)


def render_prediction(prediction, probabilities):
    style = EMOTION_STYLES.get(str(prediction).lower(), {"emoji": "VOICE", "accent": "#111827"})
    top_score = max((float(score) for score in probabilities.values()), default=0.0) if probabilities else 0.0

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {style['accent']} 0%, #111827 100%);
            border-radius: 24px;
            padding: 28px;
            color: white;
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
        ">
            <div style="font-size: 0.8rem; letter-spacing: 0.18em; text-transform: uppercase; opacity: 0.82;">
                Prediction
            </div>
            <div style="font-size: 2.5rem; font-weight: 800; margin-top: 0.4rem;">
                {style['emoji']} {str(prediction).title()}
            </div>
            <div style="margin-top: 0.7rem; font-size: 1rem; opacity: 0.88;">
                Model confidence: {top_score * 100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    load_model()

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(245, 158, 11, 0.18), transparent 24%),
                radial-gradient(circle at top right, rgba(37, 99, 235, 0.16), transparent 25%),
                linear-gradient(180deg, #fffdf8 0%, #eef4ff 100%);
        }
        .block-container {
            max-width: 1120px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        .hero {
            background: rgba(255, 255, 255, 0.78);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 28px;
            padding: 28px;
            box-shadow: 0 24px 80px rgba(15, 23, 42, 0.08);
        }
        .hero h1 {
            margin: 0;
            font-size: 3rem;
            line-height: 1.05;
            color: #0f172a;
        }
        .hero p {
            margin: 0.9rem 0 0 0;
            color: #334155;
            font-size: 1.05rem;
            max-width: 48rem;
        }
        .info-card {
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 22px;
            padding: 20px;
            color: #0f172a;
            min-height: 160px;
        }
        .info-card h3 {
            margin-top: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>Speech Emotion Recognition</h1>
            <p>
                Record your voice in the browser or upload a clip, then let the model estimate the
                emotion from the audio signal. The app uses the same preprocessing and trained model
                as the Python scripts, now wrapped in a cleaner interface.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("### Record or upload")
        st.caption(f"Recommended sample rate: {TARGET_SAMPLE_RATE} Hz. Browser recording may vary, and the app will resample automatically.")

        audio_input = st.audio_input("Use your microphone", help="Record directly in the browser and submit a short speech sample.")
        uploaded_file = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])

        selected_audio = audio_input or uploaded_file

        if selected_audio is not None:
            st.audio(selected_audio)

            if st.button("Analyze Emotion", type="primary", use_container_width=True):
                temp_path = save_uploaded_audio(selected_audio)
                try:
                    prediction, probabilities = predict_file(temp_path)

                    if prediction is None:
                        st.error("The recording was too short or too quiet. Try speaking a little longer and closer to the mic.")
                    else:
                        render_prediction(prediction, probabilities)
                        render_confidence(probabilities)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            st.info("Record a voice sample or upload an audio clip to get a prediction.")

    with right:
        st.markdown(
            """
            <div class="info-card">
                <h3>How to use it</h3>
                <p>1. Record 2 to 5 seconds of speech with a clear tone.</p>
                <p>2. Click <strong>Analyze Emotion</strong>.</p>
                <p>3. Review the predicted label and confidence scores.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            """
            <div class="info-card">
                <h3>Best results</h3>
                <p>The model was trained on acted emotional speech, so stronger expression usually gives better predictions.</p>
                <p>Quiet rooms and short pauses before speaking also help reduce false positives.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

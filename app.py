
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from datetime import datetime

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="RetinaAI | DR Screening",
    page_icon="👁️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "web_app")

B0_PATH = os.path.join(
    MODEL_DIR, "ensemble_efficientnetb0_300.keras"
)
B3_PATH = os.path.join(
    MODEL_DIR, "ensemble_efficientnetb3_300.keras"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #F4F8FC;
    color: #183153;
}
h1, h2, h3, h4 {
    color: #12304A !important;
}
p, label, .stMarkdown, .stText {
    color: #263E55;
}
[data-testid="stSidebar"] {
    background-color: #102A43;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [aria-checked="true"] {
    background-color: #176B87;
}
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #D9E2EC;
}
.stButton button p {
    color: #FFFFFF !important;
}
.stCaption {
    color: #627D98 !important;
}
.hero {
    background: linear-gradient(120deg, #0B2038, #135B70, #168C91);
    padding: 42px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(15,45,70,0.15);
}
.hero h1 {
    color: white !important;
    font-size: 42px !important;
    margin-bottom: 8px;
}
.hero p {
    color: #E1F7F7 !important;
    font-size: 17px;
}
.tag {
    background: rgba(255,255,255,0.15);
    padding: 7px 13px;
    border-radius: 20px;
    display: inline-block;
    color: white;
    font-size: 12px;
    margin-bottom: 18px;
}
.info-card {
    background: white;
    border: 1px solid #DFEAF1;
    border-radius: 16px;
    padding: 22px;
    min-height: 145px;
    box-shadow: 0 4px 14px rgba(20,50,80,0.05);
}
.info-card h3 {
    color: #176B87 !important;
    font-size: 14px !important;
}
.info-card h2 {
    color: #12304A !important;
    font-size: 25px !important;
}
.info-card p {
    color: #627D98 !important;
    font-size: 13px;
}
.grade-card {
    background: white;
    border: 1px solid #DFEAF1;
    border-radius: 14px;
    padding: 16px 8px;
    text-align: center;
    min-height: 135px;
}
.grade-card h2 {
    font-size: 30px;
    margin: 0;
}
.grade-card p {
    color: #12304A !important;
    font-weight: 650;
    font-size: 13px;
}
.section-note {
    color: #627D98 !important;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- ENSEMBLE MODELS ----------------

@st.cache_resource
def load_models():
    b0 = tf.keras.models.load_model(B0_PATH)
    b3 = tf.keras.models.load_model(B3_PATH)
    return b0, b3

if not os.path.exists(B0_PATH) or not os.path.exists(B3_PATH):
    st.error("One or both ensemble model files are missing.")
    st.write("Expected model folder:", MODEL_DIR)
    st.stop()

try:
    b0_model, b3_model = load_models()
except Exception as exc:
    st.error("Could not load the EfficientNet ensemble models.")
    st.exception(exc)
    st.stop()

# ---------------- PREDICTION FUNCTION ----------------

def predict_ensemble(image):
    processed_image = image.resize((300, 300))

    image_array = np.array(
        processed_image, dtype=np.float32
    )
    image_array = np.expand_dims(image_array, axis=0)

    # Horizontal-flip test-time augmentation
    flipped_array = np.flip(
        image_array, axis=2
    ).copy()

    # EfficientNetB0 predictions
    b0_original = b0_model.predict(image_array, verbose=0)
    b0_flipped = b0_model.predict(flipped_array, verbose=0)
    b0_tta = (b0_original + b0_flipped) / 2

    # EfficientNetB3 predictions
    b3_original = b3_model.predict(image_array, verbose=0)
    b3_flipped = b3_model.predict(flipped_array, verbose=0)
    b3_tta = (b3_original + b3_flipped) / 2

    # Average both model outputs
    final_predictions = (b0_tta + b3_tta) / 2
    predictions = np.asarray(final_predictions).reshape(-1)

    if predictions.size != 5:
        raise ValueError(
            f"Expected five class scores, got {predictions.size}."
        )

    if not np.all(np.isfinite(predictions)):
        raise ValueError("The model returned invalid prediction values.")

    return predictions

# ---------------- DISEASE GRADES ----------------

DR_GRADES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}

GRADE_INFO = {
    0: "No visible signs of diabetic retinopathy in the image.",
    1: "Mild changes may be present in the retinal blood vessels.",
    2: "Moderate retinal changes may be present.",
    3: "Severe retinal abnormalities may be present.",
    4: "Advanced retinal changes may be present."
}

# ---------------- SESSION HISTORY ----------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------

st.sidebar.title("👁️ RetinaAI")
st.sidebar.caption("AI-Based Retinal Image Analysis")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Analyze Image", "History & Analytics"]
)

st.sidebar.divider()
st.sidebar.info(
    "Research prototype based on the APTOS 2019 dataset."
)
st.sidebar.caption("EfficientNetB0 + EfficientNetB3 Ensemble")

# ---------------- HOME PAGE ----------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="tag">✦ AI-POWERED RETINAL ANALYSIS</div>
        <h1>Welcome to RetinaAI</h1>
        <p>
        Exploring diabetic retinopathy through deep learning
        and intelligent retinal image classification.
        </p>
        <p>
        EfficientNetB0 + EfficientNetB3 Ensemble
        &nbsp; • &nbsp; APTOS 2019 Dataset
        </p>
    </div>
    """, unsafe_allow_html=True)

    total = len(st.session_state.history)

    st.markdown("### ✦ Dashboard at a glance")
    st.markdown(
        '<p class="section-note">Your application overview</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="info-card">
            <h3>◉ TOTAL PREDICTIONS</h3>
            <h2>{total}</h2>
            <p>Images analyzed in this session</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>◎ CLASSIFICATION</h3>
            <h2>05 Grades</h2>
            <p>Severity categories supported</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
            <h3>✧ DEEP LEARNING MODEL</h3>
            <h2>B0 + B3 Ensemble</h2>
            <p>Two EfficientNet architectures</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### ✦ Understanding the five grades")
    st.markdown(
        '<p class="section-note">Five categories used by the classification model</p>',
        unsafe_allow_html=True
    )

    grade_data = [
        ("0", "No DR", "#21865B"),
        ("1", "Mild", "#3688B7"),
        ("2", "Moderate", "#C58A26"),
        ("3", "Severe", "#D46C3D"),
        ("4", "Proliferative", "#B74755")
    ]

    cols = st.columns(5)

    for col, (number, name, color) in zip(cols, grade_data):
        with col:
            st.markdown(f"""
            <div class="grade-card">
                <h2 style="color:{color} !important;">{number}</h2>
                <p>{name}</p>
                <span style="color:#829AB1;font-size:11px;">
                    DR GRADE
                </span>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### ✦ How RetinaAI works")

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="info-card">
            <h2>📤</h2>
            <h3>01 · UPLOAD IMAGE</h3>
            <p>Choose a retinal fundus photograph in JPG or PNG format.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="info-card">
            <h2>🧠</h2>
            <h3>02 · AI ANALYSIS</h3>
            <p>Two EfficientNet models analyze the image and its horizontal flip.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="info-card">
            <h2>📊</h2>
            <h3>03 · VIEW RESULTS</h3>
            <p>Review the ensemble prediction, grade scores and session history.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div style="
        background:#E7F3F7;
        border-left:5px solid #168C91;
        padding:20px;
        border-radius:12px;
    ">
        <h3 style="color:#12304A;">About the project</h3>
        <p style="color:#36566F;">
        RetinaAI is a research application developed using
        the APTOS 2019 Blindness Detection dataset. It explores
        five-class diabetic retinopathy image classification
        using an EfficientNetB0 and EfficientNetB3 ensemble
        with horizontal-flip test-time augmentation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.warning(
        "Research prototype only. Model predictions are not "
        "medical diagnoses. Consult an ophthalmologist for "
        "clinical assessment."
    )

# ---------------- IMAGE ANALYSIS ----------------

elif page == "Analyze Image":

    st.title("🔬 Analyze Retinal Image")
    st.write(
        "Upload a retinal fundus photograph for AI-based classification."
    )

    uploaded_file = st.file_uploader(
        "Choose retinal image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image).convert("RGB")
        except Exception:
            st.error("The uploaded file could not be opened as an image.")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Retinal Image",
                use_container_width=True
            )

        with col2:
            st.write("### Image Information")
            st.write("File name:", uploaded_file.name)
            st.write("Image dimensions:", image.size)
            st.write("Model input:", "300 × 300")

        if st.button("Analyze Image", type="primary"):

            with st.spinner(
                "Analyzing image with EfficientNet ensemble..."
            ):
                try:
                    predictions = predict_ensemble(image)

                    predicted_grade = int(np.argmax(predictions))
                    score = float(predictions[predicted_grade] * 100)

                    result = {
                        "Grade": predicted_grade,
                        "Diagnosis": DR_GRADES[predicted_grade],
                        "Score": round(score, 2),
                        "Time": datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        ),
                        "Image": uploaded_file.name,
                        "Grade Scores": [
                            round(float(value) * 100, 2)
                            for value in predictions
                        ]
                    }

                    st.session_state.last_prediction = result
                    st.session_state.history.append(result.copy())

                except Exception as exc:
                    st.error("An error occurred while analyzing the image.")
                    st.exception(exc)
                    st.stop()

        if "last_prediction" in st.session_state:
            result = st.session_state.last_prediction

            st.divider()
            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Grade",
                f"Grade {result['Grade']}"
            )

            col2.metric(
                "Ensemble Score",
                f"{result['Score']:.2f}%"
            )

            st.success(result["Diagnosis"])
            st.write(GRADE_INFO[result["Grade"]])

            st.caption(
                "The ensemble score is not a calibrated probability "
                "or a measure of diagnostic certainty."
            )

            st.subheader("Scores Across All Five Grades")

            score_df = pd.DataFrame({
                "Grade": [f"Grade {i}" for i in range(5)],
                "Ensemble Score (%)": result["Grade Scores"]
            })

            st.bar_chart(score_df.set_index("Grade"))

            report = f"""
DIABETIC RETINOPATHY SCREENING REPORT

Image: {result['Image']}
Date: {result['Time']}

Predicted Grade: {result['Grade']}
Prediction: {result['Diagnosis']}
Ensemble Score: {result['Score']}%

Model: EfficientNetB0 + EfficientNetB3 Ensemble
Input Size: 300 x 300
Inference: Horizontal-flip test-time augmentation

IMPORTANT:
This is an AI research prototype, not a medical diagnosis.
Consult a qualified ophthalmologist for clinical assessment.
"""

            st.download_button(
                "📥 Download Prediction Report",
                data=report,
                file_name="DR_prediction_report.txt",
                mime="text/plain"
            )

    st.warning(
        "This application is for research and educational use only. "
        "It cannot diagnose or rule out diabetic retinopathy."
    )

# ---------------- HISTORY & ANALYTICS ----------------

elif page == "History & Analytics":

    st.title("📊 Prediction History & Analytics")
    history = st.session_state.history

    if len(history) == 0:
        st.info(
            "No prediction history available. Analyze an image first."
        )
    else:
        df = pd.DataFrame(history)

        st.subheader("Prediction Summary")

        col1, col2 = st.columns(2)

        col1.metric("Total Predictions", len(df))
        col2.metric(
            "Average Ensemble Score",
            f"{df['Score'].mean():.2f}%"
        )

        st.divider()
        st.subheader("Grade Distribution")

        grade_counts = (
            df["Grade"]
            .value_counts()
            .reindex(range(5), fill_value=0)
        )

        chart_df = pd.DataFrame({
            "Grade": [f"Grade {i}" for i in range(5)],
            "Number of Predictions": grade_counts.values
        })

        st.bar_chart(chart_df.set_index("Grade"))

        st.divider()
        st.subheader("Previous Predictions")

        display_df = df.drop(
            columns=["Grade Scores"], errors="ignore"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download History (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="DR_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):
            st.session_state.history = []
            if "last_prediction" in st.session_state:
                del st.session_state.last_prediction
            st.rerun()

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "RetinaAI | EfficientNetB0 + EfficientNetB3 Ensemble | "
    "APTOS 2019 Dataset | 300 × 300"
)
st.caption(
    "Research prototype only. Not intended for clinical diagnosis."
)

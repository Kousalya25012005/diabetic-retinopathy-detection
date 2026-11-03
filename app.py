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
    border: none;
}
.stButton button p {
    color: #FFFFFF !important;
}
.stCaption {
    color: #627D98 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------

MODEL_PATH = "dr_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(
        f"Could not load the model file '{MODEL_PATH}'. "
        "Make sure it is in the same folder as app.py and is a valid Keras model."
    )
    st.exception(exc)
    st.stop()

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

# ---------------- HOME PAGE ----------------

if page == "Home":
    st.title("👁️ Welcome to RetinaAI")
    st.subheader("AI-Based Diabetic Retinopathy Screening")

    st.write(
        "RetinaAI is a research prototype that uses a trained deep-learning "
        "model to classify retinal fundus images into five diabetic "
        "retinopathy grades."
    )

    st.info(
        "Select **Analyze Image** from the sidebar to upload a retinal "
        "fundus photograph and view the model's prediction."
    )

    st.subheader("Diabetic Retinopathy Grades")

    for grade, diagnosis in DR_GRADES.items():
        st.write(f"**Grade {grade}:** {diagnosis}")

    st.warning(
        "This application is for research and educational use only. "
        "It cannot diagnose or rule out diabetic retinopathy. "
        "Consult a qualified ophthalmologist for clinical assessment."
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
        except Exception as exc:
            st.error("The uploaded file could not be opened as an image.")
            st.exception(exc)
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

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing retinal image..."):
                try:
                    processed_image = image.resize((224, 224))

                    image_array = np.array(
                        processed_image,
                        dtype=np.float32
                    )
                    image_array = np.expand_dims(image_array, axis=0)

                    predictions = np.asarray(
                        model.predict(image_array, verbose=0)
                    ).reshape(-1)

                    if predictions.size != 5:
                        st.error(
                            "The model did not return five class scores. "
                            f"It returned {predictions.size} value(s). "
                            "Check that the loaded model has the expected "
                            "five-class output."
                        )
                        st.stop()

                    predicted_grade = int(np.argmax(predictions))
                    score = float(predictions[predicted_grade] * 100)

                    result = {
                        "Grade": predicted_grade,
                        "Diagnosis": DR_GRADES[predicted_grade],
                        "Score": round(score, 2),
                        "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
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
                "Model Score",
                f"{result['Score']:.2f}%"
            )

            st.success(result["Diagnosis"])
            st.write(GRADE_INFO[result["Grade"]])

            st.caption(
                "The model score is not a calibrated probability "
                "or a measure of diagnostic certainty."
            )

            st.subheader("Scores Across All Five Grades")

            score_df = pd.DataFrame({
                "Grade": [f"Grade {i}" for i in range(5)],
                "Model Score (%)": result["Grade Scores"]
            })

            st.bar_chart(score_df.set_index("Grade"))

            report = f"""
DIABETIC RETINOPATHY SCREENING REPORT

Image: {result['Image']}
Date: {result['Time']}

Predicted Grade: {result['Grade']}
Prediction: {result['Diagnosis']}
Model Score: {result['Score']}%

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
            "Average Model Score",
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

        display_df = df.drop(columns=["Grade Scores"], errors="ignore")

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

st.caption("RetinaAI | EfficientNetB0 | APTOS 2019 Dataset")
st.caption("Research prototype only. Not intended for clinical diagnosis.")

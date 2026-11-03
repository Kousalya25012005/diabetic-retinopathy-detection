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

/* Main background and default text */
.stApp {
    background-color: #F4F8FC;
    color: #183153;
}

/* Main headings */
h1, h2, h3, h4 {
    color: #12304A !important;
}

/* Paragraphs and normal text */
p, label, .stMarkdown, .stText {
    color: #263E55;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #102A43;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Sidebar selected navigation */
[data-testid="stSidebar"] [aria-checked="true"] {
    background-color: #176B87;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #D9E2EC;
}

[data-testid="stMetricLabel"] {
    color: #486581 !important;
}

[data-testid="stMetricValue"] {
    color: #12304A !important;
}

/* Input fields */
.stTextInput input,
.stFileUploader,
.stSelectbox div {
    color: #183153;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border: 1px solid #D9E2EC;
    border-radius: 10px;
}

/* Buttons */
.stButton button {
    background-color: #176B87;
    color: #FFFFFF;
    border-radius: 8px;
    border: none;
}

.stButton button p {
    color: #FFFFFF !important;
}

/* Footer */
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

model = load_model()

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


elif page == "Analyze Image":

    st.title("🔬 Analyze Retinal Image")

    st.write(
        "Upload a retinal fundus photograph for AI-based "
        "classification."
    )

    uploaded_file = st.file_uploader(
        "Choose retinal image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

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

                processed_image = image.resize((224, 224))

                image_array = np.array(
                    processed_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                predicted_grade = int(np.argmax(predictions))

                confidence = float(
                    predictions[predicted_grade] * 100
                )

            st.session_state.last_prediction = {
                "Grade": predicted_grade,
                "Diagnosis": DR_GRADES[predicted_grade],
                "Score": round(confidence, 2),
                "Time": datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),
                "Image": uploaded_file.name,
                "Grade Scores": [
                    round(float(x) * 100, 2)
                    for x in predictions
                ]
            }

            st.session_state.history.append(
                st.session_state.last_prediction.copy()
            )

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
                "Grade": [
                    f"Grade {i}" for i in range(5)
                ],
                "Model Score (%)": result["Grade Scores"]
            })

            st.bar_chart(
                score_df.set_index("Grade")
            )

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

# ---------------- IMAGE ANALYSIS ----------------

elif page == "Analyze Image":

    st.title("🔬 Analyze Retinal Image")

    st.write(
        "Upload a retinal fundus photograph for AI-based "
        "classification."
    )

    uploaded_file = st.file_uploader(
        "Choose retinal image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

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

                processed_image = image.resize((224, 224))

                image_array = np.array(
                    processed_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array, axis=0
                )

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                predicted_grade = int(np.argmax(predictions))

                confidence = float(
                    predictions[predicted_grade] * 100
                )

             st.session_state.last_prediction = {
                "Grade": predicted_grade,
                "Diagnosis": DR_GRADES[predicted_grade],
                "Score": round(confidence, 2),
                "Time": datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),
                "Image": uploaded_file.name,
                
            }

            st.session_state.history.append(
                st.session_state.last_prediction.copy()
            )

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
                "Grade": [
                    f"Grade {i}" for i in range(5)
                ],
                "Model Score (%)": [
                    round(float(x) * 100, 2)
                    for x in predictions
                ]
            })

            st.bar_chart(
                score_df.set_index("Grade")
            )

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
            "No prediction history available. "
            "Analyze an image first."
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
            "Grade": [
                f"Grade {i}" for i in range(5)
            ],
            "Number of Predictions": grade_counts.values
        })

        st.bar_chart(
            chart_df.set_index("Grade")
        )

        st.divider()

        st.subheader("Previous Predictions")

        st.dataframe(
            df,
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
    "RetinaAI | EfficientNetB0 | APTOS 2019 Dataset"
)

st.caption(
    "Research prototype only. Not intended for clinical diagnosis."
)

   
       
       

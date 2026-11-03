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


   
# ---------------- HOME PAGE ----------------

if page == "Home":

    st.markdown("""
    <style>
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
        font-size: 27px !important;
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

    <div class="hero">
        <div class="tag">✦ AI-POWERED RETINAL ANALYSIS</div>
        <h1>Welcome to RetinaAI</h1>
        <p>
        Exploring diabetic retinopathy through deep learning
        and intelligent retinal image classification.
        </p>
        <p>EfficientNetB0 &nbsp; • &nbsp; APTOS 2019 Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    total = len(st.session_state.history)
    grades_found = len(set(
        x["Grade"] for x in st.session_state.history
    ))

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
            <h2>EfficientNetB0</h2>
            <p>Image classification architecture</p>
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
                <h2 style="color:{color} !important;">
                    {number}
                </h2>
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
            <p>The EfficientNetB0 model generates classification scores.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="info-card">
            <h2>📊</h2>
            <h3>03 · VIEW RESULTS</h3>
            <p>Review the predicted grade, scores and session history.</p>
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
        using a pretrained EfficientNetB0 architecture.
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
                "Image": uploaded_file.name
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

   
       
       


import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

B0_PATH = os.path.join(BASE_DIR, "ensemble_efficientnetb0_300.keras")
B3_PATH = os.path.join(BASE_DIR, "ensemble_efficientnetb3_300.keras")

@st.cache_resource
def load_models():
    b0 = tf.keras.models.load_model(B0_PATH)
    b3 = tf.keras.models.load_model(B3_PATH)
    return b0, b3

b0_model, b3_model = load_models()

DR_GRADES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}

st.title("👁️ Diabetic Retinopathy Detection")
st.write(
    "Upload a retinal fundus image to receive an AI-based "
    "diabetic retinopathy grade prediction."
)

uploaded_file = st.file_uploader(
    "Upload retinal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")

    st.image(
        image,
        caption="Uploaded Retinal Image",
        use_container_width=True
    )

    if st.button("Analyze Retinal Image"):

        with st.spinner("Analyzing image..."):

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
            b0_original = b0_model.predict(
                image_array, verbose=0
            )
            b0_flipped = b0_model.predict(
                flipped_array, verbose=0
            )
            b0_tta = (b0_original + b0_flipped) / 2

            # EfficientNetB3 predictions
            b3_original = b3_model.predict(
                image_array, verbose=0
            )
            b3_flipped = b3_model.predict(
                flipped_array, verbose=0
            )
            b3_tta = (b3_original + b3_flipped) / 2

            # Average both model probability outputs
            predictions = (b0_tta + b3_tta) / 2
            predictions = predictions[0]

            predicted_grade = int(np.argmax(predictions))
            confidence = float(
                predictions[predicted_grade] * 100
            )

        st.subheader("Prediction Result")
        st.success(DR_GRADES[predicted_grade])

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.write("**Grade:**", predicted_grade)

        st.warning(
            "This application is a research prototype. "
            "It is not a medical diagnosis. Please consult "
            "a qualified ophthalmologist for examination "
            "and treatment decisions."
        )

st.divider()

st.caption(
    "Research prototype | APTOS 2019 dataset | "
    "EfficientNetB0 + EfficientNetB3 Ensemble | 300 × 300"
)

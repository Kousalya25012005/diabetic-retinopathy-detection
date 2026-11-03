
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="centered"
)

MODEL_PATH = "dr_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

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

    st.image(image, caption="Uploaded Retinal Image",
             use_container_width=True)

    if st.button("Analyze Retinal Image"):

        with st.spinner("Analyzing image..."):

            processed_image = image.resize((224, 224))
            image_array = np.array(
                processed_image, dtype=np.float32
            )
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(
                image_array, verbose=0
            )[0]

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
    "EfficientNetB0"
)

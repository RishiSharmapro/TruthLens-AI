import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: 700;
            text-align: center;
        }
        .subtitle {
            text-align: center;
            color: gray;
            font-size: 16px;
            margin-bottom: 20px;
        }
        .card {
            padding: 20px;
            border-radius: 12px;
            background-color: #0e1117;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">🔍 TruthLens AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detecting truth in a world of synthetic reality</div>', unsafe_allow_html=True)

st.divider()

def detect_face(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None
    
    return image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_model.h5")

model = load_model()


st.subheader("📤 Upload Image")

file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if file is not None:
    col1, col2 = st.columns(2)

    # Show image
    with col1:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image",)

    # Process + Predict
    with col2:
        with st.spinner("🔍 Analyzing..."):
            face = detect_face(img)

            if face is None:
                st.warning("No face detected ❌")

            else:
                face = face.resize((224, 224))
                face = np.array(face)

                face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                pred = model.predict(face)[0][0]

                confidence = float(pred)

                st.subheader("📊 Result")

                if pred < 0.6:
                    fake_conf = 1 - confidence
                    st.error("🚨 FAKE DETECTED")
                    st.progress(int(fake_conf * 100))
                    st.write(f"Confidence: **{fake_conf:.2f}**")
                else:
                    st.success("✅ REAL IMAGE")
                    st.progress(int(confidence * 100))
                    st.write(f"Confidence: **{confidence:.2f}**")

st.divider()

st.markdown("""
<div style="text-align: center; color: gray; font-size: 13px;">
Built with ❤️ using Deep Learning & Streamlit<br>
TruthLens AI © 2026
</div>
""", unsafe_allow_html=True)


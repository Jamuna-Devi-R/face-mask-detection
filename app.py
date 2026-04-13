import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.title("😷 Face Mask Detection App")
st.write("Upload a face image to detect whether a mask is worn!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Convert to numpy
    img_array = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        st.warning("⚠️ No face detected in image!")
    else:
        st.info(f"👤 {len(faces)} face(s) detected!")
        st.success("✅ Face detection working!")
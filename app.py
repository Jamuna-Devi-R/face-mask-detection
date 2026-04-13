import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.title("😷 Face Mask Detection App")
st.write("Real-time face detection!")

# Webcam input
img_file = st.camera_input("📸 Take a photo!")

if img_file is not None:
    img = Image.open(img_file)
    st.image(img, caption="Captured Image", use_container_width=True)
    
    img_array = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        st.warning("⚠️ No face detected!")
    else:
        st.info(f"👤 {len(faces)} face(s) detected!")
        st.success("✅ Face detection working!")
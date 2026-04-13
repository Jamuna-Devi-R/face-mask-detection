import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
import os

st.title("😷 Face Mask Detection App")
st.write("Upload a face image to detect whether a mask is worn or not!")

@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model("mask_detector.h5")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner("Detecting..."):
        model = load_model()
        prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        st.error("❌ No Mask Detected!")
    else:
        st.success("✅ Mask Detected!")
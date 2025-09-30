
# STREAMLIT APP - Garbage Classification

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# ---------------------------
# 1. Load Best Model
# ---------------------------

# Replace with the name of your saved best model
BEST_MODEL_PATH =r"C:\Users\JAI\Downloads\archive\Garbage classification\Garbage classification\mobilenet_model.h5"
model = load_model(BEST_MODEL_PATH)

# Load class indices from training generator
# Make sure you saved it before or recreate
CLASS_INDICES = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash'
}

# ---------------------------
# 2. Streamlit UI
# ---------------------------

st.set_page_config(page_title="RecycleVision", layout="centered")
st.title("♻️ RecycleVision - Garbage Image Classification")
st.write("Upload an image of garbage and the model will predict its category.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ---------------------------
    # 3. Preprocess & Predict
    # ---------------------------
    def predict(img, model, target_size=(224,224)):
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        top_idx = np.argmax(preds)
        confidence = preds[0][top_idx]
        return CLASS_INDICES[top_idx], confidence, preds[0]

    label, conf, all_preds = predict(img, model)

    # Display results
    st.write(f"**Predicted Class:** {label}")
    st.write(f"**Confidence:** {conf*100:.2f}%")

    # Optional: Show top 3 predictions
    top3_idx = np.argsort(all_preds)[-3:][::-1]
    st.write("**Top-3 Predictions:**")
    for idx in top3_idx:
        st.write(f"{CLASS_INDICES[idx]}: {all_preds[idx]*100:.2f}%")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit page configuration
st.set_page_config(page_title="♻️ RecycleVision", layout="wide")
st.title("♻️ RecycleVision - Garbage Classification App")
st.write("Upload a garbage image and choose a model to classify it. View prediction probabilities as a bar chart.")

# 🔽 Model selector
model_choice = st.selectbox(
    "Select Model",
    [
        "mobilenetv2_waste_classifier.h5",
        "vgg16_waste_classifier.h5",
        "resnet50_waste_classifier.h5"
    ]
)

# 📥 Load chosen model
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(model_choice)


# Class labels
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# File uploader
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))   # Resize to match model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    result_idx = np.argmax(predictions)
    result = class_names[result_idx]
    confidence = predictions[0][result_idx] * 100

    # Show prediction
    st.success(f"🗑 **Predicted Category:** {result.capitalize()} ({confidence:.2f}% confidence)")
    

    # Bar chart of probabilities
    fig, ax = plt.subplots()
    ax.bar(class_names, predictions[0], color="skyblue")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Class")
    ax.set_title("Prediction Probabilities")
    plt.xticks(rotation=30)

    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("🚀 Built with ❤️ using **Streamlit + TensorFlow**")



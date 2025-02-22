import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
from PIL import Image

# Load the model and class indices
model = load_model('bottle_defect_classifier.h5')
with open('class_indices.json') as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

# Function to make predictions on the uploaded image
def predict_image(image):
    img = image.resize((224, 224))  # Resize image to the required size
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    return predicted_class, confidence

# Streamlit page configuration
st.set_page_config(page_title="Bottle Defect Classifier")

# Add title to the app
st.title("Bottle Defect Detection")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image of the bottle:", type=["jpg", "jpeg", "png"])

# If the file is uploaded, display the image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to trigger prediction
    if st.button("Classify Image"):
        predicted_class, confidence = predict_image(image)
        st.subheader("Prediction Result:")
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")
else:
    st.write("Please upload an image of the bottle.")

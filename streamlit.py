import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import os

# Function to load the trained model
@st.cache_data()
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to preprocess uploaded image
def preprocess_image(image):
    try:
        # Resize image to match model input size (e.g., 128x128 pixels)
        image = image.resize((128, 128))
        image = np.asarray(image)
        image = image / 255.0  # Normalize image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        st.error(f"Error preprocessing the image: {e}")
        return None

# Function to get the label from prediction
def get_label(prediction, plant_type, threshold=0.5):
    if plant_type == "Banana":
        labels = ["Boron", "Calcium", "Healthy", "Iron", "Magnesium", "Manganese", "Potassium", "Sulphur", "Zinc"]
    elif plant_type == "Rice":
        labels = ["Nitrogen", "Phosphorus", "Potassium"]
    else:
        labels = []

    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index]

    if confidence < threshold:
        return "Not sure", confidence
    return labels[np.argmax(prediction)], confidence

# Function to display address nutrient deficiency message
def display_address_message(address, nutrient_class):
    st.write(f"At {address}, there is a nutrient deficiency: **{nutrient_class}**")

    # Recommendations or solutions based on nutrient class
    recommendations = {
        "Zinc": "Use zinc sulfate for soil or as a leaf spray",
        "Sulphur": "Add sulfur or sulfur-based fertilizers to the soil",
        "Potassium": "Use potassium chloride or potassium sulfate fertilizers",
        "Magnesium": "Use magnesium sulfate for soil or as a leaf spray",
        "Iron": "Use iron chelates for soil or as a leaf spray",
        "Calcium": "Use lime or calcium nitrate fertilizers",
        "Boron": "Use borax or boric acid for soil",
        "Phosphorus": "Use phosphorus fertilizers such as superphosphate",
        "Nitrogen": "Use urea or ammonium nitrate fertilizers",
        "Healthy": "No action needed, the plant is healthy"
    }

    if nutrient_class in recommendations:
        st.markdown(f"**Recommendations:** {recommendations[nutrient_class]}")

# Main function to run the Streamlit app
def main():
    st.title("**Plant Nutrient Deficiency Detector**")

    st.markdown("""
        <style>
            .sidebar-content {
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for selecting plant type
    st.sidebar.markdown("**Select Plant Type**")
    plant_type = st.sidebar.selectbox("", ["", "Banana", "Rice"])

    # Check if a plant type is selected
    if plant_type == "":
        st.warning("**Please select a plant type.**")
        return

    # Sidebar for uploading image
    uploaded_file = st.sidebar.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])

    # Load model based on plant type selected
    if plant_type == "Banana":
        model_path = "D:/image_classification_model_banana.h5"  # Update with your actual path
    elif plant_type == "Rice":
        model_path = "D:/image_classification_model_rice.h5"  # Update with your actual path

    # Check if model file exists
    if plant_type in ["Banana", "Rice"] and not os.path.isfile(model_path):
        st.error(f"**File not found:** {model_path}")
        return

    # Display image upload and prediction only if both plant type and file uploaded
    if uploaded_file is not None and plant_type in ["Banana", "Rice"]:
        model = load_model(model_path)

        if model is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)

                # Preprocess image
                processed_image = preprocess_image(image)

                if processed_image is not None:
                    # Predict nutrient deficiency
                    prediction = model.predict(processed_image)
                    
                    # Get the label and confidence of the prediction
                    label, confidence = get_label(prediction, plant_type)
                    confidence *= 100
                    st.write(f"Nutrient Deficiency: {label} (Confidence: {confidence:.2f}%)")

                    # Example address (replace with actual input)
                    address = st.text_input("Enter Address", "")

                    # Display nutrient deficiency message for the address
                    if st.button("Check Address"):
                        display_address_message(address, label)

            except UnidentifiedImageError:
                st.error("**Invalid image file. Please upload a valid image.**")
            except Exception as e:
                st.error(f"**Error:** {e}")

if __name__ == "__main__":
    main()

# NOTE : All data path can be change to suit your setup and be sure to have your environment setup with streamlit before running this script
# Import necessary libraries
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np

# Load the trained models
models = {
    'EfficientNetV2': {
        'model': tf.keras.models.load_model(
            "efficientnetv2_animals10_nontrain_tfhub.h5",
            custom_objects={'KerasLayer': hub.KerasLayer, 'F1Score': tfa.metrics.F1Score}
        ),
        'size': (224, 224),  # Input image size for the model
        'description': 'Accuracy: 88.7%      F1-Score:0.8861',  # Model performance description
    },
    'EfficientNetV2 Fine Tuned': {
        'model': tf.keras.models.load_model(
            "efficientnetv2_animals10_tfhub.h5",
            custom_objects={'KerasLayer': hub.KerasLayer, 'F1Score': tfa.metrics.F1Score}
        ),
        'size': (224, 224),  # Input image size for the model
        'description': 'Accuracy: 94.19%      F1-Score:0.9348',  # Model performance description
    }
}

# Set app title
st.title('Animal Classifier')

# Model selection in the sidebar
model_name = st.sidebar.selectbox('Select a model', list(models.keys()))

# Use st.file_uploader to get a file from the user
st.caption("Choose an image for classification here")
file = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')
map_dict = {0: 'butterfly', 1: 'cat', 2: 'chicken', 3: 'cow', 4: 'dog', 5: 'elephant', 6: 'horse', 7: 'sheep',
            8: 'spider', 9: 'squirrel'}

# If a file was uploaded
if file is not None:
    # Open and preprocess the image
    img = Image.open(file)
    img_rgb = img.convert('RGB')
    resized = img_rgb.resize((224, 224))
    img = load_img(file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(img_array)

    # Display the uploaded image
    st.image(preprocessed, caption="Uploaded Image", use_column_width=True)

    # Button to generate prediction
    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        # Generate prediction using the selected model
        prediction = models[model_name]['model'].predict(preprocessed)
        st.title("Predicted Label for the image is {}".format(map_dict[prediction.argmax()]))
        st.subheader("Prediction Probabilities:")
        # Display class probabilities
        for i, prob in enumerate(prediction[0]):
            st.text("Class {}: {:.2f}%".format(map_dict[i], prob * 100))

# Add a section for model performance
with st.expander('Model Performance'):
    # Display the description for the selected model
    st.markdown(models[model_name]['description'])

    # Load the performance images for the selected model
    img1 = Image.open('efficientnontrain.png')
    img2 = Image.open('efficientnontrain2.png')
    img3 = Image.open('efficient.png')
    img4 = Image.open('efficient2.png')

    # Define a dictionary to map models to their corresponding performance images
    model_images = {
        'EfficientNetV2': [(img1, "Performance Curve 1"), (img2, "Confusion Matrix 1")],
        'EfficientNetV2 Fine Tuned': [(img3, "Performance Curve 2"), (img4, "Confusion Matrix 2")],
    }

    # Display the performance images for the selected model with captions
    for img, caption in model_images[model_name]:
        st.image(img, caption=caption, use_column_width='auto')

# NOTE : All data path can be change to suit your setup and be sure to have your environment setup with streamlit before running this script

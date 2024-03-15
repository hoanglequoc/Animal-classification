
#NOTE : All data path can be change to suit your setup and be sure to have your environment setup with streamlit before running this script
# Import necessary libraries
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import streamlit as st
from PIL import Image
import numpy as np

# Define function to make API request and return predictions
models = {
    'Model 14 Layer self train 224': {
        'model': tf.keras.models.load_model(
            "selfCNN_animals10",
            custom_objects={'KerasLayer': hub.KerasLayer, 'F1Score': tfa.metrics.F1Score}
        ),
        'size': (224, 224)  # Input image size for the model
    },
    'Model 14 Layers self train 160': {
        'model': tf.keras.models.load_model(
            "self5_animals10_tfhub.h5",
            custom_objects={'KerasLayer': hub.KerasLayer, 'F1Score': tfa.metrics.F1Score}
        ),
        'size': (160, 160)  # Input image size for the model
    },
    'Model 17 Layers self train 125': {
        'model': tf.keras.models.load_model(
            "selfCNN-19_animals10_tfhub.h5",
            custom_objects={'KerasLayer': hub.KerasLayer, 'F1Score': tfa.metrics.F1Score}
        ),
        'size': (125, 125)  # Input image size for the model
    }
}

# Set app title
st.title('Animal Classifier')

# Define a placeholder for the uploaded file data
image = None

# Define a placeholder for the predictions
predictions = None

# Model selection in the sidebar
model_name = st.sidebar.selectbox('Select a model', list(models.keys()))

# Use st.file_uploader to get a file from the user
st.caption("Choose an image for classification here")
file = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')
map_dict = {0: 'butterfly', 1: 'cat', 2: 'chicken', 3: 'cow', 4: 'dog', 5: 'elephant', 6: 'horse', 7: 'sheep',
            8: 'spider', 9: 'squirrel'}
# If a file was uploaded
if file is not None:
    # Open the image file
    img = Image.open(file)

    # Convert the image to RGB
    img_rgb = img.convert('RGB')

    # Convert the image to a numpy array and preprocess it
    img = load_img(file, target_size=models[model_name]['size'])
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed = img_array / 255.0

    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        prediction = models[model_name]['model'].predict(preprocessed)
        st.title("Predicted Label for the image is {}".format(map_dict[prediction.argmax()]))
        st.subheader("Prediction Probabilities:")
        for i, prob in enumerate(prediction[0]):
            st.text("Class {}: {:.2f}%".format(map_dict[i], prob * 100))

# Define a dictionary to map models to their descriptions
model_descriptions = {
    'Model 14 Layer self train 224': 'Accuracy: 84.44%      F1-Score:0.8313',
    'Model 14 Layers self train 160': 'Accuracy: 86.77%     F1-Score:0.8580',
    'Model 17 Layers self train 125': 'Accuracy: 82.99 %    F1-Score:0.8139'
}

# Add an expander for model performance
performance_expander = st.expander("Model Performance")
with performance_expander:
    # Display the description for the selected model
    st.markdown(model_descriptions[model_name])

    # Load the performance images for the selected model
    img1 = Image.open('84 percnet.png')
    img2 = Image.open('84 percentaf.png')
    img3 = Image.open('86phantam.png')
    img4 = Image.open('86-2.png')
    img5 = Image.open('vgg19.png')
    img6 = Image.open('vgg19-2.png')

    # Define a dictionary to map models to their corresponding images
    model_images = {
        'Model 14 Layer self train 224': [(img1, "Performance Curve 1"), (img2, "Confusion Matrix 1")],
        'Model 14 Layers self train 160': [(img3, "Performance Curve 2"), (img4, "Confusion Matrix 2")],
        'Model 17 Layers self train 125': [(img5, "Performance Curve 3"), (img6, "Confusion Matrix 3")]
    }

    # Display the images for the selected model with captions
    for img, caption in model_images[model_name]:
        st.image(img, caption=caption, use_column_width='auto')
# NOTE : All data path can be change to suit your setup and be sure to have your environment setup with streamlit before running this script
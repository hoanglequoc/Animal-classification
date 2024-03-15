#NOTE : All data path can be change to suit your setup and be sure to have your environment setup with streamlit before running this script
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import random
import seaborn as sns
from tensorflow import keras
import os.path
from sklearn.metrics import confusion_matrix

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Function to seed random number generators for reproducibility
def seed_everything(seed=42):
    # Seed value for TensorFlow
    tf.random.set_seed(seed)

    # Seed value for NumPy
    np.random.seed(seed)

    # Seed value for Python's random library
    random.seed(seed)

    # Make sure that TensorFlow uses a deterministic operation wherever possible
    tf.compat.v1.set_random_seed(seed)

# Call the seed function
seed_everything()

# Set the path to the dataset and the hyperparameter
DATA_PATH = 'C:/Users/Admin/PycharmProjects/Lab/Projecttest/raw-img'
BATCH_SIZE = 16
IMG_SIZE = 224
TARGET_SIZE = (224, 224)
RANDOM_STATE = 42
DROPOUT_RATE = 0.1


# Dictionary to translate Italian animal names to English
translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider"
}

# Dictionary to store class names and counts
animals_dict = {'class': [], 'count': []}

# Iterate through directories in the dataset path
for dir_name in os.listdir(DATA_PATH):
    src = os.path.join(DATA_PATH, dir_name)  # Full path to the source directory
    # If the directory name is in the translation dictionary
    if dir_name in translate.keys():
        dst = os.path.join(DATA_PATH, translate[dir_name])  # Full path to the destination directory
        os.rename(src, dst)  # Rename the directory
        animals_dict['class'].append(translate[dir_name])  # Append translated class name
        animals_dict['count'].append(len(os.listdir(dst)))  # Append count of images in the directory
    else:
        animals_dict['class'].append(dir_name)  # Append original class name
        animals_dict['count'].append(len(os.listdir(src)))  # Append count of images in the directory

# Create a DataFrame from the class dictionary
animals_df = pd.DataFrame(animals_dict)
print(animals_df)

# Plot pie chart to show class distribution
fig, ax = plt.subplots(figsize=(7, 6))
colors = plt.cm.Pastel1(range(len(animals_df)))
wedges, texts, autotexts = ax.pie(animals_df['count'],
                                  labels=animals_df['class'],
                                  autopct='%1.1f%%', colors=colors,
                                  textprops=dict(color="black"))

# Draw a white circle at the center for aesthetics
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Set equal aspect ratio for the pie chart
ax.axis('equal')

# Set title for the pie chart
plt.title('Class distribution in the dataset', size=15)

# Set edge color and width for wedges
for w in wedges:
    w.set_edgecolor('black')
    w.set_linewidth(2)

plt.show()

# Split dataset into training and validation sets
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    directory=DATA_PATH,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
    validation_split=0.3,
    subset='both',
    seed=RANDOM_STATE
)

# Function to inverse transform one-hot encoded labels to class names
def inverse_transform(labels: tf.Tensor):
    prediction = pd.Series(np.argmax(labels, axis=1)).map(dict(zip(range(10), train_ds.class_names)))
    return prediction

# Dictionary to store class weights for imbalance
class_dirs = os.listdir(DATA_PATH)
class_weights = {}
total_samples = sum([len(os.listdir(os.path.join(DATA_PATH, dir_label))) for dir_label in class_dirs])

# Calculate class weights
for idx, label in enumerate(class_dirs):
    class_weights[idx] = total_samples / (2 * len(os.listdir(os.path.join(DATA_PATH, label))))

# Number of classes
n_classes = len(os.listdir(DATA_PATH))

# Image preprocessing and augmentation layers
preprocessing_layer = keras.Sequential([
    keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
    keras.layers.Rescaling(1/255)
], name='preprocessing')

augmentation_layer = keras.Sequential([
    keras.layers.RandomFlip(mode='horizontal_and_vertical'),
    keras.layers.RandomRotation(0.2)
], name='augmentation')

# Load EfficientNet V2 model from TensorFlow Hub
model_url = 'https://www.kaggle.com/models/google/efficientnet-v2/frameworks/TensorFlow2/variations/imagenet1k-s-classification/versions/2'
efficientnet_v2_s = hub.KerasLayer(model_url, trainable=True, name='efficientnet_v2_s')

# Build the model architecture
model = keras.Sequential([
    keras.layers.InputLayer((None, None, 3)),
    preprocessing_layer,
    augmentation_layer,
    efficientnet_v2_s,
    keras.layers.Dense(n_classes, activation='softmax', name='output')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[
        keras.metrics.CategoricalAccuracy(),
        tfa.metrics.F1Score(num_classes=n_classes, average='macro')
    ]
)

# Callback for early stopping
stop_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-3,
    patience=3,
    restore_best_weights=True
)

# Print model summary
model.summary()

# Train the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[stop_callback],
    class_weight=class_weights
)

# Save the trained model
model.save('efficientnetv2_animals10_tfhub.h5')
model.save('efficientnetv2_animals10')
# If yoou are going to train a model without the layers being trainable you should enable this two lines below and
# disable the two lines above
# model.save('efficientnetv2_animals10_nontrain_tfhub.h5')
# model.save('efficientnetv2_animals10_nontrain')

# Plot training history (accuracy and loss)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy evolution')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss evolution')

plt.show()

# Compute the confusion matrix
predictions = model.predict(val_ds)
y_pred = np.argmax(predictions, axis=1)
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_true = np.argmax(y_true, axis=1)

cm = confusion_matrix(y_true, y_pred)

# Get the class names
class_names = list(val_ds.class_names)

# Plot the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()
#NOTE : All data path can be change to suit your setup and be sure to have your environment setup with streamlit before running this script


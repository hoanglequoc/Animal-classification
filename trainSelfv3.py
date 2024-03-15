#NOTE : All data path can be change to suit your setup and be sure to have your environment setup with streamlit before running this script
# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import Model
from sklearn.metrics import confusion_matrix
import os.path
import seaborn as sns
import random

# Suppress TensorFlow GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get the list of all physical GPUs
gpus = tf.config.list_physical_devices('GPU')

# Print the number of GPUs available
print(f"Number of GPUs Available: {len(gpus)}")

# Set the path to your dataset NOTE: IMPORTANT
DATA_PATH = 'C:/Users/Admin/PycharmProjects/Lab/Projecttest/raw-img'
BATCH_SIZE = 64
IMG_SIZE = 125
TARGET_SIZE = (125, 125)
RANDOM_STATE = 42
DROPOUT_RATE = 0.1
EPOCHS = 10

# Function to set seed for reproducibility
def seed_everything(seed=42):
    # Seed value for TensorFlow
    tf.random.set_seed(seed)

    # Seed value for NumPy
    np.random.seed(seed)

    # Seed value for Python's random library
    random.seed(seed)

    # Make sure that TensorFlow uses a deterministic operation wherever possible
    tf.compat.v1.set_random_seed(seed)

# Call seed_everything function to set seed
seed_everything()

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

# Dictionary to store class names and their corresponding counts
animals_dict = {'class': [], 'count': []}

# Loop through directories in the dataset path
for dir_name in os.listdir(DATA_PATH):
    src = os.path.join(DATA_PATH, dir_name)  # Full path to the source directory
    if dir_name in translate.keys():
        dst = os.path.join(DATA_PATH, translate[dir_name])  # Full path to the destination directory
        os.rename(src, dst)  # Rename directory to translate Italian names to English
        animals_dict['class'].append(translate[dir_name])
        animals_dict['count'].append(len(os.listdir(dst)))  # Count number of images in the directory
    else:
        animals_dict['class'].append(dir_name)
        animals_dict['count'].append(len(os.listdir(src)))  # Count number of images in the directory

# Create a DataFrame from the animals_dict
animals_df = pd.DataFrame(animals_dict)
print(animals_df)

# Plot a pie chart showing class distribution in the dataset
fig, ax = plt.subplots(figsize=(7, 6))  # Set the figure size
colors = plt.cm.Pastel1(range(len(animals_df)))  # Set the color palette
wedges, texts, autotexts = ax.pie(animals_df['count'],
                                  labels=animals_df['class'],
                                  autopct='%1.1f%%', colors=colors,
                                  textprops=dict(color="black"))

# Draw a white circle at the center (for aesthetics)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')

# Set the title
plt.title('Class distribution in the dataset', size=15)

# Set the edge color to black and width to 2
for w in wedges:
    w.set_edgecolor('black')
    w.set_linewidth(2)

plt.show()

# Data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1 / 255.,  # Rescale pixel values to the range [0, 1]

    zoom_range=0.2,  # Range for random zoom. Range [1-zoom_range, 1+zoom_range].

    rotation_range=20,  # Range for random rotation of the image.
                        # Images will be randomly rotated within the range [-rotation_range, rotation_range] degrees.

    width_shift_range=0.2,  # Range for random horizontal shift.
                            # Images will be randomly shifted horizontally within the range [-width_shift_range * width, width_shift_range * width].

    height_shift_range=0.2,  # Range for random vertical shift.
                             # Images will be randomly shifted vertically within the range [-height_shift_range * height, height_shift_range * height].

    shear_range=0.2,  # Range for random shearing transformation.
                      # Shear intensity (shear angle) in radians in the range [-shear_range, shear_range].

    horizontal_flip=True,  # Randomly flip inputs horizontally.

    validation_split=0.2,  # Fraction of images reserved for validation (float between 0 and 1).
                           # This splits the dataset into training and validation sets.

    fill_mode='nearest'  # Strategy for filling in newly created pixels that may appear after a transformation
                         # 'nearest' fills newly created pixels using the nearest pixel value.
)


train_generator = datagen.flow_from_directory(DATA_PATH,
                                              target_size=(IMG_SIZE, IMG_SIZE),
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              subset='training',
                                              class_mode='categorical')

val_generator = datagen.flow_from_directory(DATA_PATH,
                                            target_size=(IMG_SIZE, IMG_SIZE),
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            subset='validation',
                                            class_mode='categorical')

class_dirs = os.listdir(DATA_PATH)
print(class_dirs)
class_weights = {}
total_samples = sum([len(os.listdir(os.path.join(DATA_PATH, dir_label))) for dir_label in class_dirs])

for idx, label in enumerate(class_dirs):
    class_weights[idx] = total_samples / (2 * len(os.listdir(os.path.join(DATA_PATH, label))))
print(class_weights)

n_classes = len(os.listdir(DATA_PATH))

# Define the CNN architecture
self_CNN = Sequential([
    # Block 1
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(125, 125, 3)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),

    # Block 2
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),

    # Block 3
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),

    # Block 4
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),

    # Block 5
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
])

# Global average pooling layer
last_output = self_CNN.output
x = GlobalAveragePooling2D()(last_output)

# Dense layers (You can change this layer depending on your dataset)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layer
outputs = Dense(10, activation='softmax')(x)

# Define the model
model = Model(inputs=self_CNN.inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=[
                  keras.metrics.CategoricalAccuracy(),
                  tfa.metrics.F1Score(num_classes=n_classes, average='macro')
              ]
              )

# Print model summary
model.summary()

# Define the EarlyStopping callback(You can change the patience)
stop_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-6,
    patience=6,
    restore_best_weights=True
)

# Define the number of epochs
epochs = 80

# Define the ReduceLROnPlateau callback (not necessary but you could add it in)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.000001, verbose=1, min_delta=1e-4)

# Add ReduceLROnPlateau to the callbacks
callbacks = [stop_callback]

# Fit the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    class_weight=class_weights
)

# Plot the training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss evolution')

plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy evolution')

plt.show()

# Compute the confusion matrix
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred)

# Get the class names
class_names = list(val_generator.class_indices.keys())

# Plot the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Save the trained model
model.save('selfCNN-19_animals10')
model.save('selfCNN-19_animals10_tfhub.h5')
#NOTE : All data path can be change to suit your setup and be sure to have your environment setup with streamlit before running this script
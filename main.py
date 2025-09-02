# Import necessary libraries from TensorFlow
import tensorflow as tf
from tensorflow.keras import models, layers
import os

# Optional: Specify which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# --- 1. Define Constants ---
# Grouping parameters here makes them easy to change later.
DATASET_PATH = 'mri_images/train'
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 15
NUM_CLASSES = 4 # Non-Demented, Very Mild, Mild, Moderate

# --- 2. Prepare the Data using Keras Utility ---
# We use the modern `image_dataset_from_directory` utility, which is more
# efficient and integrated with tf.data than the older ImageDataGenerator.
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,  # Seed for reproducibility
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123, # Must use the same seed as the training set
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# --- 3. Configure Dataset for Performance ---
# Use caching and prefetching to optimize the data pipeline.
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. Build the Convolutional Neural Network (CNN) Model ---
# We use a simple Sequential model, stacking layers one after another.
model = models.Sequential([
    # Input Layer specifying the shape of the input data.
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Normalization Layer: Rescale pixel values from [0, 255] to [0, 1].
    # This is now done inside the model instead of in the data generator.
    layers.Rescaling(1./255),

    # First Convolutional Block
    # Conv2D extracts features (like edges). MaxPooling reduces the image size.
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Second Convolutional Block
    # We increase the number of filters to learn more complex features.
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Prepare for Classification
    # Flatten converts the 2D feature maps into a 1D vector.
    layers.Flatten(),

    # Fully Connected (Dense) Layer
    # A standard layer for learning patterns from the flattened features.
    layers.Dense(128, activation='relu'),
    
    # Output Layer
    # 'softmax' activation outputs a probability for each of the 4 classes.
    layers.Dense(NUM_CLASSES, activation='softmax') 
])

# --- 5. Compile the Model ---
# Configure the model's learning process.
model.compile(
    optimizer='adam',                       # Efficient optimization algorithm
    loss='categorical_crossentropy',        # Loss function for multi-class classification
    metrics=['accuracy']                    # Metric to monitor during training
)

# Display the model's architecture
model.summary()

# --- 6. Train the Model ---
# The 'fit' function trains the model on the new tf.data.Dataset objects.
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=NUM_EPOCHS
)

# --- 7. Save the Trained Model ---
# Save the model to a single file for future use in predictions.
model.save('alzheimers_cnn_model.keras')

print("\nModel training complete and saved as alzheimers_cnn_model.keras")


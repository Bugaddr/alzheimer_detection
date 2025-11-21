import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf  # The primary machine learning framework. It handles the core operations.
from tensorflow.keras import models, layers, callbacks  # Keras is TensorFlow's user-friendly API for building and training models.
import numpy as np  # A library for efficient numerical operations, especially with arrays.
from sklearn.utils.class_weight import compute_class_weight # A tool to calculate weights for our imbalanced data.

# --- 1. Configuration ---
DATASET_PATH = 'mri_images/train' # The path to the folder containing our image data.
IMG_SIZE = 128                    # We will resize all images to 128x128 pixels for consistency.
BATCH_SIZE = 32                   # The model will learn by looking at images in groups (batches) of 32.
EPOCHS = 100                      # An epoch is one full pass through the entire training dataset.
NUM_CLASSES = 4                   # The number of categories (e.g., No Impairment, Mild, etc.).
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1) # Defines the shape: 128x128 pixels, 1 color channel (grayscale).

# --- 2. Data Loading ---
def create_dataset(subset):
    return tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,      # Reserve 20% of the data for validation (testing).
        subset=subset,             # Specify whether we are loading the 'training' or 'validation' part.
        seed=123,                  # A seed ensures the random split is the same every time we run the code.
        image_size=(IMG_SIZE, IMG_SIZE), # Resize images as defined in our configuration.
        batch_size=BATCH_SIZE,     # Group images into batches.
        label_mode='categorical',  # Labels are converted to a format suitable for multi-class classification.
        color_mode='grayscale'     # Convert all images to grayscale (1 channel).
    )

# Use the function to create our training and validation datasets.
train_dataset = create_dataset("training")
validation_dataset = create_dataset("validation")
print("> Datasets created successfully.")

# --- 3. Class Imbalance and Performance ---
print("> Calculating class weights from file structure...")
y_train_indices = []
class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])

for i, class_name in enumerate(class_names):
    class_dir = os.path.join(DATASET_PATH, class_name)
    num_files = len(os.listdir(class_dir))
    y_train_indices.extend([i] * num_files)

# We use scikit-learn to calculate the exact weight needed to balance the classes.
class_weights = dict(enumerate(compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_indices),
    y=np.array(y_train_indices)
)))
print(f"> Class Weights Calculated: {class_weights}")

# Part B: Optimize for Performance
AUTOTUNE = tf.data.AUTOTUNE # Find the best settings automatically.
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE) # .cache() keeps data in memory.
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE) # .prefetch() prepares the next batch while GPU is working.
print("> Data pipeline optimized for performance.")

# --- 4. Build and Compile Model ---
model = models.Sequential([
    # The model is a sequence of layers, stacked one after the other.
    layers.Input(shape=INPUT_SHAPE),               # Defines the expected shape of our input images.
    layers.Rescaling(1./255),                      # Rescales pixel values from [0, 255] to [0, 1] for better performance.

    # --- Feature Extraction Blocks ---
    # These layers are responsible for learning to "see" patterns like edges, textures, and shapes.
    layers.Conv2D(64, 3, padding="same", activation='relu'), # 64 filters, 3x3 kernel size. 'relu' adds non-linearity.
    layers.MaxPooling2D(),                                   # Downsamples the image, reducing size but keeping features.

    layers.Conv2D(32, 3, padding="same", activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.20),                                    # Dropout randomly deactivates 20% of neurons to prevent overfitting.

    layers.Conv2D(32, 2, padding="same", activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    # --- Classifier Head ---
    # These layers take the learned features and make the final classification decision.
    layers.Flatten(),                                        # Flattens the 2D feature maps into a 1D vector.
    layers.Dense(100, activation='relu'),                    # A fully connected layer for complex pattern recognition.
    layers.Dense(50, activation='relu'),                     # Another dense layer to further refine the patterns.
    layers.Dense(NUM_CLASSES, activation='softmax')          # The final output layer. 'softmax' converts output to probabilities.
])

# Compile the model
model.compile(optimizer='adam',                   # 'adam' is an adaptive learning rate optimization algorithm.
              loss='categorical_crossentropy',    # Standard loss function for multi-class classification.
              metrics=['accuracy'])               # We want to monitor accuracy during training.

model.summary() # Print a summary of the model architecture.

# --- 5. Train the Model with Callbacks ---
training_callbacks = [
    # This callback stops training if validation accuracy doesn't improve for 20 epochs.
    # It prevents wasting time and restores the best model found.
    callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max'),

    # This callback reduces the learning rate if the validation loss plateaus (stops improving).
    # It helps the model settle into a better minimum.
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001)
]

print("\n> Starting model training...")
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,      # Apply the weights we calculated to handle imbalance.
    callbacks=training_callbacks     # Use our smart callbacks.
)

# --- 6. Save Final Model ---
model.save('trained_model.keras')
print("\n> Model training complete and saved as trained_model.keras")

# --- Step 0: Import Necessary Libraries ---
# We start by importing the tools we'll need for our project.

import tensorflow as tf  # The primary machine learning framework. It handles the core operations.
from tensorflow.keras import models, layers, callbacks  # Keras is TensorFlow's user-friendly API for building and training models.
import numpy as np  # A library for efficient numerical operations, especially with arrays (which is how images are represented).
from sklearn.utils.class_weight import compute_class_weight # A specific tool from scikit-learn to help us handle unbalanced datasets.

# --- 1. Configuration ---
# In this section, we define all the key settings for our project in one place.
# This makes it easy to experiment and change parameters later.

DATASET_PATH = 'mri_images/train' # The path to the folder containing our image data.
IMG_SIZE = 128                    # We will resize all images to 128x128 pixels for consistency.
BATCH_SIZE = 32                   # The model will learn by looking at images in groups (batches) of 32 at a time.
EPOCHS = 100                      # An epoch is one full pass through the entire training dataset. We set a maximum of 100 passes.
NUM_CLASSES = 4                   # The number of categories we are classifying (e.g., No Impairment, Mild, etc.).
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1) # Defines the shape of a single input to our model: 128x128 pixels, with 1 color channel (grayscale).

# --- 2. Data Loading ---
# Here, we load our image data from the folders and prepare it for training.

# We create a helper function to avoid repeating code. This function loads images from a directory.
def create_dataset(subset):
    """Loads a dataset from the specified path and splits it."""
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
print("✅ Datasets created successfully.")

# --- 3. Class Imbalance and Performance ---
# This section addresses two important practical steps: making training fair and making it fast.

# Part A: Handle Class Imbalance
# If we have many more images of one class than others, the model might get biased.
# We calculate 'class weights' to tell the model to pay more attention to the rarer classes.
train_labels = np.concatenate([y for _, y in train_dataset], axis=0) # Get all labels from the training set.
train_label_indices = np.argmax(train_labels, axis=1) # Convert labels to simple indices (0, 1, 2, 3).

class_weights = dict(enumerate(compute_class_weight(
    class_weight='balanced',             # This mode automatically adjusts weights inversely proportional to class frequencies.
    classes=np.unique(train_label_indices), # The unique classes present in our data.
    y=train_label_indices                # The labels themselves.
)))
print(f"⚖️ Calculated Class Weights to handle imbalance: {class_weights}")

# Part B: Optimize for Performance
# These lines help the data pipeline run faster, preventing bottlenecks during training.
AUTOTUNE = tf.data.AUTOTUNE # A special value telling TensorFlow to find the best settings automatically.
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE) # .cache() keeps data in memory, .prefetch() prepares the next batch in advance.
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
print("🚀 Data pipeline optimized for performance.")

# --- 4. Build and Compile Model ---
# This is where we design the architecture of our AI "brain" (the Convolutional Neural Network or CNN).

model = models.Sequential([
    # The model is a sequence of layers, stacked one after the other.
    layers.Input(shape=INPUT_SHAPE),               # Defines the expected shape of our input images.
    layers.Rescaling(1./255),                      # Rescales pixel values from [0, 255] to [0, 1] for better performance.

    # --- Feature Extraction Blocks ---
    # These layers are responsible for learning to "see" patterns like edges, textures, and shapes.
    layers.Conv2D(64, 3, padding="same", activation='relu'), # 64 filters, 3x3 kernel size. 'relu' is a standard activation function.
    layers.MaxPooling2D(),                                   # Downsamples the image, keeping the most important features.

    layers.Conv2D(32, 3, padding="same", activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.20),                                    # Dropout randomly deactivates 20% of neurons to prevent "overfitting" (memorizing).

    layers.Conv2D(32, 2, padding="same", activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    # --- Classifier Head ---
    # These layers take the learned features and make the final classification decision.
    layers.Flatten(),                                       # Flattens the 2D feature map into a 1D vector.
    layers.Dense(100, activation='relu'),                   # A fully connected layer with 100 neurons for complex pattern recognition.
    layers.Dense(50, activation='relu'),                    # Another dense layer to further refine the patterns.
    layers.Dense(NUM_CLASSES, activation='softmax')         # The final output layer. 'softmax' gives a probability for each of the 4 classes.
])

# After building the model, we compile it, setting up the learning process.
model.compile(optimizer='adam',                   # 'adam' is a popular and effective optimization algorithm.
              loss='categorical_crossentropy',    # This loss function is standard for multi-class classification.
              metrics=['accuracy'])               # We want to monitor the accuracy during training.

model.summary() # Print a summary of the model architecture.

# --- 5. Train the Model with Callbacks ---
# Now we start the training process, using "callbacks" to make it smarter.
# Callbacks are tools that monitor training and can take actions automatically.

training_callbacks = [
    # This callback stops training if the validation accuracy doesn't improve for 20 epochs.
    # It also restores the model weights from the best epoch it saw.
    callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max'),

    # This callback reduces the learning rate if the validation loss plateaus (stops improving).
    # This helps the model fine-tune its learning in later stages.
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001)
]

print("\n🧠 Starting model training...")
# The .fit() function starts the actual training loop.
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,      # We apply the class weights we calculated earlier.
    callbacks=training_callbacks     # We use our smart callbacks to manage the process.
)

# --- 6. Save Final Model ---
# After training is complete, we save our trained model to a file.
# This allows us to use it later for making predictions without having to retrain.
model.save('trained_model.keras')
print("\n✅ Model training complete and best version saved as trained_model.keras")


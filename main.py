import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# --- 1. Setup Data Paths and Parameters ---
# Replace this with the path to your extracted dataset folder
dataset_path = 'mri_images/train'

IMG_WIDTH, IMG_HEIGHT = 128, 128
BATCH_SIZE = 32

# --- 2. Create Image Data Generators ---
# This will automatically load images from the folders, resize them, 
# and prepare them for the model. It also splits the data into training (80%)
# and validation (20%) sets.

datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to be between 0 and 1
    validation_split=0.2  # Use 20% of the data for validation
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # For multi-class classification
    subset='training' # Specify this is the training set
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' # Specify this is the validation set
)

# --- 3. Build the CNN Model ---
# This is a simple CNN architecture.
model = Sequential([
    # 1st Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D((2, 2)),

    # 2nd Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # 3rd Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Flatten the results to feed into a dense layer
    Flatten(),

    # Dense Layer
    Dense(128, activation='relu'),
    
    # Output Layer
    # The number of units (4) must match the number of classes.
    # 'softmax' is used for multi-class classification.
    Dense(4, activation='softmax') 
])

# --- 4. Compile the Model ---
# This configures the model for training.
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print a summary of the model architecture
model.summary()

# --- 5. Train the Model ---
# This is where the model learns from the data.
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=15 # You can increase the number of epochs for better performance
)

# --- 6. Save the Model ---
# Save the trained model to a file for later use.
model.save('alzheimers_cnn_model.keras')

print("Model training complete and saved as alzheimers_cnn_model.keras")

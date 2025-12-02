# --- Step 0: Import Necessary Libraries ---
# We import the tools needed to run our prediction script.

import tensorflow as tf      # The primary machine learning framework, used to load our trained model.
import numpy as np           # A library for efficient numerical operations.
from tensorflow.keras.preprocessing import image # A Keras utility to help us load and process images.
import os                    # Allows our script to interact with the operating system (e.g., read file paths).
import random                # Used to randomly select test images.
import sys                   # Allows us to exit the script gracefully if there's an error.

# --- 1. Configuration and Setup ---
# Define the necessary file paths and class names in one place.

MODEL_PATH = 'trained_model.keras' # The path to our saved, trained model file.
TEST_DATA_PATH = 'mri_images/test' # The path to the folder containing the test images.

# These class labels MUST be in the same alphabetical order as Keras found them during training.
# This is because the model's output (e.g., index 0, 1, 2, 3) corresponds to this specific order.
CLASS_LABELS = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# --- 2. Initial Checks ---
# Before running, we verify that the necessary model file and data directory exist.

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file not found at '{MODEL_PATH}'")
    sys.exit() # Exit the script if the model isn't found.
if not os.path.exists(TEST_DATA_PATH):
    print(f"❌ Error: Test data directory not found at '{TEST_DATA_PATH}'")
    sys.exit() # Exit the script if the test data folder isn't found.

# --- 3. Load the Trained Model ---
# We load the AI "brain" that we previously trained and saved.
print(f"🧠 Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# --- 4. Prediction Function ---
# This function handles the entire process of predicting a single image.

def predict_single_image(image_path):
    """
    Takes an image file path, processes it to match the model's input
    requirements, and returns the predicted class name as a string.
    """
    # Load the image from the file path. We must specify the same size (128x128)
    # and color mode (grayscale) that we used for training.
    img = image.load_img(image_path, target_size=(128, 128), color_mode="grayscale")

    # Convert the loaded image into a format the model can understand (a numpy array).
    img_array = image.img_to_array(img)

    # The model was trained on "batches" of images. Even for a single prediction,
    # it expects the input to have that batch dimension. We add it here.
    # Shape changes from (128, 128, 1) to (1, 128, 128, 1).
    img_array = np.expand_dims(img_array, axis=0)

    # Use the loaded model to make a prediction on the prepared image.
    # 'verbose=0' keeps the output clean by hiding the progress bar.
    prediction_probabilities = model.predict(img_array, verbose=0)

    # The model outputs probabilities for each class. We find the index of the highest one.
    predicted_class_index = np.argmax(prediction_probabilities)

    # Use the index to look up the corresponding class name from our list.
    predicted_class_label = CLASS_LABELS[predicted_class_index]

    return predicted_class_label

# --- 5. Main Execution Block ---
# This part of the script runs only when the file is executed directly.
# It handles user input, runs the tests, and prints the final statistics.

if __name__ == "__main__":
    # Get a list of all the class subdirectories in the test folder.
    test_subdirectories = [d for d in os.listdir(TEST_DATA_PATH) if os.path.isdir(os.path.join(TEST_DATA_PATH, d))]

    if not test_subdirectories:
        print(f"❌ Error: No class subdirectories found in '{TEST_DATA_PATH}'")
        sys.exit()

    # --- User Input for Number of Tests ---
    while True: # This loop will continue until a valid number is entered.
        try:
            num_tests_str = input("\nEnter the number of random tests to perform: ")
            num_tests = int(num_tests_str)
            if num_tests > 0:
                break # Exit the loop if the number is positive.
            else:
                print("Please enter a number greater than zero.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    # --- Run Prediction Loop ---
    correct_predictions = 0
    for i in range(num_tests):
        print(f"\n--- Test {i + 1}/{num_tests} ---")

        # 1. Randomly select a class folder (this is the ground truth).
        actual_class = random.choice(test_subdirectories)

        # 2. Randomly select an image from within that class folder.
        image_folder_path = os.path.join(TEST_DATA_PATH, actual_class)
        image_files = os.listdir(image_folder_path)

        if not image_files:
            print(f"⚠️ Warning: No images found in '{actual_class}', skipping test.")
            continue # Skip this loop iteration if the folder is empty.

        random_image_name = random.choice(image_files)
        image_path = os.path.join(image_folder_path, random_image_name)

        # 3. Get the model's prediction for the selected image.
        predicted_class = predict_single_image(image_path)

        # 4. Display the results for this test.
        print(f"Image: '{random_image_name}'")
        print(f"      Actual Class: {actual_class}")
        print(f"   Predicted Class: {predicted_class}")

        # 5. Check if the prediction was correct and update our count.
        if predicted_class == actual_class:
            print("✅ Result: CORRECT")
            correct_predictions += 1
        else:
            print("❌ Result: INCORRECT")

    # --- 6. Display Final Statistics ---
    if num_tests > 0:
        accuracy = (correct_predictions / num_tests) * 100
        incorrect_predictions = num_tests - correct_predictions

        print("\n" + "="*40)
        print(" " * 12 + "Final Statistics")
        print("="*40)
        print(f"Total Tests Performed: {num_tests}")
        print(f"  Correct Predictions: {correct_predictions}")
        print(f"Incorrect Predictions: {incorrect_predictions}")
        print(f"             Accuracy: {accuracy:.2f}%")
        print("="*40)

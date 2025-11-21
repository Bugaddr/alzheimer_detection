import os
# Silence TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time

# --- Configuration ---
MODEL_PATH = 'trained_model.keras'
TEST_DATA_PATH = 'mri_images/test'
IMG_SIZE = 128
# IMPORTANT: These must match the training order exactly
CLASS_LABELS = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

def load_and_preprocess_image(path):
    """Helper to load and resize images for the dataset pipeline."""
    image = tf.io.read_file(path)
    # Decode image (auto-detects format), force grayscale (1 channel)
    image = tf.io.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image

def run_comprehensive_validation():
    print("="*50)
    print("   ALZHEIMER DETECTION SYSTEM - FINAL VALIDATION")
    print("="*50)

    # 1. Check Model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        return

    print(f"[INFO] Loading Model: {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[OK] Model Loaded.\n")

    # --- INPUT REQUEST ---
    try:
        user_input = input("Enter number of images to test per class (Press Enter for ALL): ").strip()
        limit = int(user_input) if user_input else 0
    except ValueError:
        limit = 0
        print("[INFO] Invalid input. Defaulting to ALL images.")

    if limit > 0:
        print(f"[INFO] Limiting test to {limit} images per class.\n")
    else:
        print(f"[INFO] Testing ALL available images.\n")
    # ---------------------

    print("[INFO] Starting Class-by-Class Validation...")
    print("-" * 50)

    # Variables to track global stats
    all_true_labels = []
    all_pred_labels = []
    total_images = 0
    total_correct = 0
    start_time = time.time()

    # 2. Loop through each Class Folder manually
    for class_index, class_name in enumerate(CLASS_LABELS):
        folder_path = os.path.join(TEST_DATA_PATH, class_name)

        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder missing: {class_name}")
            continue

        # Get all image files in this folder
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Apply Limit if requested
        if limit > 0:
            files = files[:limit]

        num_files = len(files)
        if num_files == 0:
            print(f"[WARNING] No images in: {class_name}")
            continue

        print(f"   Testing Class: '{class_name}' ({num_files} images)...", end=" ")

        # Build a dynamic dataset from file paths
        file_paths = [os.path.join(folder_path, f) for f in files]
        path_ds = tf.data.Dataset.from_tensor_slices(file_paths)

        # Map the preprocessing function and batch
        test_ds = path_ds.map(load_and_preprocess_image).batch(32)

        # Predict
        predictions = model.predict(test_ds, verbose=0)
        predicted_indices = np.argmax(predictions, axis=1)

        # The true label for EVERY image in this folder is 'class_index'
        true_indices = np.full(shape=(num_files,), fill_value=class_index)

        # Calculate accuracy for this specific class
        correct_count = np.sum(predicted_indices == true_indices)

        # Store for global stats
        all_true_labels.extend(true_indices)
        all_pred_labels.extend(predicted_indices)
        total_images += num_files
        total_correct += correct_count

        # Print Result
        if correct_count == num_files:
            print(f"[OK] 100% Accuracy ({correct_count}/{num_files})")
        else:
            print(f"[FAIL] {num_files - correct_count} Errors ({correct_count}/{num_files})")

    # 3. Final Calculations
    if total_images == 0:
        print("\n[ERROR] No images were tested.")
        return

    end_time = time.time()
    processing_time = end_time - start_time
    global_accuracy = (total_correct / total_images) * 100

    print("-" * 50)
    print("FINAL STATISTICS")
    print("-" * 50)
    print(f"Total Images Tested: {total_images}")
    print(f"Total Correct:       {total_correct}")
    print(f"Global Accuracy:     {global_accuracy:.2f}%")
    print(f"Processing Time:     {processing_time:.2f} seconds")
    print(f"Avg Inference Speed: {(processing_time/total_images)*1000:.2f} ms/image")

    # 4. Generate Scientific Report
    print("\n[INFO] Generating Classification Report...")
    # Use set() to find which classes were actually tested (in case some were empty)
    present_classes = sorted(list(set(all_true_labels)))
    target_names = [CLASS_LABELS[i] for i in present_classes]

    report = classification_report(all_true_labels, all_pred_labels, target_names=target_names)
    print(report)

    # 5. Save Confusion Matrix
    print("[INFO] Saving Confusion Matrix Graph...")
    cm = confusion_matrix(all_true_labels, all_pred_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Accuracy: {global_accuracy:.2f}%)')

    save_path = 'final_confusion_matrix.png'
    plt.savefig(save_path)
    print(f"[OK] Graph saved to '{save_path}'")

if __name__ == "__main__":
    run_comprehensive_validation()

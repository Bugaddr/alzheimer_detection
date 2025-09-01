import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# --- 1. Load the Saved Model ---
# Make sure the model file is in the same directory as this script,
# or provide the full path to it.
model = tf.keras.models.load_model('alzheimers_cnn_model.keras')

# --- 2. Define the Class Labels ---
# These must be in the same order the model was trained on.
# Keras automatically orders them alphabetically.
class_labels = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

# --- 3. Create the Prediction Function ---
def predict_image(image_path):
    """
    This function takes the path to an image, preprocesses it,
    and returns the predicted class.
    """
    # Load the single image
    img = image.load_img(image_path, target_size=(128, 128))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # The model expects a "batch" of images, so we add an extra dimension
    img_array = np.expand_dims(img_array, axis=0)

    # IMPORTANT: Normalize the image data, just like you did for training
    img_array /= 255.0

    # Make the prediction
    prediction = model.predict(img_array)

    # Find the class with the highest probability
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label

# --- 4. Use the Function ---
# Replace this with the path to any new MRI scan image you want to classify.
new_image_path = 'mild.jpg'
prediction_result = predict_image(new_image_path)

print(f"The model predicts this image is: {prediction_result}")

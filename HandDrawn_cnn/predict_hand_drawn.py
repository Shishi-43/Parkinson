import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# --- Load trained model ---
model = load_model("model/saved_callback_model.h5")  # or your fine-tuned MobileNet path
IMG_SIZE = (128, 128)  # must match what you used during training

# --- Load and preprocess an image ---
img_path = "test_images/spiral_example.jpg"  # path to your spiral/wave
img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)

# --- Predict ---
prediction = model.predict(img_array)[0][0]
label = "Parkinson's Detected" if prediction > 0.5 else "Healthy"

print(f"\nPrediction: {label}")
print(f"Confidence: {prediction:.4f}")

# --- Visualize ---
plt.imshow(tf.keras.preprocessing.image.load_img(img_path))
plt.title(f"{label} ({prediction:.2%} confidence)")
plt.axis('off')
plt.show()

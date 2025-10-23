import os
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

# Define important paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")

# Create Flask app
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained CNN model
MODEL_PATH = os.path.join(BASE_DIR, "model", "saved_callback_model.h5")
model = load_model(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("hand_image")
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Prepare image for prediction
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0

            # Make prediction
            pred = model.predict(img_array)[0][0]
            # Confidence threshold logic
            if pred >= 0.7:
                result = f"Parkinson’s Detected — High Confidence ({pred*100:.2f}%)"
            elif 0.3 < pred < 0.7:
                result = f"Uncertain Result — Could be another disorder (Confidence: {pred*100:.2f}%)"
            else:
                result = f"Healthy Control — High Confidence ({(1-pred)*100:.2f}%)"

            # Create URL for displaying image
            image_url = url_for('static', filename=f'uploads/{filename}')

    return render_template("hand_index.html", result=result, image=image_url)

if __name__ == "__main__":
    app.run(debug=True)

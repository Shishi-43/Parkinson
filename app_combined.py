from flask import Flask, render_template, request, redirect, url_for
import os
import joblib
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Local imports
from predict import predict_parkinsons, plot_features

# --- Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)

UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "model", "parkinsons_XGBoost.joblib")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "HandDrawn_cnn", "model", "saved_callback_model.h5")
HISTORY_FILE = os.path.join(BASE_DIR, "history.csv")

# Load models
audio_model = joblib.load(AUDIO_MODEL_PATH)
hand_model = load_model(HAND_MODEL_PATH)

# --- Routes ---
@app.route("/clear_history", methods=["POST"])
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return redirect("/")

@app.route("/", methods=["GET", "POST"])
def index():
    audio_result = audio_chart = hand_result = hand_image = combined_result = None
    audio_proba = hand_proba = None

    try:
        # --- AUDIO UPLOAD OR RECORD ---
        if "audio_file" in request.files and request.files["audio_file"].filename != "":
            audio_file = request.files["audio_file"]
            filename = secure_filename(audio_file.filename)
            audio_path = os.path.join(UPLOAD_FOLDER, filename)
            audio_file.save(audio_path)

            # Directly use audio_path — .webm supported by predict_parkinsons
            proba, message, feats = predict_parkinsons(audio_path)
            audio_proba = proba
            audio_result = f"Estimated probability of Parkinson's: {proba:.2f}% <br> {message}"

            record = pd.DataFrame([{
                "filename": filename,
                "probability": proba,
                "message": message,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            if os.path.exists(HISTORY_FILE):
                record.to_csv(HISTORY_FILE, mode='a', header=False, index=False)
            else:
                record.to_csv(HISTORY_FILE, index=False)

            audio_chart = plot_features(feats, f"static/plots/{filename}_features.png")

        # --- HAND-DRAWN IMAGE UPLOAD ---
        if "hand_image" in request.files and request.files["hand_image"].filename != "":
            file = request.files["hand_image"]
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Prepare and predict
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0
            hand_pred = hand_model.predict(img_array)[0][0]
            hand_proba = hand_pred * 100

            if hand_pred >= 0.7:
                hand_result = f"Parkinson’s Detected — High Confidence ({hand_proba:.2f}%)"
            elif 0.3 < hand_pred < 0.7:
                leaning = "Parkinson’s" if hand_pred >= 0.5 else "Healthy Control"
                hand_result = (
                    f"Uncertain Result — leaning toward {leaning} "
                    f"(Confidence: {hand_proba:.2f}% for Parkinson’s)"
                )
            else:
                hand_result = f"Healthy Control — High Confidence ({100 - hand_proba:.2f}%)"

            hand_image = url_for('static', filename=f'uploads/{filename}')

        # --- COMBINED DECISION ---
        if audio_proba is not None and hand_proba is not None:
            combined_score = (0.6 * audio_proba + 0.4 * hand_proba)
            if combined_score > 50:
                combined_result = f"Combined Decision: Parkinson’s likely ({combined_score:.2f}%)"
            else:
                combined_result = f"Combined Decision: Healthy Control ({100 - combined_score:.2f}%)"

    except Exception as e:
        print("Traceback:", traceback.format_exc())
        audio_result = f"Error processing input: {e}"

    # --- HISTORY TABLE ---
    history = None
    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        history = history.iloc[::-1]

    return render_template(
        "index_combined.html",
        audio_result=audio_result,
        audio_chart=audio_chart,
        hand_result=hand_result,
        hand_image=hand_image,
        combined_result=combined_result,
        history=history
    )

if __name__ == "__main__":
    app.run(debug=True)

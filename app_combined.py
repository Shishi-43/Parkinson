#%% --- Imports ---
from flask import Flask, render_template, request, redirect, url_for
import os
import joblib
import traceback
import pandas as pd
import numpy as np
import subprocess
from datetime import datetime
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Local imports
from predict import predict_parkinsons, plot_features


#%% --- Paths & App Setup ---
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


#%% --- ROUTES ---
@app.route("/clear_history", methods=["POST"])
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return redirect("/")


@app.route("/", methods=["GET", "POST"])
def index():
    audio_result, audio_chart, hand_result, hand_image = None, None, None, None

    # --- AUDIO PREDICTION ---
    if "audio_file" in request.files and request.files["audio_file"].filename != "":
        audio_file = request.files["audio_file"]
        filename = secure_filename(audio_file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(file_path)

        def convert_to_wav(file_path):
            output_path = file_path.rsplit('.', 1)[0] + ".wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", file_path,
                "-ar", "16000", "-ac", "1", output_path
            ], check=True)
            return output_path

        try:
            proba, message, feats = predict_parkinsons(output_path)
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

        except Exception as e:
            print("Traceback:", traceback.format_exc())
            audio_result = f"Error processing audio: {e}"

    # --- HAND-DRAWN IMAGE PREDICTION ---
    elif "hand_image" in request.files and request.files["hand_image"].filename != "":
        file = request.files["hand_image"]
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0

            pred = hand_model.predict(img_array)[0][0]
            if pred >= 0.7:
                hand_result = f"Parkinson’s Detected — High Confidence ({pred*100:.2f}%)"
            elif 0.3 < pred < 0.7:
                leaning = "Parkinson’s" if pred >= 0.5 else "Healthy Control"
                hand_result = (
                    f"Uncertain Result — leaning toward {leaning} "
                    f"(Confidence: {pred*100:.2f}% for Parkinson’s)"
                )
            else:
                hand_result = f"Healthy Control — High Confidence ({(1-pred)*100:.2f}%)"

            hand_image = url_for('static', filename=f'uploads/{filename}')

        except Exception as e:
            print("Traceback:", traceback.format_exc())
            hand_result = f"Error processing image: {e}"

    # --- HISTORY ---
    history_df = None
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE).tail(5)

    return render_template(
        "index_combined.html",
        audio_result=audio_result,
        audio_chart=audio_chart,
        hand_result=hand_result,
        hand_image=hand_image,
        history=history_df
    )


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
import os
import joblib
import traceback
from werkzeug.utils import secure_filename
from predict import predict_parkinsons


app = Flask(__name__)
model = joblib.load('model/parkinsons_rf.joblib')

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "audio_file" not in request.files:
            return "No file uploaded", 400

        audio = request.files["audio_file"]
        filename = secure_filename(audio.filename)
        audio_path = os.path.join(UPLOAD_FOLDER, filename)
        audio.save(audio_path)

        try:
            proba , message = predict_parkinsons(audio_path)
            result = f"Estimated probability of Parkinson's: {proba:.2f}% <br> {message}"
        except Exception as e:
            print('Traceback:', traceback.format_exc())
            result = f"Error processing audio: {e}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

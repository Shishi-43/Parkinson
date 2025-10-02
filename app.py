from flask import Flask, render_template, request, redirect
import os
import joblib
import traceback
import pandas as pd
from werkzeug.utils import secure_filename
from datetime import datetime
from predict import predict_parkinsons, plot_features
 

History_file = "history.csv"

app = Flask(__name__)
model = joblib.load('model/parkinsons_XGBoost.joblib')

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/clear_history", methods=["POST"])
def clear_history():
    if os.path.exists(History_file):
        os.remove(History_file)
    return redirect("/")

@app.route("/", methods=["GET", "POST"])
def index():
    result, chart_path = None, None
    if request.method == "POST":
        if "audio_file" not in request.files:
            return "No file uploaded", 400

        audio = request.files["audio_file"]
        filename = secure_filename(audio.filename)
        audio_path = os.path.join(UPLOAD_FOLDER, filename)
        audio.save(audio_path)

        try:
            proba , message, feats = predict_parkinsons(audio_path)
            result = f"Estimated probability of Parkinson's: {proba:.2f}% <br> {message}"

            record = pd.DataFrame([{
                "filename": filename,
                "probability": proba,
                "message": message,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            if os.path.exists(History_file):
                record.to_csv(History_file, mode='a', header=False, index=False)
            else:
                record.to_csv(History_file, index=False)

            chart_path = plot_features(feats, f"static/plots/{filename}_features.png")   
            
        except Exception as e:
            print('Traceback:', traceback.format_exc())
            result = f"Error processing audio: {e}"

    history_df = None
    if os.path.exists(History_file):
        history_df = pd.read_csv(History_file).tail(5)

    return render_template("index.html", result=result, chart=chart_path, history=history_df)

if __name__ == "__main__":
    app.run(debug=True)

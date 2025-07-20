import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import joblib
import os
import tkinter as tk
from tkinter import filedialog
from pydub import AudioSegment

# %%
# Loads the saved model + column order
bundle = joblib.load('model/parkinsons_rf.joblib')
clf = bundle['model']
COLS = bundle['columns']

#%%

def convert_webm_to_wav(webm_path, wav_path):
    audio = AudioSegment.from_file(webm_path, format='webm')
    audio.export(wav_path, format='wav')

# %%

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    if audio_path.endswith(".webm"):
        wav_path = audio_path.replace(".webm", ".wav")
        convert_webm_to_wav(audio_path, wav_path)
        audio_path = wav_path

    y, sr = librosa.load(audio_path, sr = None)
    # Extract features
    features = {
        'zcr': np.mean(librosa.feature.zero_crossing_rate(y)[0]),
        'rmse': np.mean(librosa.feature.rms(y=y)[0]),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]),
        'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]),
        'pitch_mean': np.mean(librosa.yin(y, fmin=50, fmax=500, sr=sr)),  # estimates pitch
        'pitch_std': np.std(librosa.yin(y, fmin=50, fmax=500, sr=sr)),
    }

    # Add MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(1, 14):
        features[f'mfcc_{i}'] = np.mean(mfccs[i - 1])

    return features


# %%
# Records audio
def record_audio(filename='recordings/sample_voice.wav', duration=5, fs=22050):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print("Recording... Speak now.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, audio, fs)
    print(f"Saved as {filename}")
    return filename


# %%
# Or upload an existing audio file 
def upload_audio():
    root = tk.Tk()
    root.withdraw()  # Hide GUI window
    file_path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[("WAV files", "*.wav")]
    )
    return file_path




# %%
# Predict from audio

def features_to_ordered_array(feat_dict):
    ordered = [feat_dict[c] for c in COLS]
    return np.array(ordered).reshape(1, -1)


def predict_parkinsons(audio_file):
    raw_feats = extract_features(audio_file)           # Extract features from audio
    X = features_to_ordered_array(raw_feats)   
    proba = clf.predict_proba(X)[0]                    # [prob_healthy, prob_parkinsons]
    pd_prob = proba[1] * 100                           # Convert to percentage
    
    message = ("Warning: High probability of Parkinson's disease. Please consult a healthcare professional."
               if pd_prob > 50 else
               "Low probability of Parkinson's disease. No immediate concern.")
    return pd_prob, message


# %%
# === 6. Main menu ===
def main():
    print("\nChoose input method:")
    print("1. Record audio")
    print("2. Upload audio file")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        file_path = record_audio()
    elif choice == '2':
        file_path = upload_audio()
        if not file_path:
            print("No file selected.")
            return
    else:
        print("Invalid choice.")
        return

    predict_parkinsons(file_path)

# %%
if __name__ == "__main__":
    main()
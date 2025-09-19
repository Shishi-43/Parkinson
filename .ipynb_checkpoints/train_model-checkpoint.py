
# %%
import numpy as np
import os
import pandas as pd
import joblib
import librosa
import parselmouth
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score



# %%


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

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


    delta_mfccs = librosa.feature.delta(mfccs)
    for i in range(1, 14):
        features[f'delta_mfcc_{i}'] = np.mean(delta_mfccs[i - 1])

    try:
    # --- Voice quality features (jitter, shimmer, HNR) using parselmouth ---
        snd = parselmouth.Sound(audio_path)
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

        # Jitter measures
        features['jitter_local'] = parselmouth.praat.call([snd, point_process], "Get jitter (local)", 0, 0, 75, 500, 1.3)
        features['jitter_rap']   = parselmouth.praat.call([snd, point_process], "Get jitter (rap)", 0, 0, 75, 500, 1.3)

        # Shimmer measures
        features['shimmer_local'] = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 75, 500, 1.3, 1.6)
        features['shimmer_apq3']  = parselmouth.praat.call([snd, point_process], "Get shimmer (apq3)", 0, 0, 75, 500, 1.3, 1.6)

        # Harmonics-to-Noise Ratio (HNR)
        harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features['hnr'] = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    except Exception:
        # If jitter/shimmer can't be computed, fall back to NaN or 0
        features['jitter_local'] = np.nan
        features['shimmer_local'] = np.nan

    return features



# %%
# base directory for the dataset
base_dir = r'C:\Users\asus\.cache\kagglehub\datasets\nutansingh\mdvr-kcl-dataset\versions\1\26_29_09_2017_KCL\26-29_09_2017_KCL'

def get_label_from_path(path):
    if 'HC' in path:
        return 0
    elif 'PD' in path:
        return 1
    else:
        return None

def get_recording_type_from_path(path):
    if 'ReadText' in path:
        return 'ReadText'
    elif 'SpontaneousDialogue' in path:
        return 'Dialogue'
    else:
        return 'Unknown'

# %%
data = []

for recording_type in ['ReadText', 'SpontaneousDialogue']:
    for category in ['HC', 'PD']:
        folder_path = os.path.join(base_dir, recording_type, category)
        print(f"Processing {folder_path}...")
        
        for file in tqdm(os.listdir(folder_path)): 
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                try:
                    features = extract_features(file_path)  
                    features['label'] = get_label_from_path(folder_path)
                    features['recording_type'] = get_recording_type_from_path(folder_path)
                    features['filename'] = file
                    data.append(features)
                except Exception as e:
                    print(f"Error processing {file}: {e}")


df = pd.DataFrame(data)


# %%
# Separate features and labels
COLS = df.drop(columns=['label', 'filename', 'recording_type']).columns.tolist()
x = df.drop(columns=['label', 'filename', 'recording_type'])  # drop non-feature columns
y = df['label']
 


le = LabelEncoder() # Encode labels as integers
y_encoded = le.fit_transform(y)  # 'HC' becomes 0, 'PD' becomes 1
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=30) # 80% train, 20% test


clf = RandomForestClassifier(n_estimators=100, random_state=30) 
clf.fit(x_train, y_train) 


# Predict and evaluate
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save both the model and the training column order
os.makedirs("model", exist_ok=True)
joblib.dump({"model": clf, "columns": COLS}, "model/parkinsons_rf.joblib")
print("✓ Model saved to model/parkinsons_rf.joblib")

# %%

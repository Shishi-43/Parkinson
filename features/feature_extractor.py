import librosa
import numpy as np

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

    return features
# %%
import librosa
import numpy as np 
import parselmouth
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from scipy.signal import detrend
from scipy.stats import entropy
from scipy.spatial.distance import pdist


# %%


def extract_features(audio_path, fmin=60, fmax=400):
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    yt, _ = librosa.effects.trim(y, top_db=30)
    if len(yt) == 0:
        yt = y
    snd = parselmouth.Sound(yt, sr)

    # === Pitch-based features ===
    pitch = snd.to_pitch()
    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]
    mdvp_fo  = np.mean(f0) if len(f0) else 0.0
    mdvp_fhi = np.max(f0) if len(f0) else 0.0
    mdvp_flo = np.min(f0) if len(f0) else 0.0

    # === PointProcess for jitter/shimmer ===
    try:
        pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", fmin, fmax)
        npts = parselmouth.praat.call(pp, "Get number of points")
    except:
        pp, npts = None, 0

    if pp and npts > 1:
        # Jitter
        jitter_local = parselmouth.praat.call(pp, "Get jitter (local)", 0,0,0.0001,0.02,1.3)
        jitter_abs   = parselmouth.praat.call(pp, "Get jitter (local, absolute)", 0,0,0.0001,0.02,1.3)
        jitter_rap   = parselmouth.praat.call(pp, "Get jitter (rap)", 0,0,0.0001,0.02,1.3)
        jitter_ppq   = parselmouth.praat.call(pp, "Get jitter (ppq5)", 0,0,0.0001,0.02,1.3)
        jitter_ddp   = 3 * jitter_rap

        # Shimmer
        shimmer_local = parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0,0,0.0001,0.02,1.3,1.6)
        shimmer_db    = parselmouth.praat.call([snd, pp], "Get shimmer (local_dB)", 0,0,0.0001,0.02,1.3,1.6)
        shimmer_apq3  = parselmouth.praat.call([snd, pp], "Get shimmer (apq3)", 0,0,0.0001,0.02,1.3,1.6)
        shimmer_apq5  = parselmouth.praat.call([snd, pp], "Get shimmer (apq5)", 0,0,0.0001,0.02,1.3,1.6)
        shimmer_apq11 = parselmouth.praat.call([snd, pp], "Get shimmer (apq11)", 0,0,0.0001,0.02,1.3,1.6)
        shimmer_dda   = 3 * shimmer_apq3
    else:
        jitter_local = jitter_abs = jitter_rap = jitter_ppq = jitter_ddp = 0.0
        shimmer_local = shimmer_db = shimmer_apq3 = shimmer_apq5 = shimmer_apq11 = shimmer_dda = 0.0

    # === Harmonicity ===
    try:
        harm = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = parselmouth.praat.call(harm, "Get mean", 0, 0) or 0.0
    except:
        hnr = 0.0
    nhr = 1.0/hnr if hnr > 0 else 0.0

    # === Nonlinear features ===
    # RPDE
    try:
        hist, _ = np.histogram(f0, bins=30, density=True)
        hist = hist[hist > 0]
        rpde = entropy(hist)
    except:
        rpde = 0.0

    # DFA
    try:
        y_int = np.cumsum(y - np.mean(y))
        nvals = np.unique(np.floor(np.logspace(2, np.log10(len(y)//4), 20)).astype(int))
        F = []
        for n in nvals:
            segs = len(y_int)//n
            reshaped = y_int[:segs*n].reshape((segs, n))
            local_trend = detrend(reshaped, axis=1)
            F.append(np.sqrt(np.mean(local_trend**2)))
        coeffs = np.polyfit(np.log(nvals), np.log(F), 1)
        dfa = coeffs[0]
    except:
        dfa = 0.0

    # spread1/spread2
    try:
        logf0 = np.log(f0)
        mu, sigma = np.mean(logf0), np.std(logf0)
        spread1 = np.mean(((logf0-mu)/sigma)**3)
        spread2 = np.mean(((logf0-mu)/sigma)**4)
    except:
        spread1 = spread2 = 0.0

    # D2
    try:
        emb_dim, delay = 10, 2
        N = len(y) - (emb_dim-1)*delay
        if N > 0:
            embedded = np.array([y[i:i+N] for i in range(0, emb_dim*delay, delay)]).T
            dists = pdist(embedded)
            radii = np.logspace(-3, 0, 20)
            C = [np.sum(dists<r)/len(dists) for r in radii]
            coeffs = np.polyfit(np.log(radii), np.log(C), 1)
            d2 = coeffs[0]
        else:
            d2 = 0.0
    except:
        d2 = 0.0

    # PPE
    try:
        hist, _ = np.histogram(f0, bins=30, density=True)
        hist = hist[hist > 0]
        ppe = entropy(hist)
    except:
        ppe = 0.0

    return {
        "MDVP:Fo(Hz)": mdvp_fo,
        "MDVP:Fhi(Hz)": mdvp_fhi,
        "MDVP:Flo(Hz)": mdvp_flo,
        "MDVP:Jitter(%)": jitter_local,
        "MDVP:Jitter(Abs)": jitter_abs,
        "MDVP:RAP": jitter_rap,
        "MDVP:PPQ": jitter_ppq,
        "Jitter:DDP": jitter_ddp,
        "MDVP:Shimmer": shimmer_local,
        "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": shimmer_apq3,
        "Shimmer:APQ5": shimmer_apq5,
        "MDVP:APQ": shimmer_apq11,
        "Shimmer:DDA": shimmer_dda,
        "NHR": nhr,
        "HNR": hnr,
        "RPDE": rpde,
        "DFA": dfa,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe
    }


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


#%%

df = pd.DataFrame(data)
df.to_csv("clinical_features.csv", index=False)
print("✓ Features saved to features.csv")
df.head()
# %%

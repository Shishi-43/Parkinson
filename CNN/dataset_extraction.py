# %%
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# %%
# base directory for the dataset
base_dir = r'C:\Users\asus\.cache\kagglehub\datasets\nutansingh\mdvr-kcl-dataset\versions\1\26_29_09_2017_KCL\26-29_09_2017_KCL'

# Output directory for spectrograms
output_dir = "spectrograms"
os.makedirs(output_dir, exist_ok=True)

# %%
def get_label_from_path(path):
    if 'HC' in path:
        return "HC"
    elif 'PD' in path:
        return "PD"
    else:
        return "Unknown"

def get_recording_type_from_path(path):
    if 'ReadText' in path:
        return 'ReadText'
    elif 'SpontaneousDialogue' in path:
        return 'Dialogue'
    else:
        return 'Unknown'

# %%
def save_spectrogram(audio_path, save_path, offset = 25, duration = 50):
    y, sr = librosa.load(audio_path, sr=None, offset=offset, duration=duration )
    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Plot & save as image
    plt.figure(figsize=(3,3))
    librosa.display.specshow(S_db, sr=sr, cmap="magma")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# %%
for recording_type in ['ReadText', 'SpontaneousDialogue']:
    for category in ['HC', 'PD']:
        folder_path = os.path.join(base_dir, recording_type, category)
        print(f"Processing {folder_path}...")

        # Make output subfolder for category
        save_folder = os.path.join(output_dir, recording_type, category)
        os.makedirs(save_folder, exist_ok=True)

        for file in tqdm(os.listdir(folder_path)):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                try:
                    save_path = os.path.join(save_folder, file.replace(".wav", ".png"))
                    save_spectrogram(file_path, save_path)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

print(f"✓ Spectrograms saved under {output_dir}/")

# %%

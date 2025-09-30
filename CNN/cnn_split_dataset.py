#%%
import os
import shutil
import random

# Source: where your spectrograms are saved
SRC_DIR = "spectrograms"   # adjust if needed
DEST_DIR = "data_splited"


# Split ratios
train_split = 0.7
val_split = 0.2
test_split = 0.1

#%%
def make_dirs():
    for split in ["train", "val", "test"]:
        for label in ["HC", "PD"]:
            os.makedirs(os.path.join(DEST_DIR, split, label), exist_ok=True)

def split_and_copy():
    for recording_type in ["ReadText", "SpontaneousDialogue"]:
        for label in ["HC", "PD"]:
            src_folder = os.path.join(SRC_DIR, recording_type, label)
            files = [f for f in os.listdir(src_folder) if f.endswith(".png")]
            random.shuffle(files)

            n_total = len(files)
            n_train = int(train_split * n_total)
            n_val = int(val_split * n_total)

            train_files = files[:n_train]
            val_files = files[n_train:n_train+n_val]
            test_files = files[n_train+n_val:]

            # Copy to new structure
            for f in train_files:
                shutil.copy(os.path.join(src_folder, f),
                            os.path.join(DEST_DIR, "train", label, f))
            for f in val_files:
                shutil.copy(os.path.join(src_folder, f),
                            os.path.join(DEST_DIR, "val", label, f))
            for f in test_files:
                shutil.copy(os.path.join(src_folder, f),
                            os.path.join(DEST_DIR, "test", label, f))

    print("✓ Dataset split into train/val/test at", DEST_DIR)
#%%
if __name__ == "__main__":
    make_dirs()
    split_and_copy()

# %%

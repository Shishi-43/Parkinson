#%%
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

#%% 

base_dir = r"C:\Users\LAB SYSTEM\Documents\Parkinson's\HandPD dataset\Dataset\Dataset"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

#%%
# Data generator with a 70/30 split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3
)

train_data = datagen.flow_from_directory(
    base_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    base_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False  # Keep validation deterministic for metrics
)
#%%


# CNN architecture


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
#%%

# Callbacks
#callbacks = [
   # EarlyStopping(patience=5, restore_best_weights=True),
   # ModelCheckpoint('model/parkinsons_cnn.h5', save_best_only=True)
#]

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    #callbacks=callbacks
)

#%% 

# Evaluate on validation data
print("\nEvaluating on validation set...")
val_data.reset()
preds = model.predict(val_data)
pred_labels = (preds > 0.5).astype(int)
true_labels = val_data.classes

# Accuracy and detailed metrics
acc = accuracy_score(true_labels, pred_labels)
print(f"\nValidation Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(true_labels, pred_labels, target_names=list(val_data.class_indices.keys())))


# %%

# Plot training & validation accuracy
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# %%
# Plot training & validation loss
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

#%%
# Save model
os.makedirs("model", exist_ok=True)
model.save("model/parkinsons_cnn.h5")
print("\n✓ Model saved successfully at model/parkinsons_cnn.h5")
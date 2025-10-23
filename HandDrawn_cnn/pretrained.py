#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score

#%%
base_dir = r"C:\Users\LAB SYSTEM\Documents\Parkinson's\HandPD dataset\Dataset\Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- Data Augmentation ---
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.3,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    validation_split=0.3
)

train_data = datagen.flow_from_directory(
    base_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    base_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

#%%
# --- Load Pretrained MobileNetV2 ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False   # Freeze base layers for feature extraction

# --- Add custom classification head ---
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# --- Compile ---
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#%%
# --- Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('model/best_mobilenetv2_handpd.h5', save_best_only=True)
]

# --- Train ---
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=callbacks
)

#%%

# Save the model before fine-tuning
model.save("model/mobilenet_feature_extractor.h5")
print("✅ Saved pretrained (frozen) model before fine-tuning.")

#%%
# --- Fine-tuning step ---
# Unfreeze top 20 layers for slight adaptation to your dataset
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

fine_tune_history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks
)

#%%
# --- Evaluation ---
print("\nEvaluating on validation set...")
val_data.reset()
preds = model.predict(val_data)
pred_labels = (preds > 0.5).astype(int)
true_labels = val_data.classes

acc = accuracy_score(true_labels, pred_labels)
print(f"\nValidation Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(true_labels, pred_labels, target_names=list(val_data.class_indices.keys())))

#%%
# --- Plot accuracy ---
plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'] + fine_tune_history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'], label='Val Acc')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# --- Plot loss ---
plt.figure(figsize=(10,4))
plt.plot(history.history['loss'] + fine_tune_history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'] + fine_tune_history.history['val_loss'], label='Val Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#%%
# --- Save final model ---
os.makedirs("model", exist_ok=True)
model.save("model/final_mobilenetv2_handpd.h5")
print("\n✅ Model saved successfully: model/final_mobilenetv2_handpd.h5")

#%%


import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Gradient Boosting models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#%%

# Load dataset
df = pd.read_csv("final_parkinsons_data.csv")


COLS = df.drop(columns=['label', 'D2']).columns.tolist()
X = df[COLS]
y = df['label']

X.columns = [
    col.replace(":", "_") 
    .replace("(", "") 
    .replace(")", "") 
    .replace("%", "pct") 
    for col in X.columns]

#%%
# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#%%
# Define models

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        random_state=42, eval_metric="logloss", use_label_encoder=False
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=-1,
        class_weight="balanced", random_state=42
    ),
    "CatBoost": CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        class_weights=[len(y)/sum(y==0), len(y)/sum(y==1)],
        random_state=42, verbose=0
    ),
    "SVM": SVC(
        kernel="rbf", probability=True, class_weight="balanced", random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
}

#%%
# Train, Evaluate, Save

results = {}
conf_matrices = {}
best_model = None
best_acc = 0

for name, model in models.items():
    print(f"\n Training {name} ")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    results[name] = acc
    conf_matrices[name] = cm

    if acc > best_acc:
        best_acc = acc
        best_model = (name, model)

#%%
# Save best model

if best_model:
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model[1], f"model/parkinsons_{best_model[0]}.joblib")
    print(f"\n✓ Best model saved: {best_model[0]} with accuracy {best_acc:.4f}")

#%%
# Visualization


# --- Accuracy bar chart ---
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

#%%
# --- Confusion matrices ---
for name, cm in conf_matrices.items():
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Healthy", "Parkinson's"], 
                yticklabels=["Healthy", "Parkinson's"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

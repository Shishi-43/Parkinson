
# %%
import numpy as np
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



df = pd.read_csv("features_cleaned.csv")

df.head()



# %%
# Separate features and labels
COLS = df.drop(columns=['label', 'filename', 'recording_type']).columns.tolist()
x = df.drop(columns=['label', 'filename', 'recording_type'])  # drop non-feature columns
y = df['label']
 
# %%

le = LabelEncoder() # Encode labels as integers
y_encoded = le.fit_transform(y)  # 'HC' becomes 0, 'PD' becomes 1
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=30) # 80% train, 20% test


clf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=5, random_state=30) 
clf.fit(x_train, y_train) 

# %%
y_train_pred = clf.predict(x_train)
print("Training Performance:")
print(classification_report(y_train, y_train_pred))
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))


#%%
# Predict and evaluate
y_pred = clf.predict(x_test)
print("Test Performance:")
print(classification_report(y_test, y_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_pred))

# %%    
print("\n Cross-validation (5-fold)")
cv_scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
print("Fold scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean(), "±", cv_scores.std())


# %%
# Save both the model and the training column order
os.makedirs("model", exist_ok=True)
joblib.dump({"model": clf, "columns": COLS}, "model/parkinsons_rf.joblib")
print("✓ Model saved to model/parkinsons_rf.joblib")

# %%

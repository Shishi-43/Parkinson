#%%
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

#%%
df = pd.read_csv("clinical_features.csv")
COLS = df.drop(columns=['label', 'filename', 'recording_type']).columns.tolist()
X = df[COLS]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=30
)

#%%

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)

# RandomForest

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 6, 10],
    "min_samples_split": [2, 5, 10]
}

rf = RandomForestClassifier(random_state=30)
rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring="accuracy", n_jobs=-1)
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)

print("\n=== RandomForest ===")
print("Best Params:", rf_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

#%%
# SVM (with scaling)


svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True, random_state=30))
])

svm_params = {
    "clf__C": [0.1, 1, 10],
    "clf__gamma": ["scale", 0.01, 0.001]
}

svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=cv, scoring="accuracy", n_jobs=-1)
svm_grid.fit(X_train, y_train)

svm_best = svm_grid.best_estimator_
y_pred_svm = svm_best.predict(X_test)

print("\n=== SVM ===")
print("Best Params:", svm_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

#%%

# XGBoost

xgb_params = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=30)
xgb_grid = GridSearchCV(xgb, xgb_params, cv=cv, scoring="accuracy", n_jobs=-1)
xgb_grid.fit(X_train, y_train)

xgb_best = xgb_grid.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)

print("\n=== XGBoost ===")
print("Best Params:", xgb_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))


#%%
# LightGBM


lgb_params = {
    "n_estimators": [100, 200, 500],
    "max_depth": [-1, 6, 12],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 50, 100],
    "feature_fraction": [0.8, 1.0],
    "bagging_fraction": [0.8, 1.0]
}

lgb = LGBMClassifier(random_state=30)
lgb_grid = GridSearchCV(lgb, lgb_params, cv=cv, scoring="accuracy", n_jobs=-1)
lgb_grid.fit(X_train, y_train)

lgb_best = lgb_grid.best_estimator_
y_pred_lgb = lgb_best.predict(X_test)

print("\n=== LightGBM ===")
print("Best Params:", lgb_grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_lgb))
print(classification_report(y_test, y_pred_lgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lgb))




#%%


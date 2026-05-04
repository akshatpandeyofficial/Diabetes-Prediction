# diabetes_model_pipeline.py
# Requirements: pandas, scikit-learn, joblib, matplotlib
# Run: pip install pandas scikit-learn joblib matplotlib

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# ✅ FIXED FILE PATH using raw string
DATA_PATH = r"C:\Users\AKSHAT\Downloads\Diabetes Prediction Dataset(in).csv"
OUT_MODEL_PATH = "diabetes_model.pkl"
OUT_METRICS_CSV = "diabetes_model_metrics.csv"

# 1) Load
df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)
print(df.head())

# 2) Identify target column (adjust if not correct)
possible_targets = ['Outcome', 'target', 'Diabetic', 'Diabetes', 'Class', 'Diabetes_012', 'diabetes']
target = None
for c in possible_targets:
    if c in df.columns:
        target = c
        break
if target is None:
    target = df.columns[-1]
    print(f"Warning: Auto-chosen target = '{target}'. Please verify.")

print("Target:", target, "value counts:\n", df[target].value_counts())

# 3) Basic cleaning
df = df.drop_duplicates().reset_index(drop=True)

# 4) Separate X/y and column types
y = df[target]
X = df.drop(columns=[target])

numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print("Numeric cols:", numeric_cols)
print("Categorical cols:", cat_cols)

# 5) Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, cat_cols)
], remainder='drop')

# 6) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 7) Models to train
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = []
best_model = None
best_f1 = -1

for name, clf in models.items():
    pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    results.append({'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc})
    print(f"{name}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, roc_auc={roc}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = pipe

# 8) Save best model + metrics
joblib.dump(best_model, OUT_MODEL_PATH)
pd.DataFrame(results).to_csv(OUT_METRICS_CSV, index=False)
print("Saved model to", OUT_MODEL_PATH)
print("Saved metrics to", OUT_METRICS_CSV)

# 9) Optional: show feature importances if tree-based
clf = best_model.named_steps['clf']
if hasattr(clf, "feature_importances_"):
    feat_names = []
    if numeric_cols:
        feat_names += numeric_cols
    if cat_cols:
        ohe = best_model.named_steps['pre'].named_transformers_['cat'].named_steps['onehot']
        feat_names += list(ohe.get_feature_names_out(cat_cols))
    importances = clf.feature_importances_
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    print("Top features:\n", fi.head(15))

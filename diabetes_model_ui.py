import streamlit as st
# diabetes_model_ui.py
# Requirements: pandas, scikit-learn, joblib, matplotlib, streamlit
# Run these commands first:
# pip install pandas scikit-learn joblib matplotlib streamlit

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
import streamlit as st

# ================================
# 1️⃣ TRAINING SECTION
# ================================
DATA_PATH = r"C:\Users\AKSHAT\Downloads\Diabetes Prediction Dataset(in).csv"
OUT_MODEL_PATH = "diabetes_model.pkl"
OUT_METRICS_CSV = "diabetes_model_metrics.csv"

df = pd.read_csv(DATA_PATH)
st.write("✅ Dataset Loaded:", df.shape)
st.write(df.head())

possible_targets = ['Outcome', 'target', 'Diabetic', 'Diabetes', 'Class', 'Diabetes_012', 'diabetes']
target = None
for c in possible_targets:
    if c in df.columns:
        target = c
        break
if target is None:
    target = df.columns[-1]
    st.warning(f"Auto-detected target column: '{target}'")

df = df.drop_duplicates().reset_index(drop=True)

y = df[target]
X = df.drop(columns=[target])

numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

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
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipe

joblib.dump(best_model, OUT_MODEL_PATH)
pd.DataFrame(results).to_csv(OUT_METRICS_CSV, index=False)
st.success("✅ Model trained and saved successfully!")


st.title("🤖 AI Diabetes Prediction Assistant")
st.subheader("Enter patient details below to predict the risk of diabetes")

# Automatically generate inputs based on numeric & categorical columns
user_input = {}
for col in numeric_cols:
    user_input[col] = st.number_input(f"{col}", min_value=0.0, max_value=500.0, value=0.0)

for col in cat_cols:
    unique_vals = list(df[col].dropna().unique())
    user_input[col] = st.selectbox(f"{col}", unique_vals if len(unique_vals) > 0 else ["None"])

if st.button("🔍 Predict Diabetes Risk"):
    input_df = pd.DataFrame([user_input])
    model = joblib.load(OUT_MODEL_PATH)
    prediction = model.predict(input_df)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ The AI predicts this person **may have diabetes**. (Probability: {prob:.2f})")
    else:
        st.success(f"✅ The AI predicts this person is **not diabetic**. (Probability: {prob:.2f})")

    st.markdown("---")
    st.caption("AI model powered by Logistic Regression, Random Forest, and Gradient Boosting.")


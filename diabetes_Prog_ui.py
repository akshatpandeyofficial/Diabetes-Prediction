
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import streamlit as st

# ===============================================
# 🧩 App Title
# ===============================================
st.set_page_config(page_title="🧠 Diabetes Prediction AI", layout="wide")
st.title("🤖 AI-Powered Diabetes Prediction System")
st.write("This application uses Machine Learning to predict the likelihood of diabetes based on patient data.")

# ===============================================
# 📂 Dataset Loading Section
# ===============================================
DATA_PATH = r"C:\Users\AKSHAT\Downloads\Diabetes Prediction Dataset(in).csv"
OUT_MODEL_PATH = "diabetes_model.pkl"
OUT_METRICS_CSV = "diabetes_model_metrics.csv"

st.sidebar.header("⚙️ Options")

data_option = st.sidebar.radio("Choose Data Source:", ["Use Default Dataset", "Upload Custom CSV"])

if data_option == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()
else:
    df = pd.read_csv(DATA_PATH)

st.write("✅ Dataset Loaded:", df.shape)
st.dataframe(df.head())

# ===============================================
# 🎯 Target Column Detection
# ===============================================
possible_targets = ['Outcome', 'target', 'Diabetic', 'Diabetes', 'Class', 'Diabetes_012', 'diabetes']
target = next((c for c in possible_targets if c in df.columns), df.columns[-1])
st.info(f"Target column detected: **{target}**")

# ===============================================
# 🧹 Data Preparation
# ===============================================
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
])

# ===============================================
# 🧠 Model Training Section
# ===============================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
}

if st.sidebar.button("🚀 Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = []
    best_model = None
    best_f1 = -1

    progress = st.progress(0)
    for i, (name, clf) in enumerate(models.items()):
        pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        results.append({'Model': name, 'Accuracy': acc, 'Precision': prec,
                        'Recall': rec, 'F1': f1, 'ROC_AUC': roc})

        if f1 > best_f1:
            best_f1 = f1
            best_model = pipe

        progress.progress((i + 1) / len(models))

    joblib.dump(best_model, OUT_MODEL_PATH)
    pd.DataFrame(results).to_csv(OUT_METRICS_CSV, index=False)
    st.success("✅ Model trained and saved successfully!")
    st.dataframe(pd.DataFrame(results).round(3))

# ===============================================
# 🔮 Prediction Section
# ===============================================
st.header("🧮 Diabetes Prediction")
st.write("Enter the patient details below to get the prediction: ")

if Path(OUT_MODEL_PATH).exists():
    model = joblib.load(OUT_MODEL_PATH)

    input_data = {}
    cols = numeric_cols + cat_cols

    with st.form("prediction_form"):
        for col in cols:
            if col in numeric_cols:
                input_data[col] = st.number_input(f"{col}", value=0.0)
            else:
                input_data[col] = st.text_input(f"{col}")

        submitted = st.form_submit_button("🔍 Predict")

        if submitted:
            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

            st.write("---")
            if pred == 1:
                st.error(f"⚠️ The model predicts a **HIGH RISK** of Diabetes. (Probability: {prob:.2f})")
            else:
                st.success(f"✅ The model predicts **LOW RISK** of Diabetes. (Probability: {prob:.2f})")
else:
    st.warning("Please train the model first before making predictions.")
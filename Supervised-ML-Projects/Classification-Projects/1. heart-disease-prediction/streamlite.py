
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(
        "Could not load the model. Check that "
        "'heart_disease_model.pkl' exists and is valid."
    )
    st.stop()


# -----------------------------
# Application header
# -----------------------------
st.title("❤️ Heart Disease Classification")
st.write(
    "Enter the patient information below to obtain "
    "a prediction from your trained machine learning model."
)

st.info(
    "Educational and research use only. This application "
    "does not provide a medical diagnosis."
)

# -----------------------------
# Patient input form
# Common UCI Cleveland heart dataset features
# -----------------------------
with st.form("prediction_form"):

    st.subheader("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=50
        )

        sex = st.selectbox(
            "Sex (dataset encoding)",
            options=[0, 1],
            format_func=lambda x: {
                0: "0",
                1: "1"
            }[x]
        )

        cp = st.selectbox(
            "Chest pain type (cp)",
            options=[0, 1, 2, 3]
        )

        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=50, max_value=250, value=120
        )

        chol = st.number_input(
            "Serum cholesterol (mg/dl)",
            min_value=50, max_value=700, value=200
        )

    with col2:
        fbs = st.selectbox(
            "Fasting blood sugar > 120 mg/dl (fbs)",
            options=[0, 1]
        )

        restecg = st.selectbox(
            "Resting ECG result (restecg)",
            options=[0, 1, 2]
        )

        thalach = st.number_input(
            "Maximum heart rate achieved",
            min_value=40, max_value=250, value=150
        )

        exang = st.selectbox(
            "Exercise-induced angina (exang)",
            options=[0, 1]
        )

    with col3:
        oldpeak = st.number_input(
            "ST depression (oldpeak)",
            min_value=0.0, max_value=10.0,
            value=1.0, step=0.1
        )

        slope = st.selectbox(
            "Slope of peak exercise ST segment",
            options=[0, 1, 2]
        )

        ca = st.selectbox(
            "Number of major vessels (ca)",
            options=[0, 1, 2, 3, 4]
        )

        thal = st.selectbox(
            "Thalassemia category (thal)",
            options=[0, 1, 2, 3]
        )

    submitted = st.form_submit_button(
        "Predict Heart Disease",
        type="primary",
        use_container_width=True
    )


# -----------------------------
# Make prediction
# -----------------------------
if submitted:

    # IMPORTANT:
    # These column names and their order must match
    # the features used during model training.
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        # Update these labels to match your model's
        # actual target encoding.
        if prediction == 0:
            st.success(
                "Predicted class: 0 — No heart disease "
                "indicated by the model."
            )
        elif prediction == 1:
            st.warning(
                "Predicted class: 1 — Heart disease "
                "indicated by the model."
            )
        else:
            st.write(f"Predicted class: {prediction}")

        # Show probability only when supported.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            st.subheader("Model Output Probabilities")

            probability_df = pd.DataFrame({
                "Class": [str(c) for c in classes],
                "Probability (%)": [
                    round(float(p) * 100, 2)
                    for p in probabilities
                ]
            })

            st.bar_chart(
                probability_df.set_index("Class")
            )

            st.dataframe(
                probability_df,
                hide_index=True,
                use_container_width=True
            )

        st.caption(
            "These are outputs of the trained model, not a "
            "confirmed diagnosis or a clinically validated "
            "estimate of an individual's risk."
        )

    except Exception as e:
        st.error(
            "Prediction failed. Check the model's expected "
            "feature names, column order, input encoding, "
            "and preprocessing pipeline."
        )
        st.exception(e)
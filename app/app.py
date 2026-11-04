
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

from lime.lime_tabular import LimeTabularExplainer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable CKD Prediction",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "random_forest_ckd_model.joblib"
)

model = joblib.load(MODEL_PATH)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURES = [
    "al",
    "dm",
    "hemo",
    "htn",
    "pcv",
    "rbcc",
    "sg"
]

FEATURE_LABELS = {
    "al": "Albumin (al)",
    "dm": "Diabetes Mellitus (dm)",
    "hemo": "Hemoglobin (hemo)",
    "htn": "Hypertension (htn)",
    "pcv": "Packed Cell Volume (pcv)",
    "rbcc": "Red Blood Cell Count (rbcc)",
    "sg": "Specific Gravity (sg)"
}


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "Explainable Chronic Kidney Disease Prediction"
)

st.write(
    """
    This research prototype predicts the likelihood of Chronic Kidney
    Disease (CKD) using a Random Forest model and provides
    patient-level explanations using SHAP and LIME.
    """
)

st.warning(
    """
    Research prototype only. This system is not a medical diagnosis
    and should not replace professional medical evaluation.
    """
)


# ============================================================
# PATIENT INPUT
# ============================================================

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:

    al = st.number_input(
        "Albumin (al)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=1.0
    )

    dm = st.selectbox(
        "Diabetes Mellitus (dm)",
        options=[0, 1],
        format_func=lambda x:
            "No / 0" if x == 0 else "Yes / 1"
    )

    hemo = st.number_input(
        "Hemoglobin (hemo)",
        min_value=3.0,
        max_value=20.0,
        value=12.0,
        step=0.1
    )

    htn = st.selectbox(
        "Hypertension (htn)",
        options=[0, 1],
        format_func=lambda x:
            "No / 0" if x == 0 else "Yes / 1"
    )


with col2:

    pcv = st.number_input(
        "Packed Cell Volume (pcv)",
        min_value=10.0,
        max_value=60.0,
        value=40.0,
        step=1.0
    )

    rbcc = st.number_input(
        "Red Blood Cell Count (rbcc)",
        min_value=2.0,
        max_value=10.0,
        value=4.5,
        step=0.1
    )

    sg = st.number_input(
        "Specific Gravity (sg)",
        min_value=1.000,
        max_value=1.030,
        value=1.020,
        step=0.001,
        format="%.3f"
    )


# ============================================================
# INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame(
    [[
        al,
        dm,
        hemo,
        htn,
        pcv,
        rbcc,
        sg
    ]],
    columns=FEATURES
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict CKD",
    type="primary"
):

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prediction = int(
        model.predict(input_data)[0]
    )

    probabilities = model.predict_proba(
        input_data
    )[0]

    ckd_probability = float(
        probabilities[0]
    )

    not_ckd_probability = float(
        probabilities[1]
    )


    # --------------------------------------------------------
    # Prediction result
    # --------------------------------------------------------

    st.header("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 0:

            st.error(
                f"Prediction: CKD"
            )

        else:

            st.success(
                f"Prediction: Not CKD"
            )


    with result_col2:

        if prediction == 0:

            st.metric(
                "CKD Probability",
                f"{ckd_probability:.2%}"
            )

        else:

            st.metric(
                "Not CKD Probability",
                f"{not_ckd_probability:.2%}"
            )


    # Display both probabilities
    st.subheader("Model Probabilities")

    probability_df = pd.DataFrame({
        "Class": [
            "CKD",
            "Not CKD"
        ],
        "Probability": [
            ckd_probability,
            not_ckd_probability
        ]
    })

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.header("SHAP Explanation")

    st.write(
        f"""
        The SHAP explanation below corresponds to the model's
        predicted class: **{"CKD" if prediction == 0 else "Not CKD"}**.
        """
    )

    try:

        shap_explainer = shap.TreeExplainer(
            model
        )

        shap_result = shap_explainer.shap_values(
            input_data
        )

        shap_array = np.array(
            shap_result
        )

        # ----------------------------------------------------
        # Handle SHAP output for binary classification
        # ----------------------------------------------------

        if isinstance(shap_result, list):

            # Older SHAP format:
            # [class_0_values, class_1_values]

            predicted_shap = np.array(
                shap_result[prediction][0]
            )

        elif shap_array.ndim == 3:

            # New SHAP format:
            # samples × features × classes

            predicted_shap = shap_array[
                0,
                :,
                prediction
            ]

        elif shap_array.ndim == 2:

            predicted_shap = shap_array[
                0
            ]

        else:

            predicted_shap = shap_array


        # ----------------------------------------------------
        # SHAP DataFrame
        # ----------------------------------------------------

        shap_df = pd.DataFrame({
            "Feature": FEATURES,
            "Feature Value": input_data.iloc[0].values,
            "SHAP Value": predicted_shap
        })

        shap_df["Absolute SHAP"] = np.abs(
            shap_df["SHAP Value"]
        )

        shap_df = shap_df.sort_values(
            by="Absolute SHAP",
            ascending=False
        ).reset_index(drop=True)


        # ----------------------------------------------------
        # Display SHAP table
        # ----------------------------------------------------

        display_shap = shap_df.copy()

        display_shap["Feature"] = (
            display_shap["Feature"]
            .map(FEATURE_LABELS)
        )

        st.dataframe(
            display_shap,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # SHAP chart
        # ----------------------------------------------------

        chart_data = shap_df[
            ["Feature", "SHAP Value"]
        ].copy()

        chart_data["Feature"] = (
            chart_data["Feature"]
            .map(FEATURE_LABELS)
        )

        chart_data = chart_data.set_index(
            "Feature"
        )

        st.bar_chart(
            chart_data
        )


    except Exception as e:

        st.error(
            f"SHAP explanation could not be generated: {e}"
        )


    # ========================================================
    # LIME EXPLANATION
    # ========================================================

    st.header("LIME Explanation")

    st.write(
        f"""
        LIME provides a local explanation of the same patient
        prediction using a locally approximated model.
        Predicted class: **{"CKD" if prediction == 0 else "Not CKD"}**
        """
    )

    try:

        # ----------------------------------------------------
        # Load processed training data
        # ----------------------------------------------------

        processed_data_path = os.path.join(
            BASE_DIR,
            "..",
            "data",
            "processed",
            "ckd_processed.csv"
        )

        processed_df = pd.read_csv(
            processed_data_path
        )

        X_training = processed_df[
            FEATURES
        ]

        # ----------------------------------------------------
        # Create LIME explainer
        # ----------------------------------------------------

        lime_explainer = LimeTabularExplainer(
            training_data=X_training.values,
            feature_names=FEATURES,
            class_names=[
                "CKD",
                "Not CKD"
            ],
            mode="classification",
            discretize_continuous=False,
            random_state=42
        )

        # ----------------------------------------------------
        # Explain predicted class
        # ----------------------------------------------------

        lime_explanation = (
            lime_explainer.explain_instance(
                input_data.iloc[0].values,
                model.predict_proba,
                labels=[0, 1],
                num_features=len(FEATURES)
            )
        )

        lime_values = lime_explanation.as_list(
            label=int(prediction)
        )

        lime_df = pd.DataFrame(
            lime_values,
            columns=[
                "Feature",
                "LIME Contribution"
            ]
        )

        lime_df["Absolute Contribution"] = (
            np.abs(
                lime_df["LIME Contribution"]
            )
        )

        lime_df = lime_df.sort_values(
            by="Absolute Contribution",
            ascending=False
        ).reset_index(drop=True)


        # ----------------------------------------------------
        # Display LIME results
        # ----------------------------------------------------

        st.dataframe(
            lime_df,
            use_container_width=True,
            hide_index=True
        )

        lime_chart = lime_df[
            ["Feature", "LIME Contribution"]
        ].set_index(
            "Feature"
        )

        st.bar_chart(
            lime_chart
        )


    except Exception as e:

        st.error(
            f"LIME explanation could not be generated: {e}"
        )


    # ========================================================
    # INTERPRETATION NOTE
    # ========================================================

    st.info(
        """
        SHAP and LIME explain the behavior of the trained machine
        learning model for the selected patient. These explanations
        represent model contributions and should not be interpreted
        as causal medical effects or as a clinical diagnosis.
        """
    )

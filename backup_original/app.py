
import streamlit as st
import pandas as pd

from document_processor import process_document
from rule_engine import run_rule_engine
from ml_model import (
    model,
    model_metrics,
    predict_document,
    extract_ml_features
)
from document_schema import (
    validate_document_schema,
    DOCUMENT_SCHEMAS
)


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Document Verification System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Automated Official Document Verification")
st.subheader("Consistency Review System")

st.write(
    "Upload an official document to extract information "
    "and identify missing fields or inconsistencies."
)


# -----------------------------------
# DOCUMENT CATEGORY SELECTION
# -----------------------------------

st.header("1. Select Document Category")

document_type = st.selectbox(
    "Choose the type of document",
    options=list(DOCUMENT_SCHEMAS.keys())
)


# -----------------------------------
# DOCUMENT UPLOAD
# -----------------------------------

st.header("2. Upload Document")

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    try:

        # -----------------------------------
        # DOCUMENT PROCESSING
        # -----------------------------------

        extracted_text, fields = process_document(
            uploaded_file,
            document_type
        )

        if extracted_text.strip():

            # -----------------------------------
            # EXTRACTED TEXT
            # -----------------------------------

            st.header("3. Extracted Document Text")

            st.text_area(
                "Extracted text",
                extracted_text,
                height=250
            )

            # -----------------------------------
            # EXTRACTED FIELDS
            # -----------------------------------

            st.header("4. Extracted Fields")

            st.json(fields)

            # -----------------------------------
            # SCHEMA VALIDATION
            # -----------------------------------

            st.header("5. Knowledge Representation & Schema Validation")

            schema_results = validate_document_schema(
                fields,
                document_type
            )

            st.dataframe(
                schema_results,
                width="stretch",
                hide_index=True
            )

            if any(
                result["Status"] in ["MISSING", "INVALID FORMAT"]
                for result in schema_results
            ):
                st.warning(
                    "Schema validation found missing or invalid fields."
                )
            else:
                st.success(
                    "Document fields match the defined schema."
                )

            # -----------------------------------
            # PRODUCTION RULE ENGINE
            # -----------------------------------

            st.header("6. Production Rule Engine Results")

            status, triggered_rules = run_rule_engine(
                fields,
                document_type
            )

            if status == "VALID":

                st.success(
                    "VALID - All implemented consistency rules passed."
                )

            elif status == "INVALID":

                st.error(
                    "INVALID - Potential contradictions detected."
                )

            else:

                st.warning(
                    "NEEDS MANUAL REVIEW - Missing or unclear information."
                )

            st.subheader("Triggered Production Rules")

            st.dataframe(
                triggered_rules,
                width="stretch",
                hide_index=True
            )

            st.caption(
                "Passing consistency checks does not prove document authenticity."
            )

            # -----------------------------------
            # MACHINE LEARNING ANALYSIS
            # -----------------------------------

            st.header("7. Machine Learning Analysis")

            ml_features = extract_ml_features(
                fields,
                document_type
            )

            st.subheader("Extracted ML Features")

            st.json(ml_features)

            ml_result = predict_document(
                model,
                ml_features
            )

            st.subheader("ML Classification Result")

            prediction = ml_result["prediction"]

            if prediction == "CONSISTENT":

                st.success("CONSISTENT")

            elif prediction == "INCONSISTENT":

                st.error("INCONSISTENT")

            else:

                st.warning("NEEDS REVIEW")

            st.metric(
                "Model Confidence",
                f'{ml_result["confidence"]}%'
            )

            st.subheader("Class Probabilities")

            st.json(ml_result["probabilities"])

            # -----------------------------------
            # MODEL EVALUATION
            # -----------------------------------

            st.header("8. Machine Learning Model Evaluation")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Accuracy",
                f'{model_metrics["Accuracy"] * 100:.2f}%'
            )

            col2.metric(
                "Precision",
                f'{model_metrics["Precision"] * 100:.2f}%'
            )

            col3.metric(
                "Recall",
                f'{model_metrics["Recall"] * 100:.2f}%'
            )

            col4.metric(
                "F1 Score",
                f'{model_metrics["F1 Score"] * 100:.2f}%'
            )

            # -----------------------------------
            # CONFUSION MATRIX
            # -----------------------------------

            st.subheader("Confusion Matrix")

            confusion_df = pd.DataFrame(
                model_metrics["Confusion Matrix"],
                index=[
                    "Actual Consistent",
                    "Actual Needs Review",
                    "Actual Inconsistent"
                ],
                columns=[
                    "Predicted Consistent",
                    "Predicted Needs Review",
                    "Predicted Inconsistent"
                ]
            )

            st.dataframe(
                confusion_df,
                width="stretch"
            )

            # -----------------------------------
            # FEATURE IMPORTANCE
            # -----------------------------------

            st.subheader("Feature Importance Analysis")

            importance_df = pd.DataFrame({
                "Feature": model.feature_names_in_,
                "Importance": model.feature_importances_
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            st.bar_chart(
                importance_df.set_index("Feature")
            )

            st.dataframe(
                importance_df,
                width="stretch"
            )

            # -----------------------------------
            # CLASSIFICATION REPORT
            # -----------------------------------

            with st.expander("View Detailed Classification Report"):

                st.text(
                    model_metrics["Classification Report"]
                )

            # -----------------------------------
            # DISCLAIMER
            # -----------------------------------

            st.info(
                "This ML model is trained on synthetic data for demonstration. "
                "Its predictions do not establish document authenticity. "
                "Official verification requires checking the issuing authority."
            )

        else:

            st.warning(
                "No readable text found. Manual review required."
            )

    except Exception as error:

        st.error("Document processing failed.")
        st.caption(str(error))



import numpy as np
import pandas as pd

from datetime import datetime, date

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# -----------------------------------
# 1. FEATURE DEFINITIONS
# -----------------------------------

FEATURES = [
    "missing_fields_count",
    "issue_date_invalid",
    "expiry_date_invalid",
    "expiry_before_issue",
    "issue_date_future",
    "document_expired"
]

LABELS = {
    0: "CONSISTENT",
    1: "NEEDS REVIEW",
    2: "INCONSISTENT"
}


# -----------------------------------
# 2. DOCUMENT CATEGORY REQUIREMENTS
# -----------------------------------

REQUIRED_FIELDS = {

    "Educational Certificate": [
        "Student Name",
        "Certificate ID",
        "Institution",
        "Issue Date"
    ],

    "Identity Document": [
        "Name",
        "Document Number",
        "Date of Birth"
    ],

    "Government Certificate": [
        "Applicant Name",
        "Certificate ID",
        "Issuing Authority",
        "Issue Date"
    ]
}


# -----------------------------------
# 3. GENERATE SYNTHETIC DATASET
# -----------------------------------

def generate_dataset(samples=3000):

    rng = np.random.default_rng(42)

    samples_per_class = samples // 3
    data = []

    # CLASS 0: CONSISTENT
    for _ in range(samples_per_class):

        data.append([
            0, 0, 0, 0, 0, 0, 0
        ])

    # CLASS 1: NEEDS REVIEW
    for _ in range(samples_per_class):

        missing = int(rng.integers(0, 4))
        issue_invalid = int(rng.integers(0, 2))
        expiry_invalid = int(rng.integers(0, 2))

        if missing == 0 and issue_invalid == 0 and expiry_invalid == 0:
            missing = 1

        data.append([
            missing,
            issue_invalid,
            expiry_invalid,
            0,
            0,
            0,
            1
        ])

    # CLASS 2: INCONSISTENT
    for _ in range(samples_per_class):

        reason = int(rng.integers(0, 3))

        data.append([
            int(rng.integers(0, 4)),
            0,
            0,
            int(reason == 0),
            int(reason == 1),
            int(reason == 2),
            2
        ])

    columns = FEATURES + ["label"]

    dataset = pd.DataFrame(
        data,
        columns=columns
    )

    dataset = dataset.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return dataset


# -----------------------------------
# 4. TRAIN MACHINE LEARNING MODEL
# -----------------------------------

def train_model():

    dataset = generate_dataset()

    X = dataset[FEATURES]
    y = dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {

        "Accuracy": accuracy_score(
            y_test, predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Confusion Matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1, 2]
        ),

        "Classification Report": classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=[
                "Consistent",
                "Needs Review",
                "Inconsistent"
            ],
            zero_division=0
        )
    }

    return model, metrics, dataset


# -----------------------------------
# 5. PREDICT DOCUMENT CATEGORY
# -----------------------------------

def predict_document(model, features):

    input_data = pd.DataFrame(
        [features],
        columns=FEATURES
    )

    prediction = int(
        model.predict(input_data)[0]
    )

    probabilities = model.predict_proba(input_data)[0]

    confidence = float(max(probabilities)) * 100

    return {

        "prediction": LABELS[prediction],

        "confidence": round(confidence, 2),

        "probabilities": {
            LABELS[int(class_id)]: round(
                float(probability) * 100, 2
            )
            for class_id, probability in zip(
                model.classes_,
                probabilities
            )
        }
    }


# -----------------------------------
# 6. DATE PARSING
# -----------------------------------

def parse_date(value):

    if not value:
        return None

    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%d.%m.%Y"
    ]

    for fmt in formats:

        try:
            return datetime.strptime(
                value.strip(), fmt
            ).date()

        except ValueError:
            continue

    return None


# -----------------------------------
# 7. EXTRACT ML FEATURES
# -----------------------------------

def extract_ml_features(fields, document_type="Educational Certificate"):

    if document_type not in REQUIRED_FIELDS:
        raise ValueError(
            f"Unsupported document type: {document_type}"
        )

    required_fields = REQUIRED_FIELDS[document_type]

    missing_count = sum(
        1 for field in required_fields
        if not fields.get(field)
    )

    issue_value = fields.get("Issue Date")
    expiry_value = fields.get("Expiry Date")

    issue_date = parse_date(issue_value)
    expiry_date = parse_date(expiry_value)

    today = date.today()

    return {

        "missing_fields_count": missing_count,

        "issue_date_invalid": int(
            bool(issue_value) and issue_date is None
        ),

        "expiry_date_invalid": int(
            bool(expiry_value) and expiry_date is None
        ),

        "expiry_before_issue": int(
            issue_date is not None
            and expiry_date is not None
            and expiry_date < issue_date
        ),

        "issue_date_future": int(
            issue_date is not None
            and issue_date > today
        ),

        "document_expired": int(
            expiry_date is not None
            and expiry_date < today
            and not (
                issue_date is not None
                and expiry_date < issue_date
            )
        )
    }


# -----------------------------------
# 8. TRAIN MODEL WHEN MODULE IS LOADED
# -----------------------------------

model, model_metrics, training_dataset = train_model()


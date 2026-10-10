import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


train = pd.read_csv("UNSW_NB15_training-set.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv")

print("Datasets loaded successfully!")
print("Training shape:", train.shape)
print("Testing shape:", test.shape)

input("\nPress Enter to continue...")


# TASK 1: PREPROCESSING AND FEATURE ENGINEERING
print("\nTASK 1: PREPROCESSING AND FEATURE ENGINEERING")

print("\nFirst five training records:")
print(train.head().to_string())

print("\nData types:")
print(train.dtypes.to_string())

print("\nMissing values:")
print(train.isna().sum().to_string())

print("\nDuplicates without id:")
print(train.drop(columns="id").duplicated().sum())

input("\nPress Enter to continue...")


print("\nTarget variable: label")
print("0 = Normal")
print("1 = Attack")

train = train.drop(columns=["id", "attack_cat"]).drop_duplicates().copy()
test = test.drop(columns=["id", "attack_cat"]).drop_duplicates().copy()

train["label"] = train["label"].astype(int)
test["label"] = test["label"].astype(int)


for data in [train, test]:
    data["total_bytes"] = data["sbytes"] + data["dbytes"]
    data["total_packets"] = data["spkts"] + data["dpkts"]

    duration = data["dur"].clip(lower=0.001)
    data["bytes_per_second"] = data["total_bytes"] / duration

print("\nNew cybersecurity features:")
print("1. total_bytes: total transferred data; unusual volume may be informative.")
print("2. total_packets: total packet count; bursts may indicate automated activity.")
print("3. bytes_per_second: transfer speed; high rates may indicate unusual traffic.")
print("These features do not prove that a connection is an attack.")

input("\nPress Enter to continue...")


X_train = train.drop(columns="label")
y_train = train["label"]

X_test = test.drop(columns="label")
y_test = test["label"]

numerical = X_train.select_dtypes(include="number").columns.tolist()
categorical = ["proto", "service", "state"]

print("\nNumerical features:")
print(numerical)

print("\nCategorical features:")
print(categorical)

print("\nCategorical features will be encoded using OneHotEncoder.")
print("Numerical features will be filled with medians and scaled.")
print("All preprocessing parameters will be learned from training data only.")

input("\nPress Enter to continue...")


# TASK 2: BUILD AND COMPARE CLASSIFIERS
print("\nTASK 2: BUILD AND COMPARE CLASSIFIERS")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []
rf_predictions = None

for name, model in models.items():
    print("\nTraining:", name)

    numerical_processing = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_processing = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numerical", numerical_processing, numerical),
        ("categorical", categorical_processing, categorical)
    ])

    classifier = Pipeline([
        ("preprocessing", preprocessing),
        ("model", model)
    ])

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0)
    })

    if name == "Random Forest":
        rf_predictions = predictions

    print(name, "completed.")

comparison = pd.DataFrame(results)

print("\nModel comparison:")
print(comparison.to_string(index=False, float_format="%.4f"))

print("\nAccuracy alone can hide missed attacks.")
print("For example, with 99% normal traffic, predicting every connection")
print("as normal gives 99% accuracy but detects no attacks.")

input("\nPress Enter to continue...")


# TASK 3: CONFUSION MATRIX AND ERROR ANALYSIS
print("\nTASK 3: CONFUSION MATRIX AND ERROR ANALYSIS")

matrix = confusion_matrix(y_test, rf_predictions, labels=[0, 1])
tn, fp, fn, tp = matrix.ravel()

print("\nConfusion matrix:")
print(matrix)

print("\nTN - normal connections correctly identified:", tn)
print("FP - normal connections incorrectly flagged as attacks:", fp)
print("FN - attacks incorrectly classified as normal:", fn)
print("TP - attacks correctly detected:", tp)

fpr = fp / (fp + tn) if fp + tn > 0 else 0
fnr = fn / (fn + tp) if fn + tp > 0 else 0

print("\nFalse Positive Rate:", round(fpr, 4))
print("False Negative Rate:", round(fnr, 4))
print("Precision:", round(
    precision_score(y_test, rf_predictions, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, rf_predictions, zero_division=0), 4
))
print("F1-score:", round(
    f1_score(y_test, rf_predictions, zero_division=0), 4
))

print("\nFalse negatives are especially important because attacks go undetected.")
print("False positives also matter because too many alerts can overwhelm analysts.")

plt.figure(figsize=(6, 5))

sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

print("\nLAB COMPLETED")
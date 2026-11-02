import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc
)


# TASK 1: MULTICLASS MODEL TRAINING AND BASELINE EVALUATION
print("\nTASK 1: MULTICLASS MODEL TRAINING AND BASELINE EVALUATION")
train = pd.read_csv("../UNSW_NB15_training-set.csv")
test = pd.read_csv("../UNSW_NB15_testing-set.csv")
print("Training shape:", train.shape)
print("Testing shape:", test.shape)

classes = [
    "Normal", "Fuzzers", "Analysis", "Backdoor", "DoS",
    "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms"
]

# 1. PREPROCESSING
for data in [train, test]:
    data["attack_cat"] = data["attack_cat"].astype("string").str.strip()
    data.loc[data["label"] == 0, "attack_cat"] = "Normal"
    if not data["attack_cat"].isin(classes).all():
        raise ValueError("Missing or unexpected attack category.")
    data["attack_cat"] = data["attack_cat"].astype(str)
    data.drop(columns=["id", "label"], inplace=True)
    data.drop_duplicates(inplace=True)

# Prevent identical feature rows from appearing in different splits.
features = train.columns.drop("attack_cat")
overlap = pd.MultiIndex.from_frame(train[features]).isin(
    pd.MultiIndex.from_frame(test[features])
)
train = train.loc[~overlap].drop_duplicates(subset=features).copy()

for data in [train, test]:
    data["total_bytes"] = data["sbytes"] + data["dbytes"]
    data["total_packets"] = data["spkts"] + data["dpkts"]
    data["bytes_per_second"] = data["total_bytes"] / data["dur"].clip(lower=0.001)

X_train, X_val, y_train, y_val = train_test_split(
    train.drop(columns="attack_cat"), train["attack_cat"],
    test_size=0.2, stratify=train["attack_cat"], random_state=42
)
X_test, y_test = test.drop(columns="attack_cat"), test["attack_cat"]
print("Training / validation / test:", len(X_train), len(X_val), len(X_test))
input("\nPress Enter to continue...")

# 2. CLASS IMBALANCE AUDIT
print("\n2. CLASS IMBALANCE AUDIT")
counts = y_train.value_counts().reindex(classes, fill_value=0)
distribution = pd.DataFrame({"Support": counts, "Frequency (%)": counts / len(y_train) * 100})
print(distribution.round(4).to_string())
distribution["Frequency (%)"].plot(kind="bar", figsize=(10, 5), color="steelblue")
plt.title("Training Class Distribution")
plt.ylabel("Frequency (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 3. RANDOM FOREST
print("\n3. TRAINING RANDOM FOREST")
numerical = X_train.select_dtypes(include="number").columns.tolist()
categorical = [column for column in X_train if column not in numerical]
preprocessing = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])
classifier = Pipeline([
    ("preprocessing", preprocessing),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# Match probability columns to our class order.
model_classes = list(classifier.named_steps["model"].classes_)
indices = [model_classes.index(name) for name in classes]
probabilities = classifier.predict_proba(X_test)[:, indices]

# 4. CLASSIFICATION METRICS
print("\n4. BASELINE CLASSIFICATION REPORT")
print(classification_report(y_test, predictions, labels=classes, digits=4, zero_division=0))
input("\nPress Enter to continue...")

# 5. NORMALIZED CONFUSION MATRIX
print("\n5. NORMALIZED CONFUSION MATRIX")
matrix = confusion_matrix(y_test, predictions, labels=classes, normalize="true")
plt.figure(figsize=(12, 9))
sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Random Forest Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 6. ONE-VS-REST ROC-AUC
print("\n6. ONE-VS-REST ROC-AUC")
plt.figure(figsize=(10, 7))
for i, name in enumerate(classes):
    binary = (y_test == name).astype(int)
    if binary.nunique() < 2:
        print("ROC-AUC unavailable:", name)
        continue
    fpr, tpr, _ = roc_curve(binary, probabilities[:, i])
    score = auc(fpr, tpr)
    print(f"{name}: {score:.4f}")
    plt.plot(fpr, tpr, label=f"{name} (AUC = {score:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
input("\nPress Enter to continue...")


# TASK 2: PER-CLASS THRESHOLD OPTIMIZATION AND ERROR ANALYSIS
print("\nTASK 2: PER-CLASS THRESHOLD OPTIMIZATION AND ERROR ANALYSIS")
validation_probabilities = classifier.predict_proba(X_val)[:, indices]
thresholds = []
for i, name in enumerate(classes):
    binary = (y_val == name).astype(int)
    if binary.nunique() < 2:
        raise ValueError(f"Validation split cannot calibrate {name}.")
    precision, recall, candidates = precision_recall_curve(binary, validation_probabilities[:, i])
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    positive = candidates > 0
    candidates, scores = candidates[positive], scores[positive]
    best = np.flatnonzero(scores == scores.max())[-1] if len(scores) else None
    thresholds.append(candidates[best] if best is not None else 1.0)

thresholds = np.array(thresholds)
print("\nThresholds maximizing validation OvR F1:")
print(pd.DataFrame({"Class": classes, "Threshold": thresholds}).to_string(index=False))

# Multiple matches: highest probability/threshold ratio. No matches: baseline.
eligible = probabilities >= thresholds
ratios = np.where(eligible, probabilities / thresholds, -np.inf)
tuned_predictions = np.array(classes)[ratios.argmax(axis=1)]
no_match = ~eligible.any(axis=1)
tuned_predictions[no_match] = predictions[no_match]
print("\nRule: highest probability/threshold ratio among eligible classes.")
print("If none qualify, use baseline argmax. Improvement is not guaranteed.")

results = []
recall_comparison = pd.DataFrame({"Class": classes})
for name, predicted in [("Baseline", predictions), ("Tuned", tuned_predictions)]:
    report = classification_report(y_test, predicted, labels=classes, output_dict=True, zero_division=0)
    results.append({
        "Model": name,
        "Accuracy": np.mean(y_test.to_numpy() == predicted),
        "Macro F1": report["macro avg"]["f1-score"],
        "Weighted F1": report["weighted avg"]["f1-score"]
    })
    recall_comparison[name + " Recall"] = [report[c]["recall"] for c in classes]
    if name == "Tuned":
        print("\nTuned classification report:")
        print(classification_report(y_test, predicted, labels=classes, digits=4, zero_division=0))

    matrix = confusion_matrix(y_test, predicted, labels=classes)
    errors = [{"Actual": classes[i], "Predicted": classes[j], "Count": int(matrix[i, j])}
              for i in range(len(classes)) for j in range(len(classes))
              if i != j and matrix[i, j] > 0]
    print(f"\n{name}: top 3 misclassification pairs")
    print(pd.DataFrame(errors).sort_values("Count", ascending=False).head(3).to_string(index=False)
          if errors else "No misclassifications.")

    tn, fp, fn, tp = confusion_matrix(
        y_test != "Normal", predicted != "Normal", labels=[False, True]
    ).ravel()
    print(f"Binary detection: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("FPR:", round(fp / (fp + tn), 4) if fp + tn else 0)
    print("FNR:", round(fn / (fn + tp), 4) if fn + tp else 0)

print("\nTEST SET COMPARISON")
print(pd.DataFrame(results).to_string(index=False, float_format="%.4f"))
print("\nPER-CLASS RECALL COMPARISON")
print(recall_comparison.to_string(index=False, float_format="%.4f"))
input("\nPress Enter to continue...")
print("\nLAB 3 COMPLETED")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)


# LOADING DATA
train = pd.read_csv("UNSW_NB15_training-set.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv")

print("Training shape:", train.shape)
print("Testing shape:", test.shape)


# 1. PREPROCESSING
target = "attack_cat"

for data in [train, test]:
    data[target] = data[target].fillna("Normal").astype(str).str.strip()
    data.loc[data["label"] == 0, target] = "Normal"
    data.drop(columns=["id", "label"], inplace=True)

train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)

X_train = train.drop(columns=target)
y_train = train[target]

X_test = test.drop(columns=target)
y_test = test[target]

numerical = X_train.select_dtypes(include="number").columns
categorical = X_train.select_dtypes(
    include=["object", "category"]
).columns

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


# 2. CLASS IMBALANCE AUDIT
class_order = [
    "Normal", "Fuzzers", "Analysis", "Backdoor", "DoS",
    "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms"
]

distribution = y_train.value_counts().reindex(
    class_order, fill_value=0
)

print("\nClass distribution:")
print(distribution)

distribution.plot(kind="bar", figsize=(10, 5))
plt.title("Training Class Distribution")
plt.xlabel("Traffic Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 3. RANDOM FOREST
model = Pipeline([
    ("preprocessing", preprocessing),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("\nRandom Forest training completed.")


# 4. CLASSIFICATION METRICS
print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=class_order,
    zero_division=0
))


# 5. NORMALIZED CONFUSION MATRIX
matrix = confusion_matrix(
    y_test,
    predictions,
    labels=class_order,
    normalize="true"
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_order,
    yticklabels=class_order
)

plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# TASK 6: OvR ROC-AUC
y_binary = label_binarize(y_test, classes=class_order)

plt.figure(figsize=(10, 7))

for i, name in enumerate(class_order):
    if len(set(y_binary[:, i])) < 2:
        continue

    fpr, tpr, _ = roc_curve(y_binary[:, i], probabilities[:, i])
    score = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {score:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

print("\nLAB 3 - TASK 1 COMPLETED")
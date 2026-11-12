from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# TASK 1: NLP PIPELINE AND BASELINE CLASSIFIER
print("\nTASK 1: NLP PIPELINE AND BASELINE CLASSIFIER")
base = Path(__file__).resolve().parent
source = Path(r"C:\Users\Lenova\Documents\HW_AIICS\UNSW_NB15_training-set.csv")
data = pd.read_csv(source)
output = base / "lab5_results"
output.mkdir(exist_ok=True)

# 1. LOAD AND INSPECT
print("\n1. LOAD AND INSPECT")

print("Dataset:", source.name)
print("Shape:", data.shape)

required = ["proto", "service", "state", "label"]
missing = set(required) - set(data.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")
if data["label"].isna().any() or not data["label"].isin([0, 1]).all():
    raise ValueError("label must contain only 0 (Normal) and 1 (Attack).")

data["label"] = data["label"].astype(int)

print(data[required].head().to_string(index=False))
print("\nMissing values:")
print(data[required].isna().sum().to_string())
print("\nLabel counts (0 = Normal, 1 = Attack):")
print(data["label"].value_counts().sort_index().to_string())
input("\nPress Enter to continue...")

# 2. CONVERT FLOWS INTO DOCUMENTS
print("\n2. SAMPLE FLOW DOCUMENTS")

columns = ["proto", "service", "state"]
text_features = data[columns].fillna("missing").astype(str)

for column in columns:
    text_features[column] = text_features[column].str.strip().replace("", "missing")

data["document"] = text_features.agg(" ".join, axis=1)

print("Format examples from the lab sheet: tcp http FIN; udp dns INT")
print("Actual sample documents:")
print(data["document"].head(10).to_string(index=False))

# 3. SPLIT AND VECTORISE
print("\n3. SPLIT AND VECTORISE WITH TF-IDF")

train_ids, test_ids = train_test_split(
    data.index, test_size=0.2, stratify=data["label"], random_state=42
)
y_train, y_test = data.loc[train_ids, "label"], data.loc[test_ids, "label"]
vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
X_train = vectorizer.fit_transform(data.loc[train_ids, "document"])
X_test = vectorizer.transform(data.loc[test_ids, "document"])

print("Training / test records:", len(train_ids), len(test_ids))
print("TF-IDF matrix shapes:", X_train.shape, X_test.shape)
print("Vocabulary is learned from the training split only.")

# 4. TRAIN LOGISTIC REGRESSION
print("\n4. TRAIN LOGISTIC REGRESSION")
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
print("Training completed. Decision threshold: 0.5.")
input("\nPress Enter to continue...")


# TASK 2: PERFORMANCE, FALSE NEGATIVES AND MODEL INTERPRETATION
print("\nTASK 2: PERFORMANCE AND MODEL INTERPRETATION")
print("\n1. CLASSIFICATION REPORT")
report = classification_report(
    y_test, predictions, labels=[0, 1], target_names=["Normal", "Attack"],
    digits=4, zero_division=0
)
print(report)
(output / "classification_report.txt").write_text(report, encoding="utf-8")
pd.DataFrame(classification_report(
    y_test, predictions, labels=[0, 1], target_names=["Normal", "Attack"],
    output_dict=True, zero_division=0
)).T.to_csv(output / "metrics.csv", index_label="Class")

matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
tn, fp, fn, tp = matrix.ravel()
fpr = fp / (fp + tn) if fp + tn else 0
fnr = fn / (fn + tp) if fn + tp else 0
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"FPR = {fpr:.4f} ({fpr:.2%})")
print(f"FNR = {fnr:.4f} ({fnr:.2%})")
pd.DataFrame(matrix, index=["Normal", "Attack"], columns=["Normal", "Attack"]).to_csv(
    output / "confusion_matrix.csv", index_label="Actual"
)
plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("TF-IDF Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig(output / "confusion_matrix.png", dpi=150)
plt.show()

# 2. IDENTIFY FALSE NEGATIVES
print("\n2. FALSE NEGATIVES")
test_results = data.loc[test_ids].copy()
test_results["predicted_label"] = predictions
test_results["attack_probability"] = probabilities
test_results.index.name = "source_row_index"
false_negatives = test_results.loc[(y_test == 1) & (predictions == 0)]
false_negatives.to_csv(output / "false_negatives.csv", index=True)
print("Missed attacks:", len(false_negatives))
print(false_negatives[["document", "label", "predicted_label", "attack_probability"]]
      .head(10).to_string(index=False))
print("False negatives let malicious traffic pass as Normal and may delay incident response.")
print("False positives create unnecessary alerts and analyst workload.")
input("\nPress Enter to continue...")

# 3. INTERPRET TOKEN COEFFICIENTS
print("\n3. STRONGEST ATTACK PREDICTORS")
coefficients = pd.DataFrame({
    "Token": vectorizer.get_feature_names_out(),
    "Coefficient": model.coef_[0]
})
field_tokens = {column: set(text_features.loc[train_ids, column]) for column in columns}
coefficients["Field"] = coefficients["Token"].apply(
    lambda token: "/".join(column for column in columns
                          if token in field_tokens[column])
)
coefficients = coefficients.sort_values("Coefficient", ascending=False)
coefficients.to_csv(output / "token_coefficients.csv", index=False)
attack_tokens = coefficients.loc[coefficients["Coefficient"] > 0].head(10)
normal_tokens = coefficients.loc[coefficients["Coefficient"] < 0].tail(10).sort_values("Coefficient")
print("Positive coefficients increase the model's Attack score:")
print(attack_tokens.to_string(index=False, float_format="%.4f"))
print("\nNegative coefficients decrease the model's Attack score:")
print(normal_tokens.to_string(index=False, float_format="%.4f"))
print("\nThese tokens are candidate indicators, not proof that a flow is malicious.")
print("Validate candidate IDS rules on separate traffic and measure false alerts.")
print("Only proto/service/state are used; flows with identical documents receive identical scores.")
print("This random row split measures performance on flows from the selected CSV.")
print("It does not establish performance on unseen networks or unknown attacks.")
print("\nResults saved to:", output)
print("\nLAB 5 COMPLETED")
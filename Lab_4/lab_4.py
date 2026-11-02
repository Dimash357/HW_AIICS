from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)


# TASK 1: DEEP LEARNING CLASSIFICATION (0 = Normal, 1 = Attack)
print("\nTASK 1: DEEP LEARNING CLASSIFICATION")
train = pd.read_csv("../UNSW_NB15_training-set.csv")
test = pd.read_csv("../UNSW_NB15_testing-set.csv")
output = Path("lab4_results")
output.mkdir(exist_ok=True)
print("Training shape:", train.shape)
print("Testing shape:", test.shape)

# 1. PREPROCESSING
for data in [train, test]:
    data.drop(columns=["id", "attack_cat"], inplace=True)
    data.drop_duplicates(inplace=True)
    data["label"] = data["label"].astype(int)
    if not data["label"].isin([0, 1]).all():
        raise ValueError("Expected binary labels: 0 = Normal, 1 = Attack.")

features = train.columns.drop("label")
overlap = pd.MultiIndex.from_frame(train[features]).isin(
    pd.MultiIndex.from_frame(test[features])
)
train = train.loc[~overlap].drop_duplicates(subset=features).copy()
X_train, X_val, y_train, y_val = train_test_split(
    train.drop(columns="label"), train["label"],
    test_size=0.2, stratify=train["label"], random_state=42
)
X_test, y_test = test.drop(columns="label"), test["label"]
numerical = X_train.select_dtypes(include="number").columns.tolist()
categorical = [name for name in X_train if name not in numerical]
preprocessing = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical)
])
X_train = preprocessing.fit_transform(X_train).astype("float32")
X_val = preprocessing.transform(X_val).astype("float32")
X_test = preprocessing.transform(X_test).astype("float32")
print("Training / validation / test:", len(X_train), len(X_val), len(X_test))
print("Input features after encoding:", X_train.shape[1])
counts = y_train.value_counts().sort_index()
print("\nTraining class distribution:")
print(pd.DataFrame({"Count": counts, "Percent": counts / len(y_train) * 100}))
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weights = {0: float(weights[0]), 1: float(weights[1])}
print("Class weights:", class_weights)
input("\nPress Enter to continue...")


# 2. DEFINE AND COMPILE THE DNN
# Identical architecture and initialization for a fair weighting comparison.
def build_model():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")]
    )
    return model


def show_plot(filename):
    plt.tight_layout()
    plt.savefig(output / filename, dpi=150)
    plt.show()


# 3. TRAIN AND EVALUATE BASELINE AND WEIGHTED MODELS
histories, results = {}, []
roc_lines = []
for name, weighting in [("Baseline", None), ("Weighted", class_weights)]:
    print(f"\nTRAINING: {name}")
    model = build_model()
    if name == "Baseline":
        model.summary()
    checkpoint = output / f"{name.lower()}.weights.h5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.CSVLogger(str(output / f"{name.lower()}_training.csv"))
    ]
    history = model.fit(
        X_train, y_train.to_numpy(), validation_data=(X_val, y_val.to_numpy()),
        epochs=30, batch_size=256, class_weight=weighting,
        callbacks=callbacks, verbose=2
    ).history
    histories[name] = history
    # Explicitly evaluate the checkpoint with the lowest validation loss.
    model.load_weights(checkpoint)
    print("\nTest evaluation:", model.evaluate(X_test, y_test.to_numpy(), verbose=0, return_dict=True))
    probability = model.predict(X_test, verbose=0).ravel()
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_test, probability)
    roc_lines.append((name, fpr, tpr, auc(fpr, tpr)))
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1": f1_score(y_test, predicted, zero_division=0),
        "ROC-AUC": auc(fpr, tpr),
        "FPR": fp / (fp + tn) if fp + tn else 0,
        "FNR": fn / (fn + tp) if fn + tp else 0
    })
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    print("Best validation-loss epoch:", best_epoch)
    print("Inspect the curves to identify when overfitting begins.")
    print("\nLast training epochs:")
    print(pd.DataFrame(history, index=range(1, len(history["loss"]) + 1)).tail().round(4))

    plt.figure(figsize=(6, 5))
    sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.title(f"{name} DNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    show_plot(f"{name.lower()}_confusion_matrix.png")

# 4. PERFORMANCE COMPARISON
comparison = pd.DataFrame(results)
print("\nTEST PERFORMANCE (positive class = Attack)")
print(comparison.to_string(index=False, float_format="%.4f"))
comparison.to_csv(output / "metrics.csv", index=False)
input("\nPress Enter to continue...")

# 5. ACCURACY, LOSS AND RECALL CURVES
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for axis, metric in zip(axes, ["accuracy", "loss", "recall"]):
    for name, history in histories.items():
        epochs = range(1, len(history[metric]) + 1)
        axis.plot(epochs, history[metric], linestyle="--", label=f"{name} train")
        axis.plot(epochs, history["val_" + metric], label=f"{name} validation")
    axis.set_title(metric.capitalize())
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric.capitalize())
    axis.legend(fontsize=7)
show_plot("training_curves.png")

# 6. ROC CURVES
plt.figure(figsize=(8, 6))
for name, fpr, tpr, score in roc_lines:
    plt.plot(fpr, tpr, label=f"{name} (AUC = {score:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("DNN ROC Curves")
plt.legend()
show_plot("roc_curves.png")
print("Legend was created")
print("\nLAB 4 COMPLETED")
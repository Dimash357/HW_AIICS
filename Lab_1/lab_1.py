import pandas as pd


train = pd.read_csv("UNSW_NB15_training-set.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv")
df = pd.concat([train, test], ignore_index=True)

print("Datasets loaded successfully!")
print("Combined shape:", df.shape)
input("\nPress Enter to continue...")


# TASK 1: DATASET INVESTIGATION
print("\nTASK 1: DATASET INVESTIGATION")

print("\n1. Dataset shapes")
print("Training:", train.shape)
print("Testing:", test.shape)
print("Combined:", df.shape)
input("\nPress Enter to continue...")

print("\n2. First and last 10 records")
print("\nFIRST 10:")
print(df.head(10).to_string())
print("\nLAST 10:")
print(df.tail(10).to_string())
input("\nPress Enter to continue...")

print("\n3. Rows, columns and data types")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print(df.dtypes.to_string())
input("\nPress Enter to continue...")

print("\n4. Numerical and categorical attributes")
numerical = df.select_dtypes(include="number").columns
categorical = df.columns[df.dtypes.apply(lambda dtype: dtype.kind in "OUS")]
print("Numerical:", numerical.tolist())
print("Categorical:", categorical.tolist())
input("\nPress Enter to continue...")

print("\n5. Missing values")
missing = pd.DataFrame({
    "Missing values": df.isna().sum(),
    "Percentage": df.isna().mean() * 100
})
print(missing.to_string())
input("\nPress Enter to continue...")

print("\n6. Duplicated records")
print("Duplicates with id:", df.duplicated().sum())
print("Duplicates without id:", df.drop(columns="id").duplicated().sum())
input("\nPress Enter to continue...")

print("\n7. Columns with one unique value")
print(df.columns[df.nunique(dropna=False) == 1].tolist())
input("\nPress Enter to continue...")

print("\n8. Descriptive statistics")
print(df[numerical].describe().to_string())
input("\nPress Enter to continue...")

print("\n9. Unique values for categorical attributes")
for column in categorical:
    print(column, ":", df[column].nunique(dropna=False))
input("\nPress Enter to continue...")

print("\n10. Target variable")
print("Target: label (0 = normal, 1 = attack)")
print(df["label"].value_counts().to_string())
input("\nPress Enter to continue...")

print("\nFive feature explanations:")
print("dur: connection duration; keep it to detect unusual timing.")
print("proto: protocol; keep it and encode it before modeling.")
print("service: application service; keep '-' as an unknown-service category.")
print("spkts: source packet count; keep it to spot packet bursts.")
print("sbytes: bytes sent by source; keep it to spot unusual transfers.")
print("Possible redundant/leaking columns: id and attack_cat.")
input("\nPress Enter to continue to Task 2...")


# TASK 2: PREPROCESSING
print("\nTASK 2: DATA QUALITY AND PREPROCESSING")

print("\nA. MISSING VALUES")
print("Missing-value percentages in training data:")
print((train.isna().mean() * 100).to_string())
print("Total missing values in training data:", train.isna().sum().sum())
print("No missing values found. Imputation is kept as a safeguard for new data.")

print("\nB. DUPLICATE REMOVAL (excluding id and attack_cat):")
train_clean = train.drop(columns=["id", "attack_cat"]).copy()
test_clean = test.drop(columns=["id", "attack_cat"]).copy()

for name, data in [("Training", train_clean), ("Testing", test_clean)]:
    before = len(data)
    data.drop_duplicates(inplace=True)
    print(name, "before:", before, "removed:", before - len(data), "remaining:", len(data))

print("\nC. CONSTANT AND NEAR-CONSTANT TRAINING COLUMNS:")
for column in train_clean.columns:
    counts = train_clean[column].value_counts(dropna=False, normalize=True)
    if len(counts) == 1:
        print(column, "- constant")
    elif counts.iloc[0] >= 0.98:
        print(column, "- near-constant (at least 98% identical)")
print("Such columns carry little information, add dimensions and can cause overfitting.")

numeric_columns = train_clean.drop(columns="label").select_dtypes(include="number").columns
category_columns = ["proto", "service", "state"]

for column in numeric_columns:
    replacement = train_clean[column].median()
    if pd.isna(replacement):
        replacement = 0
    train_clean[column] = train_clean[column].fillna(replacement)
    test_clean[column] = test_clean[column].fillna(replacement)

for column in category_columns:
    mode = train_clean[column].mode(dropna=True)
    replacement = mode.iloc[0] if not mode.empty else "Unknown"
    train_clean[column] = train_clean[column].fillna(replacement).astype("string")
    test_clean[column] = test_clean[column].fillna(replacement).astype("string")

print("\nD. DATA TYPES")
print("Before conversion:")
print(train.dtypes.value_counts().to_string())
print("proto, service, state -> string (encoded later, after the split).")
print("label -> int (0 = normal, 1 = attack).")
train_clean["label"] = train_clean["label"].astype(int)
test_clean["label"] = test_clean["label"].astype(int)
print("is_ftp_login has", train['is_ftp_login'].nunique(), "distinct values but is a flag -> converted to binary (>0).")
for data in [train_clean, test_clean]:
    data["is_ftp_login"] = (data["is_ftp_login"] > 0).astype(int)
print("Types after conversion:")
print(train_clean.dtypes.value_counts().to_string())

print("\nE. DATA LEAKAGE")
print("Leakage: attack_cat reveals the attack type and can reveal label.")
print("The id column is just a row number and does not describe traffic, so both columns were removed.")
print("If medians are computed on train + test before splitting, test information leaks into training.")
print("For example, extreme attack traffic in the test set could change a combined median.")
print("Training shape after cleaning:", train_clean.shape)
print("Testing shape after cleaning:", test_clean.shape)


input("\nPress Enter to continue to Task 3...")


# TASK 3: FEATURE ENGINEERING
print("\nTASK 3: FEATURE ENGINEERING")

for data in [train_clean, test_clean]:
    duration = data["dur"].clip(lower=0.001)
    data["total_bytes"] = data["sbytes"] + data["dbytes"]
    data["total_packets"] = data["spkts"] + data["dpkts"]
    data["bytes_per_second"] = data["total_bytes"] / duration
    data["packets_per_second"] = data["total_packets"] / duration
    data["source_byte_fraction"] = data["sbytes"] / data["total_bytes"].clip(lower=1)
    data["packet_asymmetry"] = (
        (data["spkts"] - data["dpkts"]).abs() / data["total_packets"].clip(lower=1)
    )

print("1. total_bytes = sbytes + dbytes: transfer volume; large values may flag bulk transfers.")
print("2. total_packets = spkts + dpkts: packet volume; bursts may indicate automation.")
print("3. bytes_per_second = total_bytes / max(dur, 0.001): high rates may indicate flooding.")
print("4. packets_per_second = total_packets / max(dur, 0.001): high rates may indicate scanning.")
print("5. source_byte_fraction = sbytes / max(total_bytes, 1): one-sided traffic may be unusual.")
print("6. packet_asymmetry = abs(spkts - dpkts) / max(total_packets, 1): imbalance may indicate probes.")
print("No single feature proves that an attack occurred.")

train_clean.to_csv("train_preprocessed.csv", index=False)
test_clean.to_csv("test_preprocessed.csv", index=False)
print("\nSaved: train_preprocessed.csv and test_preprocessed.csv")
print("LAB COMPLETED")

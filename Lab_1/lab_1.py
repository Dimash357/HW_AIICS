import pandas as pd

# Load datasets
train = pd.read_csv("UNSW_NB15_training-set.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv")

# Concatenate datasets
df = pd.concat([train, test], ignore_index=True)

print("Datasets loaded successfully!")
print("Combined dataset shape:", df.shape)

input("\nPress Enter to continue...")


# 1. Load dataset
print("\n1. LOAD DATASET")

print("Training shape:", train.shape)
print("Testing shape:", test.shape)
print("Combined shape:", df.shape)

input("\nPress Enter to continue...")


# 2. First 10 and last 10 records
print("\n2. FIRST AND LAST 10 RECORDS")

print("\nFIRST 10 RECORDS:")
print(df.head(10).to_string())

print("\nLAST 10 RECORDS:")
print(df.tail(10).to_string())

input("\nPress Enter to continue...")


# 3. Rows, columns and data types
print("\n3. DATASET INFORMATION")

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nData types:")
print(df.dtypes)

input("\nPress Enter to continue...")


# 4. Numerical and categorical attributes
print("\n========== 4. ATTRIBUTE TYPES ==========")

numerical = df.select_dtypes(include="number").columns
categorical = df.select_dtypes(include=["object", "str", "category"]).columns

print("\nNumerical attributes:")
print(numerical.tolist())

print("\nCategorical attributes:")
print(categorical.tolist())

input("\nPress Enter to continue...")



# 5. Missing values
print("\n5. MISSING VALUES")

missing = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Percentage": df.isnull().mean() * 100
})

print(missing.to_string())

input("\nPress Enter to continue...")


# 6. Duplicated records
print("\n6. DUPLICATED RECORDS")

duplicates = df.duplicated().sum()

print("Number of duplicated records:", duplicates)

input("\nPress Enter to continue...")


# 7. Columns with one unique value
print("\n7. SINGLE VALUE COLUMNS")

single_value = df.columns[df.nunique() == 1].tolist()

print("Columns containing only one unique value:")
print(single_value)

input("\nPress Enter to continue...")


# 8. Descriptive statistics
print("\n8. DESCRIPTIVE STATISTICS")

print(df[numerical].describe().to_string())

input("\nPress Enter to continue...")


# 9. Unique values for categorical attributes
print("\n9. UNIQUE CATEGORICAL VALUES")

for column in categorical:
    print(column, ":", df[column].nunique())

input("\nPress Enter to continue...")


# 10. Target variable
print("\n10. TARGET VARIABLE")

print("Target variable: label")

print("\nTarget distribution:")
print(df["label"].value_counts())

input("\nPress Enter to continue...")


print("\nINVESTIGATION COMPLETED")

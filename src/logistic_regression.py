# import pandas as pd

# # Load the dataset
# data = pd.read_csv("dataset/data.csv")

# # Display first 5 rows
# print("First 5 rows:")
# print(data.head())

# # Display dataset shape
# print("\nDataset Shape:")
# print(data.shape)

# # Display information about the dataset
# print("\nDataset Information:")
# print(data.info())

# # Check missing values
# print("\nMissing Values:")
# print(data.isnull().sum())

# # Check target class distribution
# print("\nDiagnosis Distribution:")
# print(data["diagnosis"].value_counts())

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("dataset/data.csv")

# Remove unnecessary columns
data = data.drop(columns=["id", "Unnamed: 32"])

# Separate features and target
X = data.drop(columns=["diagnosis"])
y = data["diagnosis"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the results
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())
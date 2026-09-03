import pandas as pd

# Load the dataset
data = pd.read_csv("dataset/data.csv")

# Display first 5 rows
print("First 5 rows:")
print(data.head())

# Display dataset shape
print("\nDataset Shape:")
print(data.shape)

# Display information about the dataset
print("\nDataset Information:")
print(data.info())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check target class distribution
print("\nDiagnosis Distribution:")
print(data["diagnosis"].value_counts())
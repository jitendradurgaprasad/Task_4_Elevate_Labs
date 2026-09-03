import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score
)

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

# Create the scaler
scaler = StandardScaler()

# Fit the scaler on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Create the Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["B", "M"])

# Calculate precision and recall
precision = precision_score(y_test, y_pred, pos_label="M")
recall = recall_score(y_test, y_pred, pos_label="M")

# Display evaluation results
print("Confusion Matrix:")
print(cm)

print("\nPrecision:", precision)
print("Recall:", recall)

# Create confusion matrix plot
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign (B)", "Malignant (M)"]
)

display.plot()
plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()

# Save the plot
plt.savefig("outputs/plots/confusion_matrix.png")
plt.show()
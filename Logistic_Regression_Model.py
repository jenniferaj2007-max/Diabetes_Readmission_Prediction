import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load encoded training and testing data
processed_data = joblib.load("dataset/encoded_data.joblib")

X_train = processed_data["X_train"]
X_test = processed_data["X_test"]

y_train = processed_data["y_train"]
y_test = processed_data["y_test"]

print("TRAINING LOGISTIC REGRESSION MODEL...")

# 2. Create the Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

# 3. Train the model
model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED!")

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

# 6. Save the trained model
joblib.dump(
    model,
    "dataset/logistic_regression_model.joblib"
)

print("\nLOGISTIC REGRESSION MODEL SAVED!")
print("\nLOGISTIC REGRESSION COMPLETED SUCCESSFULLY!")
import joblib

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the encoded dataset
processed_data = joblib.load("dataset/encoded_data.joblib")

X_train = processed_data["X_train"]
X_test = processed_data["X_test"]

y_train = processed_data["y_train"]
y_test = processed_data["y_test"]

print("TRAINING SVM MODEL...")

# 2. Create the SVM model
model = LinearSVC(
    max_iter=3000,
    random_state=42
)

# 3. Train the model
model.fit(X_train, y_train)

print("\nSVM MODEL TRAINING COMPLETED!")

# 4. Predict the results
y_pred = model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nSVM MODEL RESULTS")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

# 6. Save the trained model
joblib.dump(
    model,
    "dataset/svm_model.joblib"
)

print("\nSVM MODEL SAVED!")
print("\nSVM MODEL COMPLETED SUCCESSFULLY!")
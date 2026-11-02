import joblib

from sklearn.ensemble import RandomForestClassifier
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

print("TRAINING RANDOM FOREST MODEL...")

# 2. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# 3. Train the model
model.fit(X_train, y_train)

print("\nRANDOM FOREST TRAINING COMPLETED!")

# 4. Predict the results
y_pred = model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

# 6. Save the trained model
joblib.dump(
    model,
    "dataset/random_forest_model.joblib"
)

print("\nRANDOM FOREST MODEL SAVED!")
print("\nRANDOM FOREST COMPLETED SUCCESSFULLY!")
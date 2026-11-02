import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load test data
processed_data = joblib.load("dataset/encoded_data.joblib")

X_test = processed_data["X_test"]
y_test = processed_data["y_test"]

# 2. Load all three trained models
models = {
    "Logistic Regression": joblib.load(
        "dataset/logistic_regression_model.joblib"
    ),
    "SVM": joblib.load(
        "dataset/svm_model.joblib"
    ),
    "Random Forest": joblib.load(
        "dataset/random_forest_model.joblib"
    )
}

results = []

print("COMPARING MACHINE LEARNING MODELS...")

# 3. Evaluate each model
for name, model in models.items():

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

# 4. Display comparison
comparison = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
print(comparison.to_string(index=False))

# 5. Save comparison results
comparison.to_csv("results/model_comparison_results.csv",index=False)                  
    


print("\nCOMPARISON RESULTS SAVED!")
print("\nMODEL COMPARISON COMPLETED SUCCESSFULLY!")

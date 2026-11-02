import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

# 1. Load test data
processed_data = joblib.load("dataset/encoded_data.joblib")

X_test = processed_data["X_test"]
y_test = processed_data["y_test"]

# 2. Load trained models
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

# 3. Define class labels
class_labels = ["<30", ">30", "NO"]

# 4. Generate confusion matrices
for name, model in models.items():

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=class_labels
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels
    )

    plt.title(name + " - Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")

    plt.tight_layout()

    filename = name.replace(" ", "_").lower()

    plt.savefig(
        "results/" + filename + "_confusion_matrix.png"
    )

    plt.show()

    plt.close()

print("\nALL CONFUSION MATRICES SAVED!")
print("CONFUSION MATRIX VISUALIZATION COMPLETED!")
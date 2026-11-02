import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Random Forest model
model = joblib.load(
    "dataset/random_forest_model.joblib"
)

# 2. Load the preprocessing object
preprocessor = joblib.load(
    "dataset/preprocessor.joblib"
)

# 3. Get feature names after encoding
feature_names = preprocessor.get_feature_names_out()

# 4. Get feature importance values
importance_values = model.feature_importances_

# 5. Create a feature importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# 6. Sort features by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# 7. Display top 15 features
top_features = importance_df.head(15)

print("\nTOP 15 IMPORTANT FEATURES")
print(top_features.to_string(index=False))

# 8. Save feature importance results
importance_df.to_csv(
    "results/feature_importance_results.csv",
    index=False
)

# 9. Create feature importance graph
plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.title("Random Forest - Top 15 Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig(
    "results/feature_importance_graph.png"
)

plt.show()

print("\nFEATURE IMPORTANCE RESULTS SAVED!")
print("FEATURE IMPORTANCE ANALYSIS COMPLETED!")
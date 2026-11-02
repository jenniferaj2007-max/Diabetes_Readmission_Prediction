import pandas as pd
import matplotlib.pyplot as plt

# 1. Load model comparison results
data = pd.read_csv(
    "results/model_comparison_results.csv"
)

# 2. Select model names
data.set_index("Model", inplace=True)

# 3. Create performance comparison graph
data.plot(
    kind="bar",
    figsize=(10, 6),
    width=0.8
)

# 4. Add graph details
plt.title("Machine Learning Model Performance Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Performance Percentage")

plt.xticks(rotation=0)
plt.ylim(0, 100)

plt.legend(title="Performance Metrics")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

# 5. Save graph
plt.savefig(
    "results/model_performance_comparison.png"
)

# 6. Display graph
plt.show()

print("\nMODEL PERFORMANCE GRAPH SAVED!")
print("MODEL VISUALIZATION COMPLETED SUCCESSFULLY!")
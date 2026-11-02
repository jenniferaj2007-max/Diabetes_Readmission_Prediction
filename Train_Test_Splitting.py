import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 1. Load prepared dataset
data = pd.read_csv("dataset/diabetes_feature_prepared.csv")

print("ORIGINAL DATASET SHAPE:", data.shape)

# 2. Separate features, target and patient groups
X = data.drop(columns=["readmitted", "patient_nbr"])
y = data["readmitted"]
groups = data["patient_nbr"]

# 3. Split data into 80% training and 20% testing
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

# 4. Display training and testing information
print("\nTRAINING DATA SHAPE:", X_train.shape)
print("TESTING DATA SHAPE:", X_test.shape)

print("\nTRAINING TARGET DISTRIBUTION")
print(y_train.value_counts())

print("\nTESTING TARGET DISTRIBUTION")
print(y_test.value_counts())

# 5. Check patient overlap
train_patients = set(groups.iloc[train_index])
test_patients = set(groups.iloc[test_index])

print("\nCOMMON PATIENTS IN BOTH SETS:",
      len(train_patients.intersection(test_patients)))

# 6. Save split datasets
train_data = X_train.copy()
train_data["readmitted"] = y_train

test_data = X_test.copy()
test_data["readmitted"] = y_test

train_data.to_csv("dataset/train_data.csv", index=False)
test_data.to_csv("dataset/test_data.csv", index=False)

print("\nTRAIN-TEST SPLITTING COMPLETED SUCCESSFULLY!")
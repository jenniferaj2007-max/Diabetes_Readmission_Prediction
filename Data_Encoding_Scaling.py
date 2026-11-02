import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. Load training and testing datasets
train_data = pd.read_csv("dataset/train_data.csv")
test_data = pd.read_csv("dataset/test_data.csv")

# 2. Separate features and target
X_train = train_data.drop(columns=["readmitted"])
y_train = train_data["readmitted"]

X_test = test_data.drop(columns=["readmitted"])
y_test = test_data["readmitted"]

# 3. Identify categorical columns
categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

# These numeric codes represent categories
coded_columns = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]

for column in coded_columns:
    if column in X_train.columns:
        categorical_columns.append(column)

# 4. Identify numerical columns
numerical_columns = [
    column for column in X_train.columns
    if column not in categorical_columns
]

print("CATEGORICAL FEATURES:", len(categorical_columns))
print("NUMERICAL FEATURES:", len(numerical_columns))

# 5. Create numerical preprocessing pipeline
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. Create categorical preprocessing pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 7. Combine both pipelines
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# 8. Fit only on training data
X_train_encoded = preprocessor.fit_transform(X_train)

# 9. Transform testing data
X_test_encoded = preprocessor.transform(X_test)

# 10. Display results
print("\nENCODING AND SCALING RESULTS")
print("Training data shape:", X_train_encoded.shape)
print("Testing data shape:", X_test_encoded.shape)

print("\nTARGET CLASSES:")
print(y_train.value_counts())

# 11. Save preprocessing information
joblib.dump(preprocessor, "dataset/preprocessor.joblib")

joblib.dump(
    {
        "X_train": X_train_encoded,
        "X_test": X_test_encoded,
        "y_train": y_train,
        "y_test": y_test
    },
    "dataset/encoded_data.joblib"
)

print("\nENCODING AND SCALING COMPLETED SUCCESSFULLY!")
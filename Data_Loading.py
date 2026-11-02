import pandas as pd
import numpy as np

data = pd.read_csv("dataset/diabetic_data.csv")

print("DIABETIC READMISSION DATASET")

print("\nfirst 5 rows:")
print(data.shape)

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

print("\nData Types:")
print(data.dtypes)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nReadmission Distribution:")
print(data["readmitted"].value_counts())

print("\nDataset Loading Completed Successfully!")
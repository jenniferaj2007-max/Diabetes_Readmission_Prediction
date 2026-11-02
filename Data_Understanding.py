import pandas as pd
import numpy as np

data = pd.read_csv("dataset/diabetic_data.csv")

print("\n1. DATASET INFORMATION")
print(data.info())

print("\n2. DATASET SHAPE")
print("Rows and Columns:",data.shape)

print("\n3. MISSING VALUES")
data.replace("?",np.nan, inplace=True)
print(data.isnull().sum())

print("\n4. DUPLICATE RECORDS")
print("Duplicates:",data.duplicated().sum())

print("\n5. READMISSION DISTRIBUTION")
print(data["readmitted"].value_counts())

print("\n6. READMISSION PERCENTAGES")
print(data["readmitted"].value_counts(normalize=True)*100)

print("\n7. UNIQUE VALUES")
print("Age categories:",
data["age"].nunique())
print("Gender categories:",
data["gender"].nunique())   
print("Race categories:",
data["race"].nunique())  

print("\nDATASET UNDERSTANDING COMPLETED!")
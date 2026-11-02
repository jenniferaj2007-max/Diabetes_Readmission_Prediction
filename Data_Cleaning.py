import pandas as pd
import numpy as np

data = pd.read_csv("dataset/diabetic_data.csv")

print("ORIGINAL DATASET SHAPE:",data.shape)

data.replace("?",np.nan,inplace=True)

missing_percentage = data.isnull().mean()*100

print("\nMISSING VALUE PERCENTAGES")
print(missing_percentage[missing_percentage>0].sort_values(ascending=False))

high_missing_columns = missing_percentage[missing_percentage > 90].index.tolist()

print("\nCOLUMNS REMOVED:")
print(high_missing_columns)

data.drop(columns=high_missing_columns,inplace=True)

categorical_columns = data.select_dtypes(include=["object"]).columns

data[categorical_columns] = data[categorical_columns].fillna("Unknown")

numerical_columns = data.select_dtypes(include=["number"]).columns

for column in numerical_columns:
    data[column] = data[column].fillna(data[column].median())

print("\nMISSING VALUES AFTER CLEANING")
print(data.isnull().sum().sum())

print("\nCLEANED DATASET SHAPE")
print(data.shape)

data.to_csv("dataset/diabetic_data_cleaned.csv", index=False)

print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")


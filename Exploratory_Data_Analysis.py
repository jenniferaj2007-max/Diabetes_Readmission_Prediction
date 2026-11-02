import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
data = pd.read_csv("dataset/diabetic_data_cleaned.csv")

sns.set_theme(style="whitegrid")

# 1. Age distribution
plt.figure(figsize=(9, 5))
sns.countplot(data=data, x="age")
plt.title("Patient Age Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Gender distribution
plt.figure(figsize=(6, 5))
sns.countplot(data=data, x="gender")
plt.title("Gender Distribution")
plt.tight_layout()
plt.show()

# 3. Readmission distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x="readmitted")
plt.title("Hospital Readmission Distribution")
plt.tight_layout()
plt.show()

# 4. Hospital stay distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x="time_in_hospital", bins=14)
plt.title("Length of Hospital Stay")
plt.tight_layout()
plt.show()

# 5. Readmission by age
plt.figure(figsize=(10, 5))
sns.countplot(data=data, x="age", hue="readmitted")
plt.title("Readmission by Age Group")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("EDA COMPLETED SUCCESSFULLY!")
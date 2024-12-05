# Databricks notebook source
df = spark.read.table("hive_metastore.default.final_depression_dataset")
df = df.toPandas()  


# COMMAND ----------

# Check the shape
print("Data Shape:", df.shape)

# View the first few rows
display(df.head())

# Check data types
print(df.info())

# Check descriptive statistics
display(df.describe(include='all'))


# COMMAND ----------

# Count missing values per column
missing_counts = df.isnull().sum()
display(missing_counts)

# COMMAND ----------

# Assuming target is "Depression" which could be Yes/No (categorical)
target_counts = df['Depression'].value_counts()
display(target_counts)

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Depression', data=df)
plt.title("Distribution of Depression")
plt.show()

# COMMAND ----------

categorical_cols = ['Gender', 'City', 'Working Professional or Student', 'Profession', 'Academic Pressure', 
                    'Work Pressure', 'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration', 
                    'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 'Financial Stress', 
                    'Family History of Mental Illness',  'Work Pressure', 'Job Satisfaction']
for col in categorical_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, data=df)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# COMMAND ----------

# Identify numerical columns 
numeric_cols = ['Age', 'CGPA', 'Work/Study Hours']

for col in numeric_cols:
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    sns.histplot(df[col], kde=True, ax=axes[0])
    sns.boxplot(x=df[col], ax=axes[1])
    axes[0].set_title(f"Histogram of {col}")
    axes[1].set_title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()


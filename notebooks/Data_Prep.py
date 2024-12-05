# Databricks notebook source
import pandas as pd

df_spark = spark.read.table("hive_metastore.default.final_depression_dataset")
df = df_spark.toPandas()
print("Raw data shape:", df.shape)


# COMMAND ----------

# Count missing values per column
missing_counts = df.isnull().sum()
display(missing_counts)

# COMMAND ----------

df['Profession'] = df['Profession'].fillna('None')

# If Academic Pressure is not null, use it; otherwise use Work Pressure
df['Pressure'] = df['Academic Pressure'].where(df['Academic Pressure'].notnull(), df['Work Pressure'])

# Drop the original columns
df.drop(['Academic Pressure', 'Work Pressure'], axis=1, inplace=True)

# Same for Satisfaction
df['Satisfaction'] = df['Study Satisfaction'].where(df['Study Satisfaction'].notnull(), df['Job Satisfaction'])

# Drop the original columns
df.drop(['Study Satisfaction', 'Job Satisfaction'], axis=1, inplace=True)


# COMMAND ----------

if 'CGPA' in df.columns:
    df.drop('CGPA', axis=1, inplace=True)

# COMMAND ----------

# Identify numeric and categorical columns
numeric_cols = ['Age', 'Work/Study Hours']
 
categorical_cols = ['Gender', 'City', 'Working Professional or Student', 'Profession',  'Sleep Duration', 
                    'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 
                    'Financial Stress', 'Family History of Mental Illness', 'Pressure', 'Satisfaction']


# COMMAND ----------

df['Depression'] = df['Depression'].map({'Yes':1, 'No':0})
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = df_encoded.drop('Depression', axis=1)
y = df_encoded['Depression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# COMMAND ----------

train_spark = spark.createDataFrame(pd.concat([X_train, y_train], axis=1))
test_spark = spark.createDataFrame(pd.concat([X_test, y_test], axis=1))

train_spark.write.mode("overwrite").saveAsTable("hive_metastore.default.train_data")
test_spark.write.mode("overwrite").saveAsTable("hive_metastore.default.test_data")


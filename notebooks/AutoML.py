# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Import/Cleaning

# COMMAND ----------

import pandas as pd
from databricks import automl
import mlflow
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# COMMAND ----------

df_spark = spark.read.table("hive_metastore.default.final_depression_dataset")
df = df_spark.toPandas()
df.shape

# COMMAND ----------

df.drop(['Name'], axis=1, inplace=True)

# COMMAND ----------

# Seperating df into 75% training, 25% inference
df_train, df_inf = df.iloc[:int(df.shape[0]*0.75)], df.iloc[int(df.shape[0]*0.75):]
print(df_train.shape)
print(df_inf.shape)

# COMMAND ----------

df_train.head()

# COMMAND ----------

df_train['Profession'] = df_train['Profession'].fillna('None')

# Combining Columns
df_train['Pressure'] = df_train['Academic Pressure'].where(df_train['Academic Pressure'].notnull(), df_train['Work Pressure'])
df_train['Satisfaction'] = df_train['Study Satisfaction'].where(df_train['Study Satisfaction'].notnull(), df_train['Job Satisfaction'])

df_train.drop(['Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction'], axis=1, inplace=True)

if 'CGPA' in df_train.columns:
    df_train.drop('CGPA', axis=1, inplace=True)

df_train['Depression'] = df_train['Depression'].map({'Yes':1, 'No':0})



# COMMAND ----------

# Adding class weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=df_train['Depression'].unique(),
    y=df_train['Depression']
)

# Create a mapping for weights
weight_map = {cls: weight for cls, weight in zip(df_train['Depression'].unique(), class_weights)}

# Add a sample weight column
df_train['sample_weight'] = df_train['Depression'].map(weight_map)

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoML Run

# COMMAND ----------

# Running AutoML
automl_results = automl.classify(
    dataset=df_train,
    target_col="Depression",
    primary_metric='f1',
    pos_label = 1,
    exclude_cols = ['sample_weight'],
    sample_weight_col="sample_weight",
    experiment_dir="/Workspace/Users/hsnowdon@uchicago.edu",
    experiment_name="depression_automl_experiment_v3",
    timeout_minutes=30  # Max time for the experiment
)

best_model = automl_results.best_trial
print(f"Best Model: {best_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

help(automl_results)  

# COMMAND ----------

len(automl_results.trials)

# COMMAND ----------

trial_f1_scores = []

for trial in automl_results.trials:
    trial_f1_scores.append(trial.metrics['test_f1_score'])

# Creating the histogram 
plt.figure(figsize=(12, 5))
plt.hist(trial_f1_scores, bins=20, color='#1f77b4', edgecolor='black', alpha=0.7) 
plt.xlabel('F1-Score') 
plt.ylabel('Frequency') 
plt.title('Trial F1-Score Distributions') 

plt.show()

# COMMAND ----------

from collections import Counter

trial_model_types = []

for trial in automl_results.trials:
    trial_model_types.append(trial.model_description.split('(')[0])

model_counts = Counter(trial_model_types)
model_names = list(model_counts.keys()) 
counts = list(model_counts.values()) 

plt.figure(figsize=(12, 5))
bars = plt.bar(model_names, counts, color='#1f77b4', edgecolor='grey')

for bar in bars: 
    height = bar.get_height() 
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')

plt.xlabel('Model Type') 
plt.ylabel('Frequency') 
plt.title('Trial Model Type Frequency') 

plt.show()

# COMMAND ----------

automl_results.best_trial.model_path

# COMMAND ----------

model_uri = best_model.model_path
mlflow.register_model(model_uri=model_uri, name="BestAutoMLModel_v3")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLFlow - recalling model

# COMMAND ----------

from databricks import automl
import mlflow.pyfunc

# Specify the DBFS path to the model
model_uri = "dbfs:/databricks/mlflow-tracking/1091853668972228/58401521d2d74d1eb0eaf9237bbee9a7/artifacts/model"

# Load the model from DBFS
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

model.metadata.signature

# COMMAND ----------

# Transforming inference data
df_inf['Profession'] = df_inf['Profession'].fillna('None')

# Combining Columns
df_inf['Pressure'] = df_inf['Academic Pressure'].where(df_inf['Academic Pressure'].notnull(), df_inf['Work Pressure'])
df_inf['Satisfaction'] = df_inf['Study Satisfaction'].where(df_inf['Study Satisfaction'].notnull(), df_inf['Job Satisfaction'])

df_inf.drop(['Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction'], axis=1, inplace=True)

if 'CGPA' in df_inf.columns:
    df_inf.drop('CGPA', axis=1, inplace=True)

df_inf['Depression'] = df_inf['Depression'].map({'Yes':1, 'No':0})

# COMMAND ----------

y_pred = model.predict(df_inf)
y_pred

# COMMAND ----------



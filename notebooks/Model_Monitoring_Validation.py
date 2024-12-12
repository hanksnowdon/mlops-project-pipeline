# Databricks notebook source
# Binary Classification with MLflow and Evidently for Monitoring

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# COMMAND ----------

from databricks import automl
import mlflow.pyfunc

# Specify the DBFS path to the model
model_uri = 'dbfs:/databricks/mlflow-tracking/1091853668972228/58401521d2d74d1eb0eaf9237bbee9a7/artifacts/model'

# Load the model from DBFS
model = mlflow.pyfunc.load_model(model_uri)

model.metadata.signature

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Import & Cleaning

# COMMAND ----------

# Data import
df_spark = spark.read.table("hive_metastore.default.final_depression_dataset")
df = df_spark.toPandas()
df.shape

# Seperating df into 75% training, 25% inference
df_train, df_inf = df.iloc[:int(df.shape[0]*0.75)], df.iloc[int(df.shape[0]*0.75):]

# Training data
df_train['Profession'] = df_train['Profession'].fillna('None')

# Combining Columns
df_train['Pressure'] = df_train['Academic Pressure'].where(df_train['Academic Pressure'].notnull(), df_train['Work Pressure'])
df_train['Satisfaction'] = df_train['Study Satisfaction'].where(df_train['Study Satisfaction'].notnull(), df_train['Job Satisfaction'])

df_train.drop(['Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction'], axis=1, inplace=True)

if 'CGPA' in df_train.columns:
    df_train.drop('CGPA', axis=1, inplace=True)

df_train['Depression'] = df_train['Depression'].map({'Yes':1, 'No':0})


# Inference data - prev. set aside and not used for model train/test/val

# Transforming inference data
df_inf['Profession'] = df_inf['Profession'].fillna('None')

# Combining Columns
df_inf['Pressure'] = df_inf['Academic Pressure'].where(df_inf['Academic Pressure'].notnull(), df_inf['Work Pressure'])
df_inf['Satisfaction'] = df_inf['Study Satisfaction'].where(df_inf['Study Satisfaction'].notnull(), df_inf['Job Satisfaction'])

df_inf.drop(['Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction'], axis=1, inplace=True)

if 'CGPA' in df_inf.columns:
    df_inf.drop('CGPA', axis=1, inplace=True)

df_inf['Depression'] = df_inf['Depression'].map({'Yes':1, 'No':0})

print(df_train.shape)
print(df_inf.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initial Model Inference - No Change

# COMMAND ----------

X_train = df_train.drop(columns=['Depression'], axis=1)
X_inf = df_inf.drop(columns=['Depression'], axis=1)
y_train = df_train['Depression']
y_inf = df_inf['Depression']

# Column Mapping

X_train_initial = X_train.copy()
X_inf_initial = X_inf.copy()

# Add pred/target to dataset
X_train_initial['prediction'] = model.predict(X_train_initial) # Training - predictions
X_inf_initial['prediction'] = model.predict(X_inf_initial) # Testing - predictions
X_train_initial['target'] = y_train # Training - ground truth
X_inf_initial['target'] = y_inf # Inference - ground truth

# COMMAND ----------

# Test the model on the initial training set and log metrics 
accuracy = accuracy_score(y_train, X_train_initial['prediction'])
precision = precision_score(y_train, X_train_initial['prediction'])
recall = recall_score(y_train, X_train_initial['prediction'])
f1 = f1_score(y_train, X_train_initial['prediction'])
roc_auc = roc_auc_score(y_train, X_train_initial['prediction'])

print(f"Initial Accuracy: {accuracy:.4f}")
print(f"Initial Precision: {precision:.4f}")
print(f"Initial Recall: {recall:.4f}")
print(f"Initial F1 Score: {f1:.4f}")
print(f"Initial ROC-AUC: {roc_auc:.4f}")

# COMMAND ----------

# Log metrics and model using MLflow
mlflow.end_run()  # Ensure no active run
with mlflow.start_run():
    # Log metrics
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1_Score", f1)
    mlflow.log_metric("ROC_AUC", roc_auc)

    # Log the model
    mlflow.sklearn.log_model(model, artifact_path="Monitoring Logs/")

    # Generate and log performance and drift reports for the initial test set (CHANGE TO OUR TARGET VARIABLE)
    column_mapping = ColumnMapping(target="target", prediction="prediction", numerical_features=list(X_train_initial.columns))
    performance_report = Report(metrics=[ClassificationPreset()])
    performance_report.run(reference_data=X_train_initial, current_data=X_inf_initial, column_mapping=column_mapping)
    performance_report.save_html("initial_performance_report.html")
    mlflow.log_artifact("initial_performance_report.html")

    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=X_train_initial, current_data=X_inf_initial, column_mapping=column_mapping)
    data_drift_report.save_html("initial_data_drift_report.html")
    mlflow.log_artifact("initial_data_drift_report.html")

    print("Performance and data drift reports logged for initial test set.")

print("Model and metrics logged to MLflow.")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## Introducing Change During Serving

# COMMAND ----------

# Feature drift experiment
mlflow.end_run()  # Ensure no active run
with mlflow.start_run():

    # Swap features in the test dataset
    df_inf_modified = df_inf.copy()
    X_inf_modified = df_inf_modified.drop(columns=['Depression'], axis=1)
    y_inf_modified = df_inf_modified['Depression']

    X_inf_modified['Age'], X_inf_modified['Financial Stress'] = (
        X_inf_modified['Financial Stress'].copy(), X_inf_modified['Age'].copy()
    )

    X_inf_modified['prediction'] = model.predict(X_inf_modified) # Inference - predictions
    X_inf_modified['target'] = y_inf # Inference - ground truth
   

    # Test the model with modified features
    modified_accuracy = accuracy_score(X_inf_modified['target'], X_inf_modified['prediction'])
    modified_precision = precision_score(X_inf_modified['target'], X_inf_modified['prediction'])
    modified_recall = recall_score(X_inf_modified['target'], X_inf_modified['prediction'])
    modified_f1 = f1_score(X_inf_modified['target'], X_inf_modified['prediction'])
    modified_roc_auc = roc_auc_score(X_inf_modified['target'], X_inf_modified['prediction'])

    print(f"Modified Accuracy: {modified_accuracy:.4f}")
    print(f"Modified Precision: {modified_precision:.4f}")
    print(f"Modified Recall: {modified_recall:.4f}")
    print(f"Modified F1 Score: {modified_f1:.4f}")
    print(f"Modified ROC-AUC: {modified_roc_auc:.4f}")

    # Log modified metrics
    mlflow.log_metric("Modified_Accuracy", modified_accuracy)
    mlflow.log_metric("Modified_Precision", modified_precision)
    mlflow.log_metric("Modified_Recall", modified_recall)
    mlflow.log_metric("Modified_F1_Score", modified_f1)
    mlflow.log_metric("Modified_ROC_AUC", modified_roc_auc)

    # Generate and log performance and drift reports for the modified test set
    performance_report = Report(metrics=[ClassificationPreset()])
    performance_report.run(reference_data=X_train_initial, current_data=X_inf_modified, column_mapping=column_mapping)
    performance_report.save_html("modified_performance_report.html")
    mlflow.log_artifact("modified_performance_report.html")

    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=X_train_initial, current_data=X_inf_modified, column_mapping=column_mapping)
    data_drift_report.save_html("modified_data_drift_report.html")
    mlflow.log_artifact("modified_data_drift_report.html")

    print("Performance and data drift reports logged for modified test set.")

    # Log the model again
    mlflow.sklearn.log_model(model, artifact_path="binary_classification_with_drift")

print("Modified feature experiment and drift logged to MLflow.")

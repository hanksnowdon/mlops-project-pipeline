# Databricks notebook source
df = spark.read.table("hive_metastore.default.final_depression_dataset")
df = df.toPandas()  # If small enough, else continue with Spark DataFrame methods

df.head()


# COMMAND ----------



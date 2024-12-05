# Databricks notebook source
df = spark.read.table("hive_metastore.default.final_depression_dataset")
df = df.toPandas()  

df.head()


# COMMAND ----------



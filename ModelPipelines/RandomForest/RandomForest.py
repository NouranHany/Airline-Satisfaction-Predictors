# pylint: disable-all
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer, StringIndexer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

spark = SparkSession.builder \
    .appName("Naive Bayse") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

def process_dataframe(df, string_indexers={}, bucketizers={}):
  # Remove null values
  df = df.dropna() # Very little data loss
  # Remove id and _c0
  df = df.drop("_c0", "id")
  stringCols = [item[0] for item in df.dtypes if item[1].startswith('string')]
  # Make a dictionary of string indexers
  if len(string_indexers) == 0:
    string_indexers = {x: StringIndexer(inputCol=x, outputCol=x + "_index").fit(df) for x in stringCols}
  # Apply them to the dataframe
  for x in string_indexers:
      df = string_indexers[x].transform(df)
  # Remove the original columns
  df = df.drop(*stringCols)
  
  numericCols = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
  # Make a dictionary of bucketizers (use mean and std to find the splits)
  number_splits = 10
  # Compute unique splits for each numeric column
  if len(bucketizers) == 0:
    bucketizers = {
        x: Bucketizer(
            splits=sorted(list(set(df.approxQuantile(x, [i / number_splits for i in range(number_splits + 1)], 0)))),
            inputCol=x,
            outputCol=x + "_bucket"
        )
        for x in numericCols
    }

  # Apply them to the dataframe
  for x in bucketizers:
      df = bucketizers[x].transform(df)
  # Remove the original columns
  df = df.drop(*numericCols)
  
  for feature in df.columns:
    df = df.withColumn(feature, df[feature].cast("integer"))
  
  return df, string_indexers, bucketizers

def importance_bar_plot(model, features):
  # Get feature importances from model
  importances = model.featureImportances.toArray()
  # Create pandas dataframe of features and importances
  feature_df = pd.DataFrame({"feature": features, "importance": importances})
  # Sort features by importance
  feature_df = feature_df.sort_values("importance", ascending=True)
  # Set dpi and style of charts
  plt.rcParams["figure.dpi"] = 300
  plt.style.use("dark_background")
  cmap = plt.get_cmap("magma")
  colors = [cmap(i) for i in np.linspace(0, 1, len(feature_df))]
  # Create the bar plot
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.barh(feature_df["feature"], feature_df["importance"], color=colors)
  # Set the labels and title
  ax.set_xlabel("Importance")
  ax.set_ylabel("Feature")
  ax.set_title("Feature Importance")
  # Show the plot
  plt.show()
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import confusion_matrix
from pyspark.sql import SparkSession
from pyspark.ml.feature import  StringIndexer
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, mean, stddev
import os
import functools



def unionAll(dfs):
    return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)
 

def read_data(preprocess=True):
    spark=SparkSession.builder.getOrCreate()
    module_dir = os.path.dirname(__file__)
    train_path = os.path.join(module_dir, '../../DataFiles/airline-train.csv')
    val_path = os.path.join(module_dir, '../../DataFiles/airline-val.csv')

    
    df_train=spark.read.csv(train_path,header=True,inferSchema=True)
    df_test=spark.read.csv(val_path,header=True,inferSchema=True)


    if preprocess:
        df_train,string_indexers,aggregates,stats=preprocess_data(df_train)
        df_test,_,_,_=preprocess_data(df_test,string_indexers,aggregates,stats)
    return df_train,df_test



def preprocess_data(df,string_indexers={},aggregates=None,stats=None):
    # 1.Dropping Nans
    df = df.dropna() 
    # 2. Dropping the columns that are not needed
    df=df.drop('_c0').drop('id')

    # 3.convert all the columns that are string to integer (e.g: Gender -> 0,1)
    stringCols = [item[0] for item in df.dtypes if item[1].startswith('string')]

    if len(string_indexers) == 0:
        string_indexers = {x: StringIndexer(inputCol=x, outputCol=x + "_index").fit(df) for x in stringCols}
    
    # Apply them to the dataframe
    for x in string_indexers:
        df = string_indexers[x].transform(df)
    df = df.drop(*stringCols)

    # 4. Need to Normalize the numeric columns 
    numericCols = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]

    if aggregates is None:
        # Compute the mean and standard deviation for each column
        aggregates = [mean(c).alias(c + "_mean") for c in numericCols] + [stddev(c).alias(c + "_stddev") for c in numericCols]
        stats = df.agg(*aggregates).collect()[0]

    # Normalize each column
    for col_name in numericCols:
        col_mean = stats[col_name + "_mean"]
        col_stddev = stats[col_name + "_stddev"]
        df = df.withColumn(col_name + "_scaled", (col(col_name) - col_mean) / col_stddev)

    # Drop the original columns
    df = df.drop(*numericCols)

    return df,string_indexers,aggregates,stats



def prepare_data(df):
    
    y="satisfaction_index"
    assembler = VectorAssembler(inputCols=list(set(df.columns) - set([y])), outputCol="features")
    df=assembler.transform(df)

    return  df.select("features", "satisfaction_index")
    





def svc_model(train_data,test_data):

    # Create an SVM model
    lsvc = LinearSVC(maxIter=10, regParam=0.1, featuresCol="features", labelCol="satisfaction_index")

    # Fit the model to the training data
    lsvcModel = lsvc.fit(train_data)

    # Make predictions on the training and test data
    train_predictions = lsvcModel.transform(train_data)
    test_predictions = lsvcModel.transform(test_data)

    # Evaluate the accuracy of the model on the training and test data
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy",labelCol="satisfaction_index")
    train_accuracy = evaluator.evaluate(train_predictions)
    test_accuracy = evaluator.evaluate(test_predictions)

    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")


def Grid_Search(train_data, test_data):
   

    # Create an SVM model
    lsvc = LinearSVC(featuresCol="features", labelCol="satisfaction_index")

    # Create a parameter grid
    paramGrid = ParamGridBuilder() \
            .addGrid(lsvc.regParam, [0.1, 0.01]) \
            .addGrid(lsvc.maxIter, [10, 100]) \
            .build()

    # Create a binary classification evaluator
    evaluator = BinaryClassificationEvaluator(labelCol="satisfaction_index")

    # Create a cross-validator
    cv = CrossValidator(estimator=lsvc,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=5)

    # Fit the cross-validator to the training data
    cvModel = cv.fit(train_data)

    # Get the best model
    bestModel = cvModel.bestModel

    # Print the best hyperparameters
    print(f"Best regParam: {bestModel._java_obj.getRegParam()}")
    print(f"Best maxIter: {bestModel._java_obj.getMaxIter()}")

    # Make predictions on the training and test data
    train_predictions = bestModel.transform(train_data)
    test_predictions = bestModel.transform(test_data)

    # Evaluate the accuracy of the model on the training and test data
    evaluator = MulticlassClassificationEvaluator(labelCol="satisfaction_index", metricName="accuracy")
    train_accuracy = evaluator.evaluate(train_predictions)
    test_accuracy = evaluator.evaluate(test_predictions)

    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
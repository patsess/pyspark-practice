
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import round, regexp_replace
from pyspark.ml.feature import (StringIndexer, VectorAssembler, Tokenizer,
    StopWordsRemover, HashingTF, IDF, OneHotEncoderEstimator, Bucketizer)
from pyspark.ml.classification import (DecisionTreeClassifier,
    LogisticRegression, GBTClassifier, RandomForestClassifier)
from pyspark.ml.evaluation import (MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator, RegressionEvaluator)
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

"""
Note: code is for reference only (taken from an online course)
"""


if __name__ == '__main__':
    ##############################
    # CONNECTING TO SPARK SESSION:
    #
    # Remote Cluster using Spark URL - spark://<IO address | DNS name>:<port>
    # Examples:
    # - spark://13.59.151.161:7077
    # - spark://ec2-18-188-22-23.us-east-2.compute.amazonaws.com:7077
    # note: 7077 is the "default" port for Spark (although must be specified)
    #
    # Local Cluster
    # Examples:
    # - local - only 1 core;
    # - local[4] - 4 cores; or
    # - local[*] - all available cores.
    ##############################

    # Create SparkSession object
    spark = SparkSession.builder \
        .master('local[*]') \
        .appName('test') \
        .getOrCreate()

    # What version of Spark?
    print(spark.version)
    # note: spark version was 2.4.2 for the course

    # Terminate the cluster (i.e. close connection to Spark)
    spark.stop()  # good practice to stop the connection
    # note: Once you are finished with the cluster, it's a good idea to shut
    # it down, which will free up its resources, making them available for
    # other processes.

    #########################################################################
    # Read data from CSV file
    flights = spark.read.csv('flights.csv',
                             sep=',',
                             header=True,
                             inferSchema=True,
                             nullValue='NA')

    # Get number of records
    print("The data contain %d records." % flights.count())

    # View the first five records
    flights.show(5)

    # Check column data types
    flights.dtypes

    #########################################################################
    # Specify column names and types
    schema = StructType([
        StructField("id", IntegerType()),
        StructField("text", StringType()),
        StructField("label", IntegerType())
    ])

    # Load data from a delimited file
    sms = spark.read.csv('sms.csv', sep=';', header=False, schema=schema)

    # Print schema of DataFrame
    sms.printSchema()

    #########################################################################
    # Remove the 'flight' column
    flights = flights.drop('flight')

    # Number of records with missing 'delay' values
    flights.filter('delay IS NULL').count()

    # Remove records with missing 'delay' values
    flights = flights.filter("delay IS NOT NULL")

    # Remove records with missing values in any column and get the number of
    # remaining rows
    flights = flights.dropna()
    print(flights.count())

    #########################################################################
    # Convert 'mile' to 'km' and drop 'mile' column
    flights_km = flights.withColumn('km', round(flights.mile * 1.60934, 0)) \
        .drop('mile')

    # Create 'label' column indicating whether flight delayed (1) or not (0)
    flights_km = flights_km.withColumn('label', (flights_km.delay >= 15).cast(
        'integer'))

    # Check first five records
    flights_km.show(5)

    #########################################################################
    # Create an indexer
    indexer = StringIndexer(inputCol='carrier', outputCol='carrier_idx')
    # note: string indexer should (almost always) then be put through an
    # encoder, such as a one-hot encoder to create indicator variables. This
    # not done in this section but is later)

    # Indexer identifies categories in the data
    indexer_model = indexer.fit(flights)

    # Indexer creates a new column with numeric index values
    flights_indexed = indexer_model.transform(flights)

    # Repeat the process for the other categorical feature
    flights_indexed = StringIndexer(inputCol='org', outputCol='org_idx').fit(
        flights_indexed).transform(flights_indexed)
    # note: string indexer should (almost always) then be put through an
    # encoder, such as a one-hot encoder to create indicator variables. This
    # not done in this section but is later)

    #########################################################################
    # Create an assembler object
    assembler = VectorAssembler(inputCols=[
        'mon', 'dom', 'dow', 'carrier_idx', 'org_idx', 'km', 'depart',
        'duration'
    ], outputCol='features')

    # Consolidate predictor columns
    flights_assembled = assembler.transform(flights)

    # Check the resulting column
    flights_assembled.select('features', 'delay').show(5, truncate=False)

    #########################################################################
    # Split into training and testing sets in a 80:20 ratio
    flights_train, flights_test = flights.randomSplit([0.8, 0.2], seed=17)

    # Check that training set has around 80% of records
    training_ratio = flights_train.count() / flights.count()
    print(training_ratio)

    #########################################################################
    # Create a classifier object and fit to the training data
    tree = DecisionTreeClassifier()
    tree_model = tree.fit(flights_train)

    # Create predictions for the testing data and take a look at the
    # predictions
    prediction = tree_model.transform(flights_test)
    prediction.select('label', 'prediction', 'probability').show(5, False)

    #########################################################################
    # Create a confusion matrix
    prediction.groupBy('label', 'prediction').count().show()

    # Calculate the elements of the confusion matrix
    TN = prediction.filter('prediction = 0 AND label = prediction').count()
    TP = prediction.filter('prediction = 1 AND label = prediction').count()
    FN = prediction.filter('prediction = 0 AND label != prediction').count()
    FP = prediction.filter('prediction = 1 AND label != prediction').count()

    # Accuracy measures the proportion of correct predictions
    accuracy = (TN + TP) / (TN + TP + FN + FP)
    print(accuracy)

    #########################################################################
    # Create a classifier object and train on training data
    logistic = LogisticRegression().fit(flights_train)

    # Create predictions for the testing data and show confusion matrix
    prediction = logistic.transform(flights_test)
    prediction.groupBy('label', 'prediction').count().show()

    #########################################################################
    # Calculate precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print('precision = {:.2f}\nrecall    = {:.2f}'.format(precision, recall))

    # Find weighted precision
    multi_evaluator = MulticlassClassificationEvaluator()
    weighted_precision = multi_evaluator.evaluate(prediction, {
        multi_evaluator.metricName: "weightedPrecision"})

    # Find AUC
    binary_evaluator = BinaryClassificationEvaluator()
    auc = binary_evaluator.evaluate(prediction, {
        binary_evaluator.metricName: 'areaUnderROC'})

    #########################################################################
    # Remove punctuation (REGEX provided) and numbers
    wrangled = sms.withColumn('text',
                              regexp_replace(sms.text, '[_():;,.!?\\-]', ' '))
    wrangled = wrangled.withColumn('text',
                                   regexp_replace(wrangled.text, '[0-9]', ' '))

    # Merge multiple spaces
    wrangled = wrangled.withColumn('text',
                                   regexp_replace(wrangled.text, ' +', ' '))

    # Split the text into words
    wrangled = Tokenizer(inputCol='text', outputCol='words').transform(
        wrangled)

    wrangled.show(4, truncate=False)

    #########################################################################
    # Remove stop words.
    wrangled = StopWordsRemover(inputCol='words', outputCol='terms') \
        .transform(sms)

    # Apply the hashing trick
    wrangled = HashingTF(inputCol='terms', outputCol='hash', numFeatures=1024) \
        .transform(wrangled)

    # Convert hashed symbols to TF-IDF
    tf_idf = IDF(inputCol='hash', outputCol='features') \
        .fit(wrangled).transform(wrangled)

    tf_idf.select('terms', 'features').show(4, truncate=False)

    #########################################################################
    # Split the data into training and testing sets
    sms_train, sms_test = sms.randomSplit([0.8, 0.2], seed=13)

    # Fit a Logistic Regression model to the training data
    logistic = LogisticRegression(regParam=0.2).fit(sms_train)

    # Make predictions on the testing data
    prediction = logistic.transform(sms_test)

    # Create a confusion matrix, comparing predictions to known labels
    prediction.groupBy('label', 'prediction').count().show()

    #########################################################################
    # Create an instance of the one hot encoder
    onehot = OneHotEncoderEstimator(inputCols=['org_idx'],
                                    outputCols=['org_dummy'])

    # Apply the one hot encoder to the flights data
    onehot = onehot.fit(flights)
    flights_onehot = onehot.transform(flights)

    # Check the results
    flights_onehot.select('org', 'org_idx', 'org_dummy').distinct().sort(
        'org_idx').show()

    #########################################################################
    # Create a regression object and train on training data
    regression = LinearRegression(labelCol='duration').fit(flights_train)

    # Create predictions for the testing data and take a look at the predictions
    predictions = regression.transform(flights_test)
    predictions.select('duration', 'prediction').show(5, False)

    # Calculate the RMSE
    RegressionEvaluator(labelCol='duration').evaluate(predictions)

    #########################################################################
    # Intercept (average minutes on ground)
    inter = regression.intercept
    print(inter)

    # Coefficients
    coefs = regression.coefficients
    print(coefs)

    # Average minutes per km
    minutes_per_km = regression.coefficients[0]
    print(minutes_per_km)

    # Average speed in km per hour
    avg_speed = 1. / (minutes_per_km / 60.)
    print(avg_speed)

    #########################################################################
    # Create a regression object and train on training data
    regression = LinearRegression(labelCol='duration').fit(flights_train)

    # Create predictions for the testing data
    predictions = regression.transform(flights_test)

    # Calculate the RMSE on testing data
    RegressionEvaluator(labelCol='duration').evaluate(predictions)

    #########################################################################
    # Average speed in km per hour
    avg_speed_hour = 60. / regression.coefficients[0]
    print(avg_speed_hour)

    # Average minutes on ground at OGG
    inter = regression.intercept
    print(inter)

    # Average minutes on ground at JFK
    avg_ground_jfk = inter + regression.coefficients[3]
    print(avg_ground_jfk)

    # Average minutes on ground at LGA
    avg_ground_lga = inter + regression.coefficients[4]
    print(avg_ground_lga)

    #########################################################################
    # Create buckets at 3 hour intervals through the day
    buckets = Bucketizer(splits=[0., 3., 6., 9., 12., 15., 18., 21., 24.],
                         inputCol='depart', outputCol='depart_bucket')

    # Bucket the departure times
    bucketed = buckets.transform(flights)
    bucketed.select('depart', 'depart_bucket').show(5)

    # Create a one-hot encoder
    onehot = OneHotEncoderEstimator(inputCols=['depart_bucket'],
                                    outputCols=['depart_dummy'])

    # One-hot encode the bucketed departure times
    flights_onehot = onehot.fit(bucketed).transform(bucketed)
    flights_onehot.select('depart', 'depart_bucket', 'depart_dummy').show(5)

    #########################################################################
    # Find the RMSE on testing data
    RegressionEvaluator(labelCol='duration').evaluate(predictions)

    # Average minutes on ground at OGG for flights departing between 21:00 and
    # 24:00
    avg_eve_ogg = regression.intercept
    print(avg_eve_ogg)

    # Average minutes on ground at OGG for flights departing between 00:00 and
    # 03:00
    avg_night_ogg = regression.intercept + regression.coefficients[8]
    print(avg_night_ogg)

    # Average minutes on ground at JFK for flights departing between 00:00 and
    # 03:00
    avg_night_jfk = regression.intercept + regression.coefficients[3] + \
                    regression.coefficients[8]
    print(avg_night_jfk)

    #########################################################################
    # Fit linear regression model to training data
    regression = LinearRegression(labelCol='duration').fit(flights_train)

    # Make predictions on testing data
    predictions = regression.transform(flights_test)

    # Calculate the RMSE on testing data
    rmse = RegressionEvaluator(labelCol='duration').evaluate(predictions)
    print("The test RMSE is", rmse)

    # Look at the model coefficients
    coeffs = regression.coefficients
    print(coeffs)

    #########################################################################
    # Fit Lasso model (α = 1) to training data
    regression = LinearRegression(labelCol='duration', regParam=1.,
                                  elasticNetParam=1).fit(flights_train)

    # Calculate the RMSE on testing data
    rmse = RegressionEvaluator(labelCol='duration').evaluate(
        regression.transform(flights_test))
    print("The test RMSE is", rmse)

    # Look at the model coefficients
    coeffs = regression.coefficients
    print(coeffs)

    # Number of zero coefficients
    zero_coeff = sum([beta == 0. for beta in regression.coefficients])
    print("Number of ceofficients equal to 0:", zero_coeff)

    #########################################################################
    # Convert categorical strings to index values
    indexer = StringIndexer(inputCol='org', outputCol='org_idx')

    # One-hot encode index values
    onehot = OneHotEncoderEstimator(
        inputCols=['org_idx', 'dow'],
        outputCols=['org_dummy', 'dow_dummy']
    )

    # Assemble predictors into a single column
    assembler = VectorAssembler(inputCols=['km', 'org_dummy', 'dow_dummy'],
                                outputCol='features')

    # A linear regression object
    regression = LinearRegression(labelCol='duration')

    #########################################################################
    # Construct a pipeline
    pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])

    # Train the pipeline on the training data
    pipeline = pipeline.fit(flights_train)

    # Make predictions on the testing data
    predictions = pipeline.transform(flights_test)

    #########################################################################
    # Break text into tokens at non-word characters
    tokenizer = Tokenizer(inputCol='text', outputCol='words')

    # Remove stop words
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(),
                               outputCol='terms')

    # Apply the hashing trick and transform to TF-IDF
    hasher = HashingTF(inputCol=remover.getOutputCol(), outputCol="hash")
    idf = IDF(inputCol=hasher.getOutputCol(), outputCol="features")

    # Create a logistic regression object and add everything to a pipeline
    logistic = LogisticRegression()
    pipeline = Pipeline(stages=[tokenizer, remover, hasher, idf, logistic])

    #########################################################################
    # Create an empty parameter grid
    params = ParamGridBuilder().build()

    # Create objects for building and evaluating a regression model
    regression = LinearRegression(labelCol='duration')
    evaluator = RegressionEvaluator(labelCol='duration')

    # Create a cross validator
    cv = CrossValidator(estimator=regression, estimatorParamMaps=params,
                        evaluator=evaluator, numFolds=5)

    # Train and test model on multiple folds of the training data
    cv = cv.fit(flights_train)

    # NOTE: Since cross-valdiation builds multiple models, the fit() method
    # can take a little while to complete.

    #########################################################################
    # Create an indexer for the org field
    indexer = StringIndexer(inputCol='org', outputCol='org_idx')

    # Create an one-hot encoder for the indexed org field
    onehot = OneHotEncoderEstimator(inputCols=['org_idx'],
                                    outputCols=['org_dummy'])

    # Assemble the km and one-hot encoded fields
    assembler = VectorAssembler(inputCols=['km', 'org_dummy'],
                                outputCol='features')

    # Create a pipeline and cross-validator.
    pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])
    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=params,
                        evaluator=evaluator)

    #########################################################################
    # Create parameter grid
    params = ParamGridBuilder()

    # Add grids for two parameters
    params = params.addGrid(regression.regParam, [0.01, 0.1, 1., 10.]) \
        .addGrid(regression.elasticNetParam, [0., 0.5, 1.])

    # Build the parameter grid
    params = params.build()
    print('Number of models to be tested: ', len(params))

    # Create cross-validator
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params,
                        evaluator=evaluator, numFolds=5)

    #########################################################################
    # Get the best model from cross validation
    best_model = cv.bestModel

    # Look at the stages in the best model
    print(best_model.stages)

    # Get the parameters for the LinearRegression object in the best model
    best_model.stages[3].extractParamMap()

    # Generate predictions on testing data using the best model then calculate
    # RMSE
    predictions = best_model.transform(flights_test)
    evaluator.evaluate(predictions)

    #########################################################################
    # Create parameter grid
    params = ParamGridBuilder()

    # Add grid for hashing trick parameters
    params = params.addGrid(hasher.numFeatures, [1024, 4096, 16384]) \
        .addGrid(hasher.binary, [True, False])

    # Add grid for logistic regression parameters
    params = params.addGrid(logistic.regParam, [0.01, 0.1, 1., 10.]) \
        .addGrid(logistic.elasticNetParam, [0., 0.5, 1.])

    # Build parameter grid
    params = params.build()

    #########################################################################
    # Create model objects and train on training data
    tree = DecisionTreeClassifier().fit(flights_train)
    gbt = GBTClassifier().fit(flights_train)

    # Compare AUC on testing data
    evaluator = BinaryClassificationEvaluator()
    evaluator.evaluate(tree.transform(flights_test))
    evaluator.evaluate(gbt.transform(flights_test))

    # Find the number of trees and the relative importance of features
    print(len(gbt.trees))
    print(gbt.featureImportances)

    #########################################################################
    # Create a random forest classifier
    forest = RandomForestClassifier()

    # Create a parameter grid
    params = ParamGridBuilder() \
        .addGrid(forest.featureSubsetStrategy,
                 ['all', 'onethird', 'sqrt', 'log2']) \
        .addGrid(forest.maxDepth, [2, 5, 10]) \
        .build()

    # Create a binary classification evaluator
    evaluator = BinaryClassificationEvaluator()

    # Create a cross-validator
    cv = CrossValidator(estimator=forest, estimatorParamMaps=params,
                        evaluator=evaluator, numFolds=5)

    #########################################################################
    # Average AUC for each parameter combination in grid
    avg_auc = cv.avgMetrics

    # Average AUC for the best model
    best_model_auc = max(avg_auc)

    # What's the optimal parameter value?
    opt_max_depth = cv.bestModel.explainParam('maxDepth')
    opt_feat_substrat = cv.bestModel.explainParam('featureSubsetStrategy')

    # AUC for best model on testing data
    best_auc = evaluator.evaluate(cv.transform(flights_test))

    #########################################################################

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, max as spark_max

# Initialize the Spark session
spark = SparkSession.builder.appName("GoalPrediction").getOrCreate()

# Path to the CSV file
file_path = "C:/Users/21651/Desktop/goal_shotcreation.csv"

# Load the CSV file, making sure to infer the schema and handle headers correctly
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Show the schema of the dataset to verify the column names and types
data.printSchema()

# Show the first few rows of the dataset to check the content
data.show(20)

# Use _c6 as the goals column
data = data.withColumn("_c6", col("_c6").cast("int"))  # Ensure _c6 is treated as an integer

# Show the _c6 column to confirm the conversion
data.select("_c6").show(20)

# Filter out rows with null or invalid values in _c6
data = data.filter(col("_c6").isNotNull())

# Rename _c6 to goals
data_goals = data.select("_c6").withColumnRenamed("_c6", "goals")

# Show the data with the renamed goals column
data_goals.show(20)

# Assemble the feature vector using goals
assembler = VectorAssembler(inputCols=["goals"], outputCol="features")

# Apply the transformation to the data
data_for_model = assembler.transform(data_goals)

# Show the transformed data with the features column
data_for_model.select("goals", "features").show(20)

# Split the data into training and test sets (80% train, 20% test)
train_data, test_data = data_for_model.randomSplit([0.8, 0.2], seed=1234)

# Check the number of records in each split
print(f"Training Data Count: {train_data.count()}")
print(f"Test Data Count: {test_data.count()}")

# Create and configure the Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="goals")

# Train the model using the training data
model = lr.fit(train_data)

# Evaluate the model on the test data
test_results = model.evaluate(test_data)

# Print the evaluation metrics
print(f"zrreur moyenne des predictions {test_results.rootMeanSquaredError}")
print(f"proportion de variance: {test_results.r2}")

# Optional: Predict the goals for the next match (on the test data)
predictions = model.transform(test_data)
# Show predictions in percentages
print("Predictions ")
predictions.select("goals").show(8)
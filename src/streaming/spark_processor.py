from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json
import logging

logger = logging.getLogger(__name__)

class SparkFraudProcessor:
    """Spark Streaming processor for fraud detection"""
    
    def __init__(self, app_name="fraud-detection", kafka_servers="localhost:9092"):
        self.app_name = app_name
        self.kafka_servers = kafka_servers
        self.spark = self._create_spark_session()
        
    def _create_spark_session(self):
        """Create Spark session with Kafka support"""
        return SparkSession.builder \
            .appName(self.app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
    
    def process_transactions(self, input_topic="transactions", output_topic="fraud_predictions"):
        """Process transaction stream for fraud detection"""
        
        # Define schema
        transaction_schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("merchant_id", StringType(), True),
            StructField("timestamp", StringType(), True),
            StructField("location", MapType(StringType(), DoubleType()), True)
        ])
        
        # Read from Kafka
        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_servers) \
            .option("subscribe", input_topic) \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse JSON and extract features
        transactions = df.select(
            from_json(col("value").cast("string"), transaction_schema).alias("data")
        ).select("data.*")
        
        # Feature engineering
        enriched = transactions.withColumn(
            "hour", hour(to_timestamp(col("timestamp")))
        ).withColumn(
            "is_night", when(col("hour").between(22, 23) | col("hour").between(0, 6), 1).otherwise(0)
        ).withColumn(
            "is_high_amount", when(col("amount") > 1000, 1).otherwise(0)
        )
        
        # Simple fraud scoring (replace with actual model)
        scored = enriched.withColumn(
            "fraud_score",
            when(col("is_night") == 1, 0.3).otherwise(0.1) +
            when(col("is_high_amount") == 1, 0.4).otherwise(0.0)
        ).withColumn(
            "is_fraud", when(col("fraud_score") > 0.5, 1).otherwise(0)
        )
        
        # Write results
        query = scored.selectExpr("to_json(struct(*)) AS value") \
            .writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_servers) \
            .option("topic", output_topic) \
            .option("checkpointLocation", "/tmp/checkpoint") \
            .start()
        
        return query

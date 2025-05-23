from py4j.java_gateway import java_import
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import ArrayType, FloatType
import fitz  # PyMuPDF
import pandas as pd
import os
import requests
import io

# HDFS paths
HDFS_INPUT_PATH = "hdfs:///data"
HDFS_OUTPUT_PATH = "hdfs://hadoop-namenode:9000/user/spark/embeddings"
EMBEDDING_API_URL = "http://embedding-api:9000/embed"

# 1. Spark session
spark = SparkSession.builder \
    .appName("IngestionPipeline") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop-namenode:9000") \
    .getOrCreate()

# 2. HDFS FileSystem handle
sc = spark.sparkContext
java_import(sc._jvm, "org.apache.hadoop.fs.Path")
hadoop_conf = sc._jsc.hadoopConfiguration()
hadoop_conf.set("fs.defaultFS", "hdfs://hadoop-namenode:9000")
fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
input_path = sc._jvm.Path(HDFS_INPUT_PATH)
output_path = sc._jvm.Path(HDFS_OUTPUT_PATH)

# Create output directory if it doesn't exist
if not fs.exists(output_path):
    fs.mkdirs(output_path)

# 3. List files
files = fs.listStatus(input_path)
documents = []

# 4. Process files
for f in files:
    file_path = f.getPath().toString()
    file_name = os.path.basename(file_path)

    if file_name.endswith(".pdf"):
        pdf_stream = fs.open(f.getPath())
        pdf_bytes = pdf_stream.readAllBytes()
        pdf_stream.close()

        with io.BytesIO(pdf_bytes) as pdf_file:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            full_text = ""
            for page in doc:
                blocks = page.get_text("blocks")
                blocks_text = "\n".join([b[4] for b in blocks if isinstance(b[4], str)])
                full_text += blocks_text + "\n"
            documents.append({"content": full_text, "source": file_name})

    elif file_name.endswith(".csv"):
        df = spark.read.option("header", True).csv(file_path)
        schema = str(df.schema)
        sample = df.limit(5).toPandas().to_string(index=False)
        documents.append({
            "content": f"Schema:\n{schema}\n\nSample:\n{sample}",
            "source": file_name
        })

# 5. Create Spark DataFrame
df = spark.createDataFrame(documents)

# 6. Pandas UDF to call embedding service
@pandas_udf(ArrayType(FloatType()))
def embed_batch_udf(texts: pd.Series) -> pd.Series:
    embeddings = []
    for text in texts:
        try:
            res = requests.post(EMBEDDING_API_URL, json={"texts": [text]})
            if res.status_code == 200:
                embedding = res.json().get("embeddings", [[]])[0]
                print("✅ Embedded:", embedding[:5], "...")  # short preview
                embeddings.append(embedding)
            else:
                print("❌ Status code:", res.status_code)
                embeddings.append([])
        except Exception as e:
            print("🚨 Exception during embedding:", e)
            embeddings.append([])
    return pd.Series(embeddings)


# 7. Apply embedding UDF
df_emb = df.withColumn("embedding", embed_batch_udf(col("content")))

# 8. Write output
df_emb.write.mode("overwrite").parquet(HDFS_OUTPUT_PATH)
print("✅ Ingestion complete. Output at:", HDFS_OUTPUT_PATH)

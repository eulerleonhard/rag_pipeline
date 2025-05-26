from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, lit
from pyspark.sql.types import ArrayType, FloatType, StringType, StructType, StructField
import pandas as pd
import fitz  # PyMuPDF
import io
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("IngestionPipeline")

# Constants
HDFS_INPUT_PATH = "hdfs:///data"
HDFS_OUTPUT_PATH = "hdfs://hadoop-namenode:9000/user/spark/embeddings"
EMBEDDING_API_URL = "http://embedding-api:9000/embed"

# Spark session
spark = SparkSession.builder \
    .appName("IngestionPipeline") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop-namenode:9000") \
    .getOrCreate()

# -----------------------------------------------------------
# 1. Read PDFs using binaryFile API
# -----------------------------------------------------------
log.info("📄 Reading PDFs from HDFS...")
pdf_df = spark.read.format("binaryFile") \
    .load(HDFS_INPUT_PATH + "/*.pdf") \
    .select("path", "content")

@pandas_udf(StringType())
def extract_pdf_text_udf(content: pd.Series) -> pd.Series:
    results = []
    for binary in content:
        try:
            with io.BytesIO(binary) as pdf_file:
                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                text = "\n".join(page.get_text("text") for page in doc)
                results.append(text)
        except Exception as e:
            log.error(f"PDF parsing error: {e}")
            results.append("")
    return pd.Series(results)

pdf_text_df = pdf_df.withColumn("content", extract_pdf_text_udf(col("content"))) \
    .withColumnRenamed("path", "source")

# -----------------------------------------------------------
# 2. Read CSVs with schema and preview sample
# -----------------------------------------------------------
log.info("📊 Reading CSVs from HDFS...")
csv_paths = [row.path for row in spark._jvm.org.apache.hadoop.fs.FileSystem \
             .get(spark._jsc.hadoopConfiguration()) \
             .listStatus(spark._jvm.org.apache.hadoop.fs.Path(HDFS_INPUT_PATH))
             if row.getPath().getName().endswith(".csv")]

csv_docs = []
for path in csv_paths:
    file_path = path.toString()
    file_name = file_path.split("/")[-1]
    try:
        df = spark.read.option("header", True).csv(file_path)
        schema_str = df.schema.simpleString()
        sample = df.limit(5).toPandas().to_markdown(index=False)
        csv_docs.append({
            "content": f"Schema: {schema_str}\n\nSample:\n{sample}",
            "source": file_name
        })
    except Exception as e:
        log.error(f"CSV error {file_name}: {e}")

csv_text_df = spark.createDataFrame(csv_docs, schema=StructType([
    StructField("content", StringType(), True),
    StructField("source", StringType(), True),
]))

# -----------------------------------------------------------
# 3. Combine PDF and CSV docs
# -----------------------------------------------------------
log.info("🔗 Combining PDF and CSV documents...")
doc_df = pdf_text_df.select("content", "source").unionByName(csv_text_df)

# -----------------------------------------------------------
# 4. Embedding using REST API
# -----------------------------------------------------------
@pandas_udf(ArrayType(FloatType()))
def embed_batch_udf(texts: pd.Series) -> pd.Series:
    batch = texts.tolist()
    try:
        res = requests.post(EMBEDDING_API_URL, json={"texts": batch})
        if res.status_code == 200:
            return pd.Series(res.json().get("embeddings", [[] for _ in batch]))
        else:
            log.error(f"Embedding API error: {res.status_code}")
            return pd.Series([[] for _ in batch])
    except Exception as e:
        log.error(f"Embedding exception: {e}")
        return pd.Series([[] for _ in batch])

log.info("🧠 Generating embeddings...")
df_emb = doc_df.withColumn("embedding", embed_batch_udf(col("content")))

# -----------------------------------------------------------
# 5. Write to HDFS as Parquet
# -----------------------------------------------------------
log.info(f"💾 Writing embeddings to HDFS: {HDFS_OUTPUT_PATH}")
df_emb.write.mode("overwrite").parquet(HDFS_OUTPUT_PATH)

log.info("✅ Ingestion complete.")

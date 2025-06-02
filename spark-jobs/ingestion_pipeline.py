from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, explode
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, ArrayType, FloatType
import pandas as pd
import fitz  # PyMuPDF
import io
import uuid
import requests
import logging
import hashlib
import struct

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("IngestionPipeline")

# Constants
HDFS_INPUT_PATH = "hdfs:///data"
HDFS_OUTPUT_PATH = "hdfs://hadoop-namenode:9000/user/spark/embeddings"
EMBEDDING_API_URL = "http://embedding-api:9000/embed"

# Sliding window config
CHUNK_SIZE = 200   # number of words per chunk
CHUNK_OVERLAP = 50 # number of words to overlap

# Spark session
spark = SparkSession.builder \
    .appName("IngestionPipeline") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://hadoop-namenode:9000") \
    .getOrCreate()

def uuid_to_int64(u: str) -> int:
    """
    Convert UUID string to a signed 64-bit integer by hashing.
    This ensures Spark LongType compatibility.
    """
    h = hashlib.sha256(u.encode('utf-8')).digest()
    unsigned_val = struct.unpack('>Q', h[:8])[0]
    if unsigned_val >= 2**63:
        return unsigned_val - 2**64
    else:
        return unsigned_val

# -----------------------------------------------------------
# 1. Read PDFs from HDFS
# -----------------------------------------------------------
log.info("📄 Reading PDFs from HDFS...")
pdf_df = spark.read.format("binaryFile") \
    .load(HDFS_INPUT_PATH + "/*.pdf") \
    .select("path", "content")

# -----------------------------------------------------------
# 2. Extract and chunk PDF text using sliding window
# -----------------------------------------------------------
@pandas_udf(ArrayType(StructType([
    StructField("chunk_id", StringType()),
    StructField("chunk_int_id", LongType()),               # New stable int64 ID
    StructField("source", StringType()),
    StructField("page_num", IntegerType()),
    StructField("chunk_index", IntegerType()),
    StructField("content", StringType())
])))
def extract_chunks_udf(contents: pd.Series, paths: pd.Series) -> pd.Series:
    all_chunks = []
    for binary, path in zip(contents, paths):
        chunks = []
        try:
            with io.BytesIO(binary) as pdf_file:
                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text("text")
                    words = text.split()
                    i = 0
                    chunk_index = 0
                    while i < len(words):
                        chunk_words = words[i:i + CHUNK_SIZE]
                        chunk_text = " ".join(chunk_words)
                        if chunk_text.strip():
                            cid = str(uuid.uuid4())
                            chunks.append({
                                "chunk_id": cid,
                                "chunk_int_id": uuid_to_int64(cid),  # Stable int64 ID
                                "source": path.split("/")[-1],
                                "page_num": page_num,
                                "chunk_index": chunk_index,
                                "content": chunk_text
                            })
                            chunk_index += 1
                        i += CHUNK_SIZE - CHUNK_OVERLAP
        except Exception as e:
            log.error(f"Failed to parse PDF: {e}")
        all_chunks.append(chunks)
    return pd.Series(all_chunks)

log.info("✂️ Extracting and chunking PDF content...")
chunked_df = pdf_df.withColumn("chunks", extract_chunks_udf(col("content"), col("path"))) \
    .select(explode(col("chunks")).alias("chunk")) \
    .select(
        col("chunk.chunk_id"),
        col("chunk.chunk_int_id"),              # include int64 ID
        col("chunk.source"),
        col("chunk.page_num"),
        col("chunk.chunk_index"),
        col("chunk.content")
    )

# -----------------------------------------------------------
# 3. Embedding using external API
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
        log.error(f"Embedding request failed: {e}")
        return pd.Series([[] for _ in batch])

log.info("🧠 Generating embeddings for text chunks...")
df_emb = chunked_df.withColumn("embedding", embed_batch_udf(col("content")))

# -----------------------------------------------------------
# 4. Write to HDFS
# -----------------------------------------------------------
log.info(f"💾 Writing enriched embeddings to HDFS: {HDFS_OUTPUT_PATH}")
df_emb.write.mode("overwrite").parquet(HDFS_OUTPUT_PATH)

log.info("✅ Ingestion pipeline complete.")
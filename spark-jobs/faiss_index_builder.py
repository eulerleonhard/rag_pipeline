import faiss
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType
import os

# Config
HDFS_PARQUET_PATH = "hdfs://hadoop-namenode:9000/user/spark/embeddings"
FAISS_INDEX_PATH = "/app/faiss_index/index.faiss"  # local path inside spark container
METADATA_PATH = "/app/faiss_index/metadata.parquet"

spark = SparkSession.builder.appName("FaissIndexBuilder").getOrCreate()

def main():
    # Load enhanced embeddings parquet (chunk_id, source, page_num, chunk_index, content, embedding)
    df = spark.read.parquet(HDFS_PARQUET_PATH).select(
        "chunk_id", "source", "page_num", "chunk_index", "content", "embedding"
    )

    # Collect embeddings and metadata locally (assumes manageable size)
    data = df.collect()
    embeddings = np.array([row.embedding for row in data]).astype('float32')
    metadata = [{
        "chunk_id": row.chunk_id,
        "source": row.source,
        "page_num": row.page_num,
        "chunk_index": row.chunk_index,
        "content": row.content
    } for row in data]

    # Ensure target directory exists
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    # Build FAISS index (Flat L2)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index to disk
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save metadata with Spark for retrieval
    meta_schema = StructType([
        StructField("chunk_id", StringType()),
        StructField("source", StringType()),
        StructField("page_num", IntegerType()),
        StructField("chunk_index", IntegerType()),
        StructField("content", StringType()),
    ])
    meta_df = spark.createDataFrame(metadata, schema=meta_schema)
    meta_df.write.mode("overwrite").parquet(METADATA_PATH)

    print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")

if __name__ == "__main__":
    main()
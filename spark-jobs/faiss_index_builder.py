import faiss
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
import os

# Config
HDFS_PARQUET_PATH = "hdfs://hadoop-namenode:9000/user/spark/embeddings"
FAISS_INDEX_PATH = "/app/faiss_index/index.faiss"  # local path inside spark container
METADATA_PATH = "/app/faiss_index/metadata.parquet"

spark = SparkSession.builder.appName("FaissIndexBuilder").getOrCreate()

def main():
    # Load embeddings parquet including chunk_int_id
    df = spark.read.parquet(HDFS_PARQUET_PATH).select(
        "chunk_id", "chunk_int_id", "source", "page_num", "chunk_index", "content", "embedding"
    )

    data = df.collect()

    # Extract embeddings and IDs
    embeddings = np.array([row.embedding for row in data]).astype('float32')
    ids = np.array([row.chunk_int_id for row in data]).astype('int64')

    # Extract metadata (include chunk_int_id)
    metadata = [{
        "chunk_id": row.chunk_id,
        "chunk_int_id": row.chunk_int_id,
        "source": row.source,
        "page_num": row.page_num,
        "chunk_index": row.chunk_index,
        "content": row.content
    } for row in data]

    # Ensure target directory exists
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    dim = embeddings.shape[1]

    # Load existing index if available, else create new
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        # Wrap with ID map if not already
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)
    else:
        flat_index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(flat_index)

    # Add new vectors with their IDs
    index.add_with_ids(embeddings, ids)

    # Save updated index
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save metadata including chunk_int_id
    meta_schema = StructType([
        StructField("chunk_id", StringType()),
        StructField("chunk_int_id", LongType()),
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
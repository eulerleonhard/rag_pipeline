import os
import fitz  # PyMuPDF
import pandas as pd
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, ArrayType, StructType, StructField
from pyspark.sql import Row
from py4j.java_gateway import java_import

# Initialize Spark
spark = SparkSession.builder.appName("ResearchIngestionPipeline").getOrCreate()
sc = spark.sparkContext

# HDFS config
input_dir = "hdfs:///data/"
output_dir = "hdfs:///embeddings/"

# Ensure /embeddings exists
java_import(sc._gateway.jvm, "org.apache.hadoop.fs.FileSystem")
java_import(sc._gateway.jvm, "org.apache.hadoop.fs.Path")
fs = sc._jvm.FileSystem.get(sc._jsc.hadoopConfiguration())
embeddings_path = sc._jvm.Path(output_dir)
if not fs.exists(embeddings_path):
    fs.mkdirs(embeddings_path)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_pdf_content(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        content_blocks = []
        for page in doc:
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4].strip()
                if len(text) > 30:
                    content_blocks.append(text)
        return content_blocks
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return []

def extract_csv_content(csv_path):
    try:
        df = pd.read_csv(csv_path)
        schema = str(df.dtypes.to_dict())
        samples = df.sample(min(3, len(df))).to_dict(orient="records")
        samples_text = [str(sample) for sample in samples]
        return [schema] + samples_text
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return []

def process_file(hdfs_path):
    local_tmp_path = f"/tmp/{os.path.basename(hdfs_path)}"
    # Download file from HDFS
    os.system(f"hdfs dfs -copyToLocal {hdfs_path} {local_tmp_path}")
    
    if local_tmp_path.lower().endswith(".pdf"):
        chunks = extract_pdf_content(local_tmp_path)
    elif local_tmp_path.lower().endswith(".csv"):
        chunks = extract_csv_content(local_tmp_path)
    else:
        print(f"Skipping unsupported file type: {local_tmp_path}")
        return []
    
    # Generate embeddings
    if not chunks:
        return []
    
    embeddings = model.encode(chunks, batch_size=16, show_progress_bar=False)
    
    results = []
    for text, vec in zip(chunks, embeddings):
        results.append(Row(content=text, source=os.path.basename(hdfs_path), embedding=list(map(float, vec))))
    
    return results

# List files in /data
input_files = fs.listStatus(sc._jvm.Path(input_dir))
all_rows = []

for file_status in input_files:
    path = file_status.getPath().toString()
    rows = process_file(path)
    all_rows.extend(rows)

# Convert to Spark DataFrame
if all_rows:
    schema = StructType([
        StructField("content", StringType(), True),
        StructField("source", StringType(), True),
        StructField("embedding", ArrayType(StringType()), True)
    ])
    df = spark.createDataFrame(all_rows, schema=schema)
    df.write.mode("overwrite").parquet(output_dir + "/embeddings.parquet")
    print(f"[✓] Ingestion complete. Output saved to: {output_dir}")
else:
    print("[!] No data processed.")
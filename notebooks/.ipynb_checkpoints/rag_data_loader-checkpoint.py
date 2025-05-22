from pyspark.sql import SparkSession
import pdfplumber
from pyspark.sql import Row

spark = SparkSession.builder.getOrCreate()

# TXT example (from HDFS)
txt_df = spark.read.text("hdfs://hadoop-namenode:9000/data/sample.txt")
txt_df.show()

# # CSV example
# csv_df = spark.read.option("header", "true").csv("hdfs://hadoop-namenode:9000/data/sample.csv")
# csv_df.show()

# # PDF example (requires PDF parsing)
# # Step 1: Extract text from PDF using PyPDF2 or pdfplumber

# pdf_path = "/hdfs_data/sample.pdf"  # Local copy
# text = ""
# with pdfplumber.open(pdf_path) as pdf:
#     for page in pdf.pages:
#         text += page.extract_text() + "\n"

# # Step 2: Convert to DataFrame for Spark processing

# lines = text.splitlines()
# pdf_df = spark.createDataFrame([Row(line=l) for l in lines])
# pdf_df.show()
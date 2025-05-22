from fastapi import FastAPI, UploadFile, File
from hdfs import InsecureClient
import os

# WebHDFS client — match the host/port in your docker-compose
hdfs_url = os.getenv("WEBHDFS_URL", "http://hadoop-namenode:9870")
hdfs_client = InsecureClient(hdfs_url, user='root')

app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    hdfs_path = f"/data/{file.filename}"
    with hdfs_client.write(hdfs_path, overwrite=True) as writer:
        writer.write(contents)
    return {"filename": file.filename, "hdfs_path": hdfs_path}

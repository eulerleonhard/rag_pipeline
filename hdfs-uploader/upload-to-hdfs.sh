#!/bin/bash

set -e

echo "[Uploader] Waiting for HDFS to be ready..."
until hdfs dfs -ls / >/dev/null 2>&1; do
    sleep 2
done

echo "[Uploader] HDFS is ready. Syncing files from /hdfs_data to HDFS at /data/..."

hdfs dfs -mkdir -p /data

for file in /hdfs_data/*; do
    if [ -f "$file" ]; then
        echo "[Uploader] Uploading $(basename "$file") to HDFS..."
        hdfs dfs -put -f "$file" /data/
    fi
done

echo "[Uploader] Final file list in HDFS /data:"
hdfs dfs -ls /data

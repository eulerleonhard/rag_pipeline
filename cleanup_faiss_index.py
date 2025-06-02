import os
import glob
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Cleanup")

def cleanup_index_folder(folder_path):
    # Patterns of files to remove
    patterns = ["*.faiss", "*.parquet", "*.lock", "*.tmp"]
    
    files_removed = 0
    for pattern in patterns:
        files = glob.glob(os.path.join(folder_path, pattern))
        for f in files:
            try:
                os.remove(f)
                log.info(f"Removed old file: {f}")
                files_removed += 1
            except Exception as e:
                log.error(f"Failed to remove {f}: {e}")
    if files_removed == 0:
        log.info("No old files found to remove.")
    else:
        log.info(f"Cleanup complete: {files_removed} files removed.")

if __name__ == "__main__":
    # Adjust this to your bind mount local path
    INDEX_FOLDER = "./retrieval-api/faiss_index"
    cleanup_index_folder(INDEX_FOLDER)
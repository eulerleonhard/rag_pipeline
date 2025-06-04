FROM bitnami/spark:3.5.0

USER root

# Install Python dependencies
# RUN pip install --no-cache-dir pandas requests PyMuPDF
RUN apt-get update && \
    apt-get install -y python3-pip libglib2.0-0 && \
    pip3 install --no-cache-dir \
        pandas \
        pyarrow \
        PyMuPDF \
        requests \
        faiss-cpu

USER 1000
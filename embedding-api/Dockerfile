FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install numpy==1.26.4 && \
    pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install sentence-transformers fastapi uvicorn

CMD ["python", "main.py"]

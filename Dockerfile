FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface

COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

COPY demo_server.py .
COPY score_candidates.py .
COPY prepare_external_benchmark.py .
COPY web ./web

EXPOSE 7860

CMD ["sh", "-c", "python demo_server.py --host 0.0.0.0 --port ${PORT:-7860} --embedding-device cpu"]

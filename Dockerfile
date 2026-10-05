FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PADDLEOCR_HOME=/models

# System deps for opencv-headless / paddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 5004

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5004"]

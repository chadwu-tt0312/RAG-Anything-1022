FROM python:3.11-slim

# System deps: common runtime libs for opencv/Pillow, plus build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libglib2.0-0 \
    libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install core deps first (better layer cache)
COPY requirements.txt /app/requirements.txt
COPY eval/integration/requirements-integration.txt /app/requirements-integration.txt
RUN pip install --no-cache-dir -r /app/requirements.txt -r /app/requirements-integration.txt

# Install project
COPY . /app
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "eval.integration.api_service_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]

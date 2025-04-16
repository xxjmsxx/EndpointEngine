FROM python:3.12-slim

WORKDIR /app

# System dependencies for pandas/openpyxl/Excel handling
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# No need to preload SentenceTransformer weights since we're using API calls now

# Copy project files
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Start the API
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]

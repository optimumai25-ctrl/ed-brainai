FROM python:3.11-slim

# System deps (FAISS, PDF, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libopenblas-dev wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Environment defaults (Railway injects $PORT)
ENV PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Make sure boot.sh is executable (used by UI)
RUN if [ -f /app/boot.sh ]; then chmod +x /app/boot.sh; fi

# No CMD here — we will set per-service Start Command in Railway

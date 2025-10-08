FROM python:3.11-slim

# System deps for PyMuPDF/FAISS on Debian "trixie"
# - libgl1 replaces the old libgl1-mesa-glx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080
# Keep shell form so $PORT is expanded by the shell in Cloud Run
CMD streamlit run chat_ceo.py --server.port=$PORT --server.address=0.0.0.0

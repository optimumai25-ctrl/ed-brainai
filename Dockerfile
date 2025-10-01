FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libopenblas-dev wget supervisor \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Env defaults
ENV PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Ensure boot.sh is executable (used by UI)
RUN if [ -f /app/boot.sh ]; then chmod +x /app/boot.sh; fi

# Supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Start both API and UI using supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

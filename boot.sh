#!/usr/bin/env bash
set -e

mkdir -p /app/.streamlit

# If provided via env, render Streamlit secrets TOML
if [ -n "$APP_USER" ] || [ -n "$APP_PASS" ] || [ -n "$GDRIVE_JSON" ] || [ -n "$SHARED_DRIVE_ID" ]; then
  # Escape backslashes in JSON to avoid TOML parse errors
  SAFE_JSON=$(printf "%s" "$GDRIVE_JSON" | sed 's/\\/\\\\/g')
  cat > /app/.streamlit/secrets.toml <<TOML
app_user = "${APP_USER}"
app_pass = "${APP_PASS}"

[gdrive]
json = """$SAFE_JSON"""
shared_drive_id = "${SHARED_DRIVE_ID}"
TOML
fi

# Launch Streamlit UI
exec streamlit run chat_ceo.py --server.address 0.0.0.0 --server.port "${PORT:-8080}"

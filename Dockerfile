FROM python:3.11-slim

# --- Install system dependencies ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# --- Set working directory ---
WORKDIR /app

# --- Copy and install Python dependencies ---
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy FastAPI app ---
COPY voice_api/ /app/voice_api/

# --- Expose the port for Render ---
EXPOSE 7860

# --- Run the dependency setup and start FastAPI ---
CMD python -c "from voice_api import main; main.setup_dependencies()" && \
    uvicorn voice_api.main:app --host 0.0.0.0 --port 7860 --workers 1

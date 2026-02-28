FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Python deps (minimal — no streamlit/moviepy)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# App code
COPY server.py utils.py ./

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

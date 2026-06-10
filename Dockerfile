# CPU-only image for the Mythos inference server.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# install the CPU build of torch explicitly so no CUDA wheels are pulled
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# artifacts/ is produced by the pipeline; mount it or bake it in.
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

# bind to 0.0.0.0 inside the container so the gateway can reach it
CMD ["uvicorn", "mythos.serving.server:app", "--host", "0.0.0.0", "--port", "8000"]

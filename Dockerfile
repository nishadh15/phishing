FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# install packages first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# only what the API needs (see .dockerignore)
COPY src/ src/
COPY dataset/phishing_domains_1000.csv dataset/
COPY models/model.pkl models/

# run as a normal user, not root; /app must be writable for MLflow when /train is called
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Render sets PORT itself (default 10000); locally it falls back to 10000
ENV PORT=10000
EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/openapi.json')"

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]

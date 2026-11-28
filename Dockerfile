FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STT_PROVIDER=cloud \
    TTS_PROVIDER=cloud \
    VAD_PROVIDER=energy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "numpy>=1.26"

COPY app ./app
COPY frontend ./frontend
RUN mkdir -p /app/logs && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

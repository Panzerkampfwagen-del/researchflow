# Stage 1 — build dependencies into an isolated prefix
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
COPY vendor/ vendor/
RUN pip install --no-cache-dir --no-index --find-links=vendor/ --prefix=/install -r requirements.txt

# Stage 2 — minimal runtime
FROM python:3.11-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

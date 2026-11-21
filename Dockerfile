FROM python:3.13-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . && useradd --create-home app && mkdir -p /app/data && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "caseworker.api:app", "--host", "0.0.0.0", "--port", "8000"]

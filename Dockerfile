FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml .
RUN pip install --no-cache-dir -r requirements.txt && pip install -e .

COPY . .

CMD ["uvicorn", "src.ai_congress.api.main:app", "--host", "0.0.0.0", "--port", "8100"]

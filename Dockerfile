# Phase 5, Step 3: containerize the API service.
#
# python:3.12-slim is a minimal base image -- Python plus just enough OS
# to run it, not a full desktop Linux install. Keeps the final image small.
FROM python:3.12-slim

WORKDIR /app

# Copy requirements FIRST, before the rest of the code. Docker caches each
# step -- if only your .py files change (not requirements.txt), Docker
# reuses the cached "pip install" step instead of re-running it, making
# rebuilds much faster during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# deployment/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose the port
EXPOSE ${PORT}

# Run the application
CMD uvicorn api.app:app --host 0.0.0.0 --port ${PORT}

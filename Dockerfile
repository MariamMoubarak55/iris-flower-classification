# Dockerfile for Iris Classification API (Slide 21)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code, tests, and entry points
COPY src/ ./src/
COPY tests/ ./tests/
COPY run.py .
COPY run_app.py .

# Run tests during build to ensure reliability (Slide 21)
RUN pytest tests/

# Expose API port
EXPOSE 5000

# Run Flask application with unbuffered output
CMD ["python", "-u", "run_app.py"]

# SageMaker-compatible Docker image for psoriasis flare prediction

FROM python:3.10-slim

# Set working directory
WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference code
COPY inference.py .

# Set environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# SageMaker uses port 8080 for serving
EXPOSE 8080

# Copy entrypoint script
COPY dockerd-entrypoint.py .

# Entry point for SageMaker
ENTRYPOINT ["python", "dockerd-entrypoint.py"]

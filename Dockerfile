FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ .

# Create directory for output images
RUN mkdir -p etl9g_images

# Create directory for ETL9G data
RUN mkdir -p ETL9G

# Set the default command
CMD ["python", "parse_sample.py"] 
# Use Python 3.9 slim image as base
FROM python:3.9-slim



# Install system dependencies, including those required for RDKit
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
    

# Copy application code and models
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]

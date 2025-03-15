# Use an official Python runtime as a parent image
FROM python:3.13.1-slim

# Set the working directory
WORKDIR /app

# Install system dependencies including OpenGL libraries
RUN apt-get update && apt-get install -y \
    ca-certificates \
    openssl \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy the requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port your FastAPI app runs on
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
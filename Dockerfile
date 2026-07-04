# Use an official Python lightweight image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (optimizes Docker build cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the port Uvicorn will use
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

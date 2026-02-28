FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway/Render inject PORT env var)
EXPOSE 8000

# Start the server
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

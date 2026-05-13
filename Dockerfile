FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command: run in scan-only mode
CMD ["python", "main.py", "--mode", "scan-only", "--exchange", "binance"]

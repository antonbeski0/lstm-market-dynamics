FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy full project
COPY . /app

# Default command
CMD ["python", "-m", "src.lstm_train"]

# Use the official Python 3.12 slim runtime
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Upgrade pip to handle newer wheel formats properly
RUN pip install --no-cache-dir --upgrade pip

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the merged model into the container
COPY main.py .
COPY merged_model/ ./merged_model/

# Expose port 8000 for the API
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
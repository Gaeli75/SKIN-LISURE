# Use official Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app and model into the container
COPY . .

# Set Streamlit environment variables
ENV STREAMLIT_SERVER_PORT=8000
ENV STREAMLIT_SERVER_ENABLECORS=false

# Expose the port the app runs on
EXPOSE 8000

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.enableCORS=false"]

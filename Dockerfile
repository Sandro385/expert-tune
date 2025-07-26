# Use a slim Python base image.  A slim variant keeps the image small
FROM python:3.11-slim

# Install system packages required for certain Python libraries.
# We install git so that pip can install packages directly from GitHub
# (unsloth is installed via a git URL) and build-essential to compile
# any wheels that require compilation.
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the dependency specification and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Configure Streamlit to listen on the standard port and address.
# These environment variables are honoured by Streamlit at runtime.
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start the Streamlit application.  When running on Render, the
# external PORT will be mapped to 8501 via the `docker run -p $PORT:8501` command.
CMD ["streamlit", "run", "app.py"]
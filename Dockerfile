# Use a lightweight Python base image suitable for running Streamlit
FROM python:3.11-slim AS base

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port Render will connect to. Render automatically sets a
# PORT environment variable; Streamlit should bind to that port and
# 0.0.0.0 so that it is reachable externally. We choose 10000 as a
# default but Streamlit will override it at runtime via the $PORT env.
EXPOSE 10000

# Run the Streamlit application. We use bash -c so that the $PORT
# environment variable injected by Render is expanded at runtime. If
# PORT is unset locally, Streamlit will default to 8501.
CMD ["bash", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
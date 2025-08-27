FROM python:3.12

# Install system dependencies required to build dgl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and set working directory
RUN pip install --upgrade pip
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -f https://data.dgl.ai/wheels/torch-2.3/repo.html


# Copy application code and expose port
COPY . .
EXPOSE 7861

# Specify command to run the app
CMD ["python", "main.py"]

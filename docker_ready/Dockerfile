# Python 3.13.2 on Debian slim (good wheels support for numpy/scipy)
FROM python:3.13.2-slim

# Keep Python from writing .pyc files & bufferless logs; make matplotlib write config to a writable dir
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# System packages commonly required by numpy/scipy/matplotlib wheels
# (build-essential & gfortran are a safety net if any wheel falls back to build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libstdc++6 \
    libpng16-16 \
    libfreetype6 \
    fonts-dejavu-core \
    tzdata \
    ca-certificates \
    curl \
    gnupg \
    apt-transport-https \
    tzdata \
    unixodbc \
    unixodbc-dev \
 && rm -rf /var/lib/apt/lists/*

# Microsoft repo + ODBC Driver 17 for SQL Server
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
      | gpg --dearmor -o /usr/share/keyrings/microsoft.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
      > /etc/apt/sources.list.d/microsoft-prod.list \
 && apt-get update \
 && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 \
 && rm -rf /var/lib/apt/lists/*
 
# Workdir
WORKDIR /app

ENV TZ=Asia/Singapore

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip & install deps
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy your application code last (if you have any)
COPY . .

# Default command (adjust to your entrypoint)
CMD ["python", "run_pipeline.py"]

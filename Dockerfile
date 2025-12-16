    # Base image
    FROM python:3.11-slim

    # Set working directory
    WORKDIR /app

    # Install system dependencies
    # libgl1 and libglib2.0-0 are required for OpenCV
    RUN apt-get update && apt-get install -y \
        libgl1 \
        libglib2.0-0 \
        gcc \
        g++ \
        git \
        libxext6 \
        libxrender1 \
        libxcb-shape0 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxkbcommon-x11-0 \
        libxcb-cursor0 \
        libsm6 \
        && rm -rf /var/lib/apt/lists/*

    # Copy requirements file
    COPY requirements.txt .

    # Install Python dependencies
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy application code
    COPY . .

    # Default command (can be overridden)
    CMD ["/bin/bash"]

# Use CUDA-enabled Python 3.9 image as base
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Install Python 3.9
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    gzip \
    tar \
    git \
    build-essential \
    cmake \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install additional tools for data processing
RUN apt-get update && apt-get install -y \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install CUDA-compatible PyTorch first (as specified in README)
RUN pip install --no-cache-dir torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install numpy and other basic dependencies first
RUN pip install --no-cache-dir numpy scipy

# Install requirements with fallback for problematic packages
RUN pip install --no-cache-dir -r requirements.txt || \
    (pip install --no-cache-dir --only-binary=pyarrow,faiss-cpu -r requirements.txt || \
     pip install --no-cache-dir --no-binary=pyarrow,faiss-cpu -r requirements.txt)

# Download and install spaCy English model
RUN python -m spacy download en_core_web_sm

# Create directories for data and results
RUN mkdir -p /app/data/dpr \
    && mkdir -p /app/result \
    && mkdir -p /app/sgpt/encode_result

CMD ["/bin/bash"]

FROM anx/processors/baseprocessor

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ADD requirements.txt ./

RUN pip install -r requirements.txt

ADD processor.py ./
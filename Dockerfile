FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg git libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
# PyTorch под CUDA 12.1
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY src/api ./api
EXPOSE 8000
ENV DEVICE=auto
CMD ["uvicorn", "api.app:core", "--host", "0.0.0.0", "--port", "8000"]

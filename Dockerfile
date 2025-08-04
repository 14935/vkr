FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg git
WORKDIR /srv
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY src/api ./api
EXPOSE 8000
CMD ["uvicorn", "api.app:core", "--host", "0.0.0.0", "--port", "8000"]

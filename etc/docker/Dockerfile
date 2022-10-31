FROM tensorflow/tensorflow:2.2.3-gpu-py3 AS build

# install requirements first. it will speed up last stage
COPY requirements.txt .
RUN pip3 install -r requirements.txt && \
    pip3 install --upgrade protobuf

## Necessary dependencies to run OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx

WORKDIR /opt/is-weapons-detector
COPY src src/
COPY setup.py .
COPY etc/conf/options.json conf/options.json
RUN pip3 install . && \
    pip3 install --upgrade protobuf

CMD ["is-weapons-detector"]
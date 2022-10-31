FROM python:3.6-slim as package_step

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender-dev libglib2.0-0

RUN pip install requests

WORKDIR /opt/intelbras
COPY ./requirements.txt .
COPY ./src ./src

RUN pip install requests
RUN pip install -r requirements.txt
RUN pip install --upgrade protobuf

WORKDIR /opt/intelbras/src
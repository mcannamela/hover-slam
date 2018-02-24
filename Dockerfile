FROM tensorflow/tensorflow:latest-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3-tk
RUN pip install bokeh keras

RUN mkdir /work
WORKDIR /work
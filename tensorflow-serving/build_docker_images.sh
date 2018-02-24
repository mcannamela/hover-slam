#!/usr/bin/env bash
docker build -t $USER/tensorflow-serving-mine -f Dockerfile.mine .
docker build -t $USER/tensorflow-serving-client -f Dockerfile.client .
docker pull tensorflow/tensorflow:latest-py3

#!/usr/bin/env bash

docker build -t $USER/tensorflow -f Dockerfile .
#docker build -t $USER/tensorflow-gui -f Dockerfile.gui .
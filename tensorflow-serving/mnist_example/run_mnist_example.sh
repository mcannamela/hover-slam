#!/usr/bin/env bash

docker run \
--mount type=bind,source="$(pwd)",target=/work \
-w="/work" \
-it $USER/tensorflow-serving-client python mnist_saved_model.py output

docker network create mnist-example


docker run \
    --name tf-serving-mnist \
    --network mnist-example \
    --mount type=bind,source="$(pwd)"/output,target=/output \
    -dt $USER/tensorflow-serving-mine tensorflow_model_server \
        --port=9000  \
        --model_base_path=/output \
        --model_name=mnist

docker run \
    --name tf-mnist-client \
    --network mnist-example \
    --mount type=bind,source="$(pwd)",target=/work \
    -it $USER/tensorflow-serving-client \
    python mnist_client.py \
        --num_tests=1000 \
        --server=tf-serving-mnist:9000

docker container stop tf-serving-mnist
docker container rm tf-serving-mnist
docker container stop tf-mnist-client
docker container rm tf-mnist-client


#!/usr/bin/env bash

#!/usr/bin/env bash

docker run \
    --name gen-chess-prop \
    --mount type=bind,source="$(pwd)",target=/work \
    -dt $USER/tensorflow-serving-client python generate_stubs.py

docker network create grpc-example

docker run \
    --name chess-prop \
    --network grpc-example \
    --mount type=bind,source="$(pwd)",target=/work \
    -dt $USER/tensorflow-serving-client python server.py

docker run \
    --name chess-prop-client \
    --network grpc-example \
    --mount type=bind,source="$(pwd)",target=/work \
    -it $USER/tensorflow-serving-client python client.py

#docker container stop chess-prop
#docker container rm chess-prop
#docker container stop chess-prop-client
#docker container rm chess-prop-client
docker container stop gen-chess-prop
docker container rm gen-chess-prop


#!/usr/bin/env bash

docker container stop chess-prop
docker container rm chess-prop

docker container stop chess-prop-client
docker container rm chess-prop-client

docker container stop gen-chess-prop
docker container rm gen-chess-prop
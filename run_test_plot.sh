#!/usr/bin/env bash

docker run \
--user=$(id -u) \
   --env="DISPLAY" \
   -v "/etc/group:/etc/group:ro" \
   -v "/etc/passwd:/etc/passwd:ro" \
   -v "/etc/shadow:/etc/shadow:ro" \
   -v "/etc/sudoers.d:/etc/sudoers.d:ro" \
   -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   -v "$(pwd):/work" -it $USER/tensorflow python test_plot.py

#docker run --user=$(id -u) \
#--env="DISPLAY" \
#--mount type=bind,source="$(pwd)",target=/work \
#-it $USER/tensorflow-gui python test_plot.py
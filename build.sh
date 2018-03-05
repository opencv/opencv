#!/bin/sh
nvidia-docker  build -t opencv-build:latest -f Dockerfile .

nvidia-docker run -it  -u $(id -u):$(id -g)  -v $PWD/packages:/packages opencv-build  /bin/bash

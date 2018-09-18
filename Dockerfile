FROM ubuntu:18.04
RUN apt update && apt upgrade -y
RUN apt install wget git build-essential pkg-config cmake -y
WORKDIR /root
RUN git clone https://github.com/opencv/opencv.git --depth=1 && git clone https://github.com/opencv/opencv_contrib.git --depth=1
RUN mkdir opencv/build
WORKDIR /root/opencv/build
RUN cmake ../ && make && make install && pkg-config --modversion opencv
RUN /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' && ldconfig
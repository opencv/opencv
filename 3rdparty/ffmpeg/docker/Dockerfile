FROM ubuntu:14.04

RUN \
  apt-get update && \
  apt-get install -y \
    build-essential pkg-config \
    software-properties-common \
    curl git man mc vim nano \
    && \
  rm -rf /var/lib/apt/lists/*

RUN \
  apt-get update && \
  apt-get install -y \
    texinfo yasm bison subversion cvs m4 flex pax \
    && \
  rm -rf /var/lib/apt/lists/*

RUN \
  apt-get update && \
  apt-get install -y \
    mingw-w64 mingw32 g++-mingw-w64-x86-64 g++-mingw-w64-i686 yasm nasm \
    && \
  rm -rf /var/lib/apt/lists/*

CMD ["/app/docker/entry.sh"]

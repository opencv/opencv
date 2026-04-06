#!/bin/bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DWITH_TBB=ON \       # enables TBB for parallel_for_
      ..
make -j$(nproc)

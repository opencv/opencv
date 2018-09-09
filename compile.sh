#!/bin/sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DPYTHON3_EXECUTABLE=/anaconda3/bin/python3 \
    -DPYTHON_INCLUDE_DIR=/anaconda3/include/python3.6m \
    -DPYTHON3_LIBRARY=/anaconda3/lib/libpython3.6m.dylib \
    -DPYTHON3_NUMPY_INCLUDE_DIRS=/anaconda3/lib/python3.6/site-packages/numpy/core/include \
    -DPYTHON3_PACKAGES_PATH=/anaconda3/lib/python3.6/site-packages \
    -DWITH_OPENCL=ON  ..

make install
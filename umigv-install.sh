######################################
# INSTALL OPENCV ON UBUNTU OR DEBIAN #
######################################

#Adapted from original by Nico Ramirez

# 1. KEEP UBUNTU OR DEBIAN UP TO DATE

sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y autoremove


# 2. INSTALL THE DEPENDENCIES

# Build tools:
sudo apt-get install -y build-essential cmake

# GUI (if you want to use GTK instead of Qt, replace 'qt5-default' with 'libgtkglext1-dev' and remove '-DWITH_QT=ON' option in CMake):
sudo apt-get install -y qt5-default libvtk6-dev

# Media I/O:
sudo apt-get install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libjasper-dev libopenexr-dev libgdal-dev

# Video I/O:
sudo apt-get install -y libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev

# Parallelism and linear algebra libraries:
sudo apt-get install -y libtbb-dev libeigen3-dev

# Python:
sudo apt-get install -y python-dev python-tk python-numpy python3-dev python3-tk python3-numpy

# Java:
sudo apt-get install -y ant default-jdk

# Documentation:
sudo apt-get install -y doxygen


# 3. INSTALL THE LIBRARY (YOU CAN CHANGE '3.2.0' FOR THE LAST STABLE VERSION)

mkdir build
cd build
sudo cmake -DCMAKE_BUILD_TYPE=RELEASE \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_TESTS=OFF \
-DBUILD_opencv_java=OFF \
-DCMAKE_INSTALL_PREFIX=/usr/local ..
# Uncomment the lines below to use virtualenvs
"""
sudo cmake -DCMAKE_BUILD_TYPE=RELEASE \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_TESTS=OFF \
-DBUILD_opencv_java=OFF \
-DPYTHON3_EXECUTABLE=$VIRTUAL_ENV/bin/python \
-DPYTHON3_PACKAGES_PATH=$VIRTUAL_ENV/lib/python3.5/site-packages \
-DPYTHON3_INCLUDE_DIR=$VIRTUAL_ENV/include/python3.5m \
-DPYTHON3_LIBRARY=$VIRTUAL_ENV/lib/python3.5/config-3.5m-x86_64-linux-gnu/libpython3.5.so \
-DCMAKE_INSTALL_PREFIX=/usr/local ..
"""
sudo make -j8
sudo make install
sudo ldconfig

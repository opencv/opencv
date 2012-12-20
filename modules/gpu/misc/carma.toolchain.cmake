set(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)

set(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabi-gcc-4.5)
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabi-g++-4.5)

SET(CMAKE_FIND_ROOT_PATH /usr/arm-linux-gnueabi)

set(CARMA 1)
add_definitions(-DCARMA)

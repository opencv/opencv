# http://wiki.debian.org/BuildingCrossCompilers
# Usage:
#
#  $ cmake ../gdcm -DCMAKE_TOOLCHAIN_FILE=../gdcm/CMake/Toolchain-gcc-arm-linux-gnueabi.cmake
#
# For gdcm you need at least the following three packages (squeeze suite)
#
# fix /etc/apt/source.lists
#  + deb http://www.emdebian.org/debian squeeze main
#
#  // prebuilt Emdebian project
#  sudo apt-get install g++-4.4-arm-linux-gnueabi
#
#  sudo xapt -S squeeze -M http://ftp.fr.debian.org/debian/ -a armel -m zlib1g-dev uuid-dev libexpat1-dev
#
#  qemu-arm -L /usr/arm-linux-gnueabi/ ./bin/gdcminfo test.acr
#
#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

# the name of the target operating system
set(CMAKE_SYSTEM_NAME Linux)

# which compilers to use for C and C++
set(CMAKE_C_COMPILER arm-linux-gnueabi-gcc-4.4)
set(CMAKE_CXX_COMPILER arm-linux-gnueabi-g++-4.4)

# here is the target environment located
set(CMAKE_FIND_ROOT_PATH   /usr/arm-linux-gnueabi)

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

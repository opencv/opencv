# http://www.cmake.org/Wiki/CmakeMingw
# Usage:
#
#  $ cmake ../trunk -DCMAKE_TOOLCHAIN_FILE=../trunk/CMake/Toolchain-mingw32.cmake
#
# For gdcm you need at least the following three package (2008/08/19):
#
#  apt-cross --arch i386 -i zlib1g-dev
#  apt-cross --arch i386 -i uuid-dev
#  apt-cross --arch i386 -i libexpat1-dev
#
# Do not forget to set to on the following:
# GDCM_USE_SYSTEM_EXPAT / GDCM_USE_SYSTEM_ZLIB / GDCM_USE_SYSTEM_UUID
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
set(CMAKE_C_COMPILER gcc)
set(CMAKE_C_FLAGS -m32)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS -m32)

# here is the target environment located
set(CMAKE_FIND_ROOT_PATH   /usr/i486-linux-gnu )

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# http://www.cmake.org/Wiki/CmakeMingw
# http://doc.cliss21.com/index.php?title=QEMU
# Usage:
#
#  $ cmake ../trunk -DCMAKE_TOOLCHAIN_FILE=../trunk/CMake/Toolchain-mingw32.cmake
#
# For gdcm you need at least the following three package (2008/08/19):
#
# fix /etc/apt/source.lists
#  + deb http://www.emdebian.org/debian/ unstable main
#
#  // prebuilt Emdebian project
#  sudo apt-get install g++-4.1-powerpc-linux-gnu
#
#  apt-cross --arch powerpc -i zlib1g-dev
#  apt-cross --arch powerpc -i uuid-dev
#  apt-cross --arch powerpc -i libexpat1-dev
#
#I was getting:
#$ qemu-ppc ./a.out
#/lib/ld.so.1: No such file or directory
#
#Two approach for solving it:
#1.
#CMAKE_EXE_LINKER_FLAGS:STRING=-static
#2.
#$ qemu-ppc -L /usr/powerpc-linux-gnu/ ./a.out
#Hello cross-compiling world!
#
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
set(CMAKE_C_COMPILER powerpc-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER powerpc-linux-gnu-g++)

# here is the target environment located
set(CMAKE_FIND_ROOT_PATH   /usr/powerpc-linux-gnu )

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

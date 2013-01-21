set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER    arm-linux-gnueabihf-gcc-4.6)
set(CMAKE_CXX_COMPILER  arm-linux-gnueabihf-g++-4.6)

#suppress compiller varning
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-psabi" )
set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wno-psabi" )

# can be any other plases
set(ARM_LINUX_SYSROOT /usr/arm-linux-gnueabihf CACHE PATH "ARM cross compilation system root")

set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${ARM_LINUX_SYSROOT})

set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)


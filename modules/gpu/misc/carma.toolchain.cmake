set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER    arm-linux-gnueabi-gcc-4.5)
set(CMAKE_CXX_COMPILER  arm-linux-gnueabi-g++-4.5)

#suppress compiller varning
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-psabi" )
set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wno-psabi" )

# can be any other plases
set(__arm_linux_eabi_root /usr/arm-linux-gnueabi)

set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${__arm_linux_eabi_root})

if(EXISTS ${CUDA_TOOLKIT_ROOT_DIR})
    set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${CUDA_TOOLKIT_ROOT_DIR})
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)

set(CARMA 1)
add_definitions(-DCARMA)

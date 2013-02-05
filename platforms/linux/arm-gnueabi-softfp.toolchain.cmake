set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER    arm-linux-gnueabi-gcc-4.6)
set(CMAKE_CXX_COMPILER  arm-linux-gnueabi-g++-4.6)

set(CMAKE_CXX_FLAGS           ""                    CACHE STRING "c++ flags")
set(CMAKE_C_FLAGS             ""                    CACHE STRING "c flags")
set(CMAKE_SHARED_LINKER_FLAGS ""                    CACHE STRING "shared linker flags")
set(CMAKE_MODULE_LINKER_FLAGS ""                    CACHE STRING "module linker flags")
set(CMAKE_EXE_LINKER_FLAGS    "-Wl,-z,nocopyreloc"  CACHE STRING "executable linker flags")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mthumb -fdata-sections -Wa,--noexecstack -fsigned-char -Wno-psabi")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -mthumb -fdata-sections -Wa,--noexecstack -fsigned-char -Wno-psabi")

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--fix-cortex-a8 -Wl,--no-undefined -Wl,--gc-sections -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now ${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "-Wl,--fix-cortex-a8 -Wl,--no-undefined -Wl,--gc-sections -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now ${CMAKE_MODULE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS    "-Wl,--fix-cortex-a8 -Wl,--no-undefined -Wl,--gc-sections -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now ${CMAKE_EXE_LINKER_FLAGS}")

if(USE_NEON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon")
elseif(USE_VFPV3)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=vfpv3")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=vfpv3")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=vfpv3-d16")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=vfpv3-d16")
endif()

# can be any other plases
set(ARM_LINUX_SYSROOT /usr/arm-linux-gnueabi CACHE PATH "ARM cross compilation system root")

set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${ARM_LINUX_SYSROOT})

if(EXISTS ${CUDA_TOOLKIT_ROOT_DIR})
    set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${CUDA_TOOLKIT_ROOT_DIR})
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)

if (CARMA)
    add_definitions(-DCARMA)
endif()

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(NOT DEFINED CMAKE_C_COMPILER)
  find_program(CMAKE_C_COMPILER NAMES riscv64-unknown-linux-gnu-gcc
                                PATHS /opt/riscv/bin ENV PATH)
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
  find_program(CMAKE_CXX_COMPILER NAMES riscv64-unknown-linux-gnu-g++
                                  PATHS /opt/riscv/bin ENV PATH)
endif()

get_filename_component(RISCV_GCC_INSTALL_ROOT ${CMAKE_C_COMPILER} DIRECTORY)
get_filename_component(RISCV_GCC_INSTALL_ROOT ${RISCV_GCC_INSTALL_ROOT} DIRECTORY)

set(CMAKE_SYSROOT ${RISCV_GCC_INSTALL_ROOT}/sysroot CACHE PATH "RISC-V sysroot")

# Don't run the linker on compiler check
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_FLAGS "-march=rv64gcv_zfh ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-march=rv64gcv_zfh ${CXX_FLAGS}")

set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
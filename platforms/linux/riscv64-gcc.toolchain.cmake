set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)
set(GNU_MACHINE riscv64-unknown-linux-gnu CACHE STRING "GNU compiler triple")

if(NOT DEFINED CMAKE_CXX_FLAGS)  # guards toolchain multiple calls
  set(CMAKE_C_FLAGS "-march=rv64gc")
  set(CMAKE_CXX_FLAGS "-march=rv64gc")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/riscv-gnu.toolchain.cmake")

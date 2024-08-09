set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)
set(GNU_MACHINE riscv64-unknown-linux-gnu CACHE STRING "GNU compiler triple")

message(STATUS "!!! ENABLE_RVV=${ENABLE_RVV}")
message(STATUS "!!! RISCV_RVV_SCALABLE=${RISCV_RVV_SCALABLE}")

set(PLATFORM_STR "rv64gc")
if(ENABLE_RVV OR RISCV_RVV_SCALABLE)
  set(PLATFORM_STR "rv64gcv")
endif()

set(CMAKE_C_FLAGS_INIT "-march=${PLATFORM_STR}")
set(CMAKE_CXX_FLAGS_INIT "-march=${PLATFORM_STR}")

include("${CMAKE_CURRENT_LIST_DIR}/riscv-gnu.toolchain.cmake")

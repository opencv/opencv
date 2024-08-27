set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(GNU_MACHINE riscv64-unknown-linux-gnu CACHE STRING "GNU compiler triple")

include("${CMAKE_CURRENT_LIST_DIR}/flags-riscv64.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/riscv-gnu.toolchain.cmake")

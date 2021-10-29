set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(GNU_MACHINE riscv64-linux-gnu CACHE STRING "GNU compiler triple")
set(GCC_COMPILER_VERSION "" CACHE STRING "GCC Compiler version")

include("${CMAKE_CURRENT_LIST_DIR}/riscv.toolchain.cmake")

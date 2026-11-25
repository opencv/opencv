set(CMAKE_SYSTEM_PROCESSOR loongarch64)
set(GCC_COMPILER_VERSION "" CACHE STRING "GCC Compiler version")
set(GNU_MACHINE "loongarch64-linux-gnu" CACHE STRING "GNU compiler triple")
include("${CMAKE_CURRENT_LIST_DIR}/loongarch.toolchain.cmake")

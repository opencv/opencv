set(CMAKE_SYSTEM_PROCESSOR mips32r5el)
set(GCC_COMPILER_VERSION "" CACHE STRING "GCC Compiler version")
set(GNU_MACHINE "mips-mti-linux-gnu" CACHE STRING "GNU compiler triple")
include("${CMAKE_CURRENT_LIST_DIR}/mips.toolchain.cmake")

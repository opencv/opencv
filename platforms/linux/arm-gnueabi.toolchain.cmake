set(GCC_COMPILER_VERSION "4.6" CACHE STRING "GCC Compiler version")
set(GNU_MACHINE "arm-linux-gnueabi" CACHE STRING "GNU compiler triple")
include("${CMAKE_CURRENT_LIST_DIR}/arm.toolchain.cmake")

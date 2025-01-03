set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(GNU_MACHINE riscv64-unknown-linux-gnu CACHE STRING "GNU compiler triple")

include("${CMAKE_CURRENT_LIST_DIR}/flags-riscv64.cmake")
if(COMMAND ocv_set_platform_flags)
  ocv_set_platform_flags(CMAKE_CXX_FLAGS_INIT)
  ocv_set_platform_flags(CMAKE_C_FLAGS_INIT)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/riscv-gnu.toolchain.cmake")

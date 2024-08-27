set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(GNU_MACHINE riscv64-unknown-linux-gnu CACHE STRING "GNU compiler triple")

include("${CMAKE_CURRENT_LIST_DIR}/flags-riscv64.cmake")
if(OCV_FLAGS)
  set(CMAKE_CXX_FLAGS_INIT "${OCV_FLAGS}")
  set(CMAKE_C_FLAGS_INIT "${OCV_FLAGS}")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/riscv-gnu.toolchain.cmake")

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(GNU_MACHINE riscv64-unknown-linux-gnu CACHE STRING "GNU compiler triple")

include("${CMAKE_CURRENT_LIST_DIR}/flags-riscv64.cmake")

if(COMMAND ocv_set_platform_flags)
  ocv_set_platform_flags(CMAKE_CXX_FLAGS_INIT)
  ocv_set_platform_flags(CMAKE_C_FLAGS_INIT)
endif()

# HACK: GCC 14 and GCC 15.2 for Spacelit autovectorization bug
# Imgproc_resize_area.regression crashes after https://github.com/opencv/opencv/pull/30007
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -fno-tree-vectorize")
set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} -fno-tree-vectorize")

include("${CMAKE_CURRENT_LIST_DIR}/riscv-gnu.toolchain.cmake")

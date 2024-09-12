if((CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    AND NOT CMAKE_CROSSCOMPILING
    AND NOT CMAKE_TOOLCHAIN_FILE)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64") # Maybe use AARCH64 variable?
    include(${CMAKE_CURRENT_LIST_DIR}/../../platforms/linux/flags-aarch64.cmake)
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
    include(${CMAKE_CURRENT_LIST_DIR}/../../platforms/linux/flags-riscv64.cmake)
  endif()
endif()

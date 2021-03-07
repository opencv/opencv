cmake_minimum_required(VERSION 2.8)

if(COMMAND toolchain_save_config)
  return() # prevent recursive call
endif()

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)

include("${CMAKE_CURRENT_LIST_DIR}/gnu.toolchain.cmake")

MESSAGE(STATUS "Debug: CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")

if(NOT "x${GCC_COMPILER_VERSION}" STREQUAL "x")
  set(__GCC_VER_SUFFIX "-${GCC_COMPILER_VERSION}")
endif()

if(NOT DEFINED CMAKE_C_COMPILER)
  MESSAGE("Looking for compler.. ${GNU_MACHINE}-gcc${__GCC_VER_SUFFIX}")
  find_program(CMAKE_C_COMPILER NAMES ${GNU_MACHINE}-gcc${__GCC_VER_SUFFIX})
else()
  #message(WARNING "CMAKE_C_COMPILER=${CMAKE_C_COMPILER} is defined")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
  find_program(CMAKE_CXX_COMPILER NAMES ${GNU_MACHINE}-g++${__GCC_VER_SUFFIX})
else()
  #message(WARNING "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} is defined")
endif()
if(NOT DEFINED CMAKE_LINKER)
  find_program(CMAKE_LINKER NAMES ${GNU_MACHINE}-ld${__GCC_VER_SUFFIX} ${GNU_MACHINE}-ld)
else()
  #message(WARNING "CMAKE_LINKER=${CMAKE_LINKER} is defined")
endif()
if(NOT DEFINED CMAKE_AR)
  find_program(CMAKE_AR NAMES ${GNU_MACHINE}-ar${__GCC_VER_SUFFIX} ${GNU_MACHINE}-ar)
else()
  #message(WARNING "CMAKE_AR=${CMAKE_AR} is defined")
endif()

if(NOT DEFINED RISCV_LINUX_SYSROOT AND DEFINED GNU_MACHINE)
  set(RISCV_LINUX_SYSROOT /usr/${GNU_MACHINE})
endif()

if(NOT DEFINED CMAKE_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS           "${CMAKE_CXX_FLAGS} -fdata-sections -Wa,--noexecstack -fsigned-char -Wno-psabi")
  set(CMAKE_C_FLAGS             "${CMAKE_C_FLAGS} -fdata-sections -Wa,--noexecstack -fsigned-char -Wno-psabi")
  set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,nocopyreloc")

  set(RISCV_LINKER_FLAGS "-Wl,--no-undefined -Wl,--gc-sections -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
  set(CMAKE_SHARED_LINKER_FLAGS "${RISCV_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
  set(CMAKE_MODULE_LINKER_FLAGS "${RISCV_LINKER_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS    "${RISCV_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
else()
  message(STATUS "User provided flags are used instead of defaults")
endif()

set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${RISCV_LINUX_SYSROOT})

set(TOOLCHAIN_CONFIG_VARS ${TOOLCHAIN_CONFIG_VARS}
    RISCV_LINUX_SYSROOT
)
toolchain_save_config()

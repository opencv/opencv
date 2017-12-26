if(COMMAND toolchain_save_config)
  return() # prevent recursive call
endif()

option(AT_PATH        "Advance Toolchain directory" "")
option(AT_RPATH       "Add new directories to runtime search path" "")
option(AT_HOST_LINK   "Enable/disable Link against host advance toolchain runtime" OFF)
option(AT_NO_AUTOVEC  "Disable/enable Auto Vectorizer optimization" OFF)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)

include("${CMAKE_CURRENT_LIST_DIR}/gnu.toolchain.cmake")

if(NOT DEFINED CMAKE_C_COMPILER)
  string(REGEX REPLACE "/+$" "" AT_PATH "${AT_PATH}")

  if(NOT AT_PATH)
    message(FATAL_ERROR "'AT_PATH' option is required. Please set it to Advance Toolchain path to get toolchain works")
  endif()

  if(NOT EXISTS ${AT_PATH})
    message(FATAL_ERROR "'${AT_PATH}' Advance Toolchain path isn't exist")
  endif()

  set(CMAKE_C_COMPILER "${AT_PATH}/bin/${GNU_MACHINE}-gcc")

  if(NOT EXISTS ${CMAKE_C_COMPILER})
    message(FATAL_ERROR "GNU C compiler isn't exist on path '${CMAKE_C_COMPILER}'. Please install Advance Toolchain with ${CMAKE_SYSTEM_PROCESSOR} supports")
  endif()
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "${AT_PATH}/bin/${GNU_MACHINE}-g++")

  if(NOT EXISTS ${CMAKE_CXX_COMPILER})
    message(FATAL_ERROR "GNU C++ compiler isn't exist. Invalid install of Advance Toolchain")
  endif()
endif()

if(NOT DEFINED AT_GCCROOT_PATH)
  set(AT_GCCROOT_PATH "${AT_PATH}/${GNU_MACHINE}")

  if(NOT EXISTS ${AT_GCCROOT_PATH})
    message(FATAL_ERROR "GCC root path '${AT_GCCROOT_PATH}' isn't exist. Invalid install of Advance Toolchain")
  endif()
endif()

if(NOT DEFINED AT_SYSROOT_PATH)
  if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "ppc64")
    set(AT_SYSROOT_PATH "${AT_PATH}/ppc")
  else()
    set(AT_SYSROOT_PATH "${AT_PATH}/${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  if(NOT EXISTS ${AT_SYSROOT_PATH})
    message(FATAL_ERROR "System root path '${AT_SYSROOT_PATH}' isn't exist. Invalid install of Advance Toolchain")
  endif()
endif()

if(NOT DEFINED CMAKE_EXE_LINKER_FLAGS)
  set(CMAKE_CXX_FLAGS           "" CACHE INTERAL "")
  set(CMAKE_C_FLAGS             "" CACHE INTERAL "")
  set(CMAKE_EXE_LINKER_FLAGS    "" CACHE INTERAL "")
  set(CMAKE_SHARED_LINKER_FLAGS "" CACHE INTERAL "")
  set(CMAKE_MODULE_LINKER_FLAGS "" CACHE INTERAL "")

  if(AT_RPATH)
    string(REPLACE "," ";" RPATH_LIST ${AT_RPATH})
  endif()

  if(AT_HOST_LINK)
    #get 64-bit dynamic linker path
    file(STRINGS "${AT_SYSROOT_PATH}/usr/bin/ldd" RTLDLIST LIMIT_COUNT 1 REGEX "^RTLDLIST=[\"*\"]")
    string(REGEX REPLACE "RTLDLIST=|\"" "" RTLDLIST "${RTLDLIST}")
    string(REPLACE " " ";" RTLDLIST "${RTLDLIST}")

    #RTLDLIST must contains 32 and 64 bit paths
    list(LENGTH RTLDLIST RTLDLIST_LEN)
    if(NOT RTLDLIST_LEN GREATER 1)
      message(FATAL_ERROR "Could not fetch dynamic linker path. Invalid install of Advance Toolchain")
    endif()

    list (GET RTLDLIST 1 LINKER_PATH)
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,--dynamic-linker=${AT_SYSROOT_PATH}${LINKER_PATH}")

    list(APPEND RPATH_LIST "${AT_GCCROOT_PATH}/lib64/")
    list(APPEND RPATH_LIST "${AT_SYSROOT_PATH}/lib64/")
    list(APPEND RPATH_LIST "${AT_SYSROOT_PATH}/usr/lib64/")
    list(APPEND RPATH_LIST "${PROJECT_BINARY_DIR}/lib/")
  endif()

  list(LENGTH RPATH_LIST RPATH_LEN)
  if(RPATH_LEN GREATER 0)
    set(AT_LINKER_FLAGS "${AT_LINKER_FLAGS} -Wl")
    foreach(RPATH ${RPATH_LIST})
      set(AT_LINKER_FLAGS "${AT_LINKER_FLAGS},-rpath,${RPATH}")
    endforeach()
  endif()

  set(CMAKE_SHARED_LINKER_FLAGS "${AT_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
  set(CMAKE_MODULE_LINKER_FLAGS "${AT_LINKER_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS    "${AT_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")

  if(AT_NO_AUTOVEC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-tree-vectorize -fno-tree-slp-vectorize")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -fno-tree-vectorize -fno-tree-slp-vectorize")
  endif()

endif()

set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${AT_SYSROOT_PATH} ${AT_GCCROOT_PATH})
set(CMAKE_SYSROOT ${AT_SYSROOT_PATH})

# what about ld.gold?
if(NOT DEFINED CMAKE_LINKER)
  find_program(CMAKE_LINKER NAMES ld)
endif()

if(NOT DEFINED CMAKE_AR)
  find_program(CMAKE_AR NAMES ar)
endif()

set(TOOLCHAIN_CONFIG_VARS ${TOOLCHAIN_CONFIG_VARS}
    CMAKE_SYSROOT
    AT_SYSROOT_PATH
    AT_GCCROOT_PATH
)
toolchain_save_config()

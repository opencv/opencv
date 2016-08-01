cmake_minimum_required(VERSION 2.8)

# load settings in case of "try compile"
set(TOOLCHAIN_CONFIG_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/toolchain.config.cmake")
get_property(__IN_TRY_COMPILE GLOBAL PROPERTY IN_TRY_COMPILE)
if(__IN_TRY_COMPILE)
  include("${CMAKE_CURRENT_SOURCE_DIR}/../toolchain.config.cmake" OPTIONAL) # CMAKE_BINARY_DIR is different
  macro(toolchain_save_config)
    # nothing
  endmacro()
else()
  macro(toolchain_save_config)
    set(__config "#message(\"Load TOOLCHAIN config...\")\n")
    get_cmake_property(__variableNames VARIABLES)
    set(__vars_list ${ARGN})
    list(APPEND __vars_list
        ${TOOLCHAIN_CONFIG_VARS}
        CMAKE_SYSTEM_NAME
        CMAKE_SYSTEM_VERSION
        CMAKE_SYSTEM_PROCESSOR
        CMAKE_C_COMPILER
        CMAKE_CXX_COMPILER
        CMAKE_C_FLAGS
        CMAKE_CXX_FLAGS
        CMAKE_SHARED_LINKER_FLAGS
        CMAKE_MODULE_LINKER_FLAGS
        CMAKE_EXE_LINKER_FLAGS
        CMAKE_SKIP_RPATH
        CMAKE_FIND_ROOT_PATH
        GCC_COMPILER_VERSION
    )
    foreach(__var ${__variableNames})
      foreach(_v ${__vars_list})
        if("x${__var}" STREQUAL "x${_v}")
          if(${__var} MATCHES " ")
            set(__config "${__config}set(${__var} \"${${__var}}\")\n")
          else()
            set(__config "${__config}set(${__var} ${${__var}})\n")
          endif()
        endif()
      endforeach()
    endforeach()
    if(EXISTS "${TOOLCHAIN_CONFIG_FILE}")
      file(READ "${TOOLCHAIN_CONFIG_FILE}" __config_old)
    endif()
    if("${__config_old}" STREQUAL "${__config}")
      # nothing
    else()
      #message("Update TOOLCHAIN config: ${__config}")
      file(WRITE "${TOOLCHAIN_CONFIG_FILE}" "${__config}")
    endif()
    unset(__config)
    unset(__config_old)
    unset(__vars_list)
    unset(__variableNames)
  endmacro()
endif() # IN_TRY_COMPILE

set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)

# macro to find programs on the host OS
macro(find_host_program)
 set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
 set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
 set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
 if(CMAKE_HOST_WIN32)
  SET(WIN32 1)
  SET(UNIX)
 elseif(CMAKE_HOST_APPLE)
  SET(APPLE 1)
  SET(UNIX)
 endif()
 find_program(${ARGN})
 SET(WIN32)
 SET(APPLE)
 SET(UNIX 1)
 set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)
 set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
 set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
endmacro()

# macro to find packages on the host OS
macro(find_host_package)
 set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
 set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
 set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
 if(CMAKE_HOST_WIN32)
  SET(WIN32 1)
  SET(UNIX)
 elseif(CMAKE_HOST_APPLE)
  SET(APPLE 1)
  SET(UNIX)
 endif()
 find_package(${ARGN})
 SET(WIN32)
 SET(APPLE)
 SET(UNIX 1)
 set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY)
 set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
 set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
endmacro()

set(CMAKE_SKIP_RPATH TRUE CACHE BOOL "If set, runtime paths are not added when using shared libraries.")

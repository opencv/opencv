# Locate Intel Threading Building Blocks include paths and libraries
# TBB can be found at http://www.threadingbuildingblocks.org/
# Written by Hannes Hofmann, hannes.hofmann _at_ informatik.uni-erlangen.de
# Adapted by Gino van den Bergen gino _at_ dtecta.com
#         and Vinogradov Vladislav

# This module defines
# TBB_INCLUDE_DIRS, where to find task_scheduler_init.h, etc.
# TBB_LIBRARY_DIRS, where to find libtbb, libtbbmalloc
# TBB_LIBRARIES, the libraries to link against to use TBB.
# TBB_FOUND, If false, don't try to use TBB.

if(WIN32)
    set(TBB_DEFAULT_INSTALL_DIR "C:/Program Files/Intel/TBB")

    #-- Find TBB install dir and set ${_TBB_INSTALL_DIR} and cached ${TBB_INSTALL_DIR}
    # first: use CMake variable TBB_INSTALL_DIR
    if(TBB_INSTALL_DIR)
        set(_TBB_INSTALL_DIR ${TBB_INSTALL_DIR})
    endif()

    # second: use environment variable
    if(NOT _TBB_INSTALL_DIR)
        if(NOT "$ENV{TBB_INSTALL_DIR}" STREQUAL "")
            set(_TBB_INSTALL_DIR $ENV{TBB_INSTALL_DIR})
        endif()

        # Intel recommends setting TBB21_INSTALL_DIR
        if(NOT "$ENV{TBB21_INSTALL_DIR}" STREQUAL "")
            set(_TBB_INSTALL_DIR $ENV{TBB21_INSTALL_DIR})
        endif()

        if(NOT "$ENV{TBB22_INSTALL_DIR}" STREQUAL "")
            set(_TBB_INSTALL_DIR $ENV{TBB22_INSTALL_DIR})
        endif()

        if(NOT "$ENV{TBB30_INSTALL_DIR}" STREQUAL "")
            set(_TBB_INSTALL_DIR $ENV{TBB30_INSTALL_DIR})
        endif()
    endif()

    # third: try to find path automatically
    if(NOT _TBB_INSTALL_DIR)
        set(_TBB_INSTALL_DIR ${TBB_DEFAULT_INSTALL_DIR})
    endif()

    # finally: set the cached CMake variable TBB_INSTALL_DIR
    set(TBB_INSTALL_DIR ${_TBB_INSTALL_DIR} CACHE PATH "Intel TBB install directory")

    if(MSVC71)
        set(_TBB_COMPILER "vc7.1")
    elseif(MSVC80)
        set(_TBB_COMPILER "vc8")
    elseif(MSVC90)
        set(_TBB_COMPILER "vc9")
    elseif(MSVC10)
        set(_TBB_COMPILER "vc10")
    endif()

    set(TBB_COMPILER ${_TBB_COMPILER} CACHE STRING "TBB Compiler")
    mark_as_advanced(TBB_COMPILER)

    if(NOT TBB_COMPILER)
        message(WARNING "TBB supports only VC 7.1, 8, 9 and 10 compilers on Windows platforms.")
    endif()

    if(X86_64 OR CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_TBB_ARCHITECTURE "intel64")
    else(X86_64 OR CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_TBB_ARCHITECTURE "ia32")
    endif(X86_64 OR CMAKE_SIZEOF_VOID_P EQUAL 8)

    set(TBB_ARCHITECTURE ${_TBB_ARCHITECTURE} CACHE STRING "TBB Architecture (ia32 OR intel64)")
    mark_as_advanced(TBB_ARCHITECTURE)

    set(TBB_INC_SEARCH_DIR "${TBB_INSTALL_DIR}/include")
    set(TBB_LIBRARY_DIR "${TBB_INSTALL_DIR}/lib/${TBB_ARCHITECTURE}/${TBB_COMPILER}")
elseif(UNIX AND NOT APPLE)
    set(TBB_INC_SEARCH_DIR "usr/include")
    set(TBB_LIBRARY_DIR "usr/lib")
endif()

#-- Look for include directory and set ${TBB_INCLUDE_DIR}
find_path(TBB_INCLUDE_DIRS NAMES tbb/tbb.h PATHS ${TBB_INC_SEARCH_DIR})
mark_as_advanced(TBB_INCLUDE_DIRS)

#-- Look for libraries
set(TBB_LIB_NAME "tbb")
set(TBB_LIB_DEBUG_NAME "${TBB_LIB_NAME}_debug")
set(TBB_LIB_MALLOC_NAME "${TBB_LIB_NAME}malloc")
set(TBB_LIB_MALLOC_DEBUG_NAME "${TBB_LIB_MALLOC_NAME}_debug")
find_library(TBB_LIBRARY        NAMES ${TBB_LIB_NAME}        PATHS ${TBB_LIBRARY_DIR})
find_library(TBB_MALLOC_LIBRARY NAMES ${TBB_LIB_MALLOC_NAME} PATHS ${TBB_LIBRARY_DIR})

#-- Extract path from TBB_LIBRARY name
get_filename_component(TBB_LIBRARY_DIR ${TBB_LIBRARY} PATH)
mark_as_advanced(TBB_LIBRARY TBB_MALLOC_LIBRARY)

#-- Look for debug libraries
if(WIN32)
    find_library(TBB_LIBRARY_DEBUG        NAMES ${TBB_LIB_DEBUG_NAME}        PATHS ${TBB_LIBRARY_DIR})
    find_library(TBB_MALLOC_LIBRARY_DEBUG NAMES ${TBB_LIB_MALLOC_DEBUG_NAME} PATHS ${TBB_LIBRARY_DIR})
    mark_as_advanced(TBB_LIBRARY_DEBUG TBB_MALLOC_LIBRARY_DEBUG)
endif()

if(TBB_INCLUDE_DIRS AND TBB_LIBRARY)
    if(WIN32)
        set(TBB_LIBRARIES optimized ${TBB_LIBRARY} debug ${TBB_LIBRARY_DEBUG} optimized ${TBB_MALLOC_LIBRARY} debug ${TBB_MALLOC_LIBRARY_DEBUG})
    else()
        set(TBB_LIBRARIES ${TBB_LIBRARY} ${TBB_MALLOC_LIBRARY})
    endif()

    set(TBB_LIBRARY_DIRS ${TBB_LIBRARY_DIR} CACHE PATH "TBB library directory" FORCE)
    mark_as_advanced(TBB_LIBRARY_DIRS)
endif()

find_package(PackageHandleStandardArgs REQUIRED)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TBB REQUIRED_VARS TBB_INCLUDE_DIRS TBB_LIBRARY)

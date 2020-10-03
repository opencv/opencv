cmake_minimum_required(VERSION 2.8.0)
project(harfbuzz)

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 OLD)
endif()

## Limit framework build to Xcode generator
if (BUILD_FRAMEWORK)
  # for a framework build on macOS, use:
  # cmake -DBUILD_FRAMEWORK=ON -Bbuild -H. -GXcode && cmake --build build
  if (NOT "${CMAKE_GENERATOR}" STREQUAL "Xcode")
    message(FATAL_ERROR
      "You should use Xcode generator with BUILD_FRAMEWORK enabled")
  endif ()
  set (CMAKE_OSX_ARCHITECTURES "$(ARCHS_STANDARD_32_64_BIT)")
  set (CMAKE_MACOSX_RPATH ON)
  set (BUILD_SHARED_LIBS ON)
endif ()


## Disallow in-source builds, as CMake generated make files can collide with autotools ones
if (NOT MSVC AND "${PROJECT_BINARY_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")
  message(FATAL_ERROR
    "
In-source builds are not permitted!  Make a separate folder for"
    " building, e.g.,"
    "
  mkdir build; cd build; cmake .."
    "
Before that, remove the files created by this failed run with"
    "
  rm -rf CMakeCache.txt CMakeFiles")
endif ()


## HarfBuzz build configurations
set(HB_HAVE_FREETYPE ON)

include_directories(AFTER
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)

# We need PYTHON_EXECUTABLE to be set for running the tests...
include (FindPythonInterp)

## Functions and headers
include (CheckFunctionExists)
include (CheckIncludeFile)
macro (check_funcs) # Similar to AC_CHECK_FUNCS of autotools
  foreach (func_name ${ARGN})
    string(TOUPPER ${func_name} definition_to_add)
    check_function_exists(${func_name} HAVE_${definition_to_add})
    if (${HAVE_${definition_to_add}})
      add_definitions(-DHAVE_${definition_to_add})
    endif ()
  endforeach ()
endmacro ()
if (UNIX)
  list(APPEND CMAKE_REQUIRED_LIBRARIES m)
endif ()
check_funcs(atexit mprotect sysconf getpagesize mmap isatty)
check_include_file(unistd.h HAVE_UNISTD_H)
if (${HAVE_UNISTD_H})
  add_definitions(-DHAVE_UNISTD_H)
endif ()
check_include_file(sys/mman.h HAVE_SYS_MMAN_H)
if (${HAVE_SYS_MMAN_H})
  add_definitions(-DHAVE_SYS_MMAN_H)
endif ()
check_include_file(stdbool.h HAVE_STDBOOL_H)
if (${HAVE_STDBOOL_H})
  add_definitions(-DHAVE_STDBOOL_H)
endif ()


if (MSVC)
  add_definitions(-wd4244 -wd4267 -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_WARNINGS)
endif ()


## Detect if we are running inside a distribution or regular repository folder
# if (EXISTS "${PROJECT_SOURCE_DIR}/ChangeLog")
#   # perhaps we are on dist directory
#   set (IN_HB_DIST TRUE)
#   #set (HB_VERSION_H "${PROJECT_SOURCE_DIR}/src/hb-version.h")
# endif ()


## Extract variables from Makefile files
function (extract_make_variable variable makefile_source)
  string(REGEX MATCH "${variable} = ([^$]+)\\$" temp "${makefile_source}")
  string(REGEX MATCHALL "[^ \n\t\\]+" listVar "${CMAKE_MATCH_1}")
  set (${variable} ${listVar} PARENT_SCOPE)
endfunction ()

# https://stackoverflow.com/a/27630120
function (add_prefix_to_list var prefix)
  set (listVar "")
  foreach (f ${${var}})
    list(APPEND listVar "${prefix}${f}")
  endforeach ()
  set (${var} "${listVar}" PARENT_SCOPE)
endfunction ()

file(READ ${PROJECT_SOURCE_DIR}/src/Makefile.sources SRCSOURCES)
#file(READ ${PROJECT_SOURCE_DIR}/util/Makefile.sources UTILSOURCES)

extract_make_variable(HB_BASE_headers ${SRCSOURCES})
add_prefix_to_list(HB_BASE_headers "${PROJECT_SOURCE_DIR}/src/")

extract_make_variable(HB_SUBSET_sources ${SRCSOURCES})
add_prefix_to_list(HB_SUBSET_sources "${PROJECT_SOURCE_DIR}/src/")

extract_make_variable(HB_SUBSET_headers ${SRCSOURCES})
add_prefix_to_list(HB_SUBSET_headers "${PROJECT_SOURCE_DIR}/src/")

extract_make_variable(HB_BASE_RAGEL_GENERATED_sources ${SRCSOURCES})
#if (IN_HB_DIST)
  add_prefix_to_list(HB_BASE_RAGEL_GENERATED_sources "${PROJECT_SOURCE_DIR}/src/")
#else ()
#  add_prefix_to_list(HB_BASE_RAGEL_GENERATED_sources "${PROJECT_BINARY_DIR}/src/")
#endif ()

#extract_make_variable(HB_VIEW_sources ${UTILSOURCES})
#add_prefix_to_list(HB_VIEW_sources "${PROJECT_SOURCE_DIR}/util/")
#extract_make_variable(HB_SHAPE_sources ${UTILSOURCES})
#add_prefix_to_list(HB_SHAPE_sources "${PROJECT_SOURCE_DIR}/util/")
#extract_make_variable(HB_SUBSET_CLI_sources ${UTILSOURCES})
#add_prefix_to_list(HB_SUBSET_CLI_sources "${PROJECT_SOURCE_DIR}/util/")
#extract_make_variable(HB_OT_SHAPE_CLOSURE_sources ${UTILSOURCES})
#add_prefix_to_list(HB_OT_SHAPE_CLOSURE_sources "${PROJECT_SOURCE_DIR}/util/")


file(READ configure.ac CONFIGUREAC)
string(REGEX MATCH "\\[(([0-9]+)\\.([0-9]+)\\.([0-9]+))\\]" HB_VERSION_MATCH ${CONFIGUREAC})
set (HB_VERSION ${CMAKE_MATCH_1})
set (HB_VERSION_MAJOR ${CMAKE_MATCH_2})
set (HB_VERSION_MINOR ${CMAKE_MATCH_3})
set (HB_VERSION_MICRO ${CMAKE_MATCH_4})

## Define sources and headers of the project
set (project_sources ${PROJECT_SOURCE_DIR}/src/harfbuzz.cc) # use amalgam source
set (subset_project_sources ${HB_SUBSET_sources})
set (project_extra_sources)
set (project_headers ${HB_BASE_headers})
set (subset_project_headers ${HB_SUBSET_headers})

## Find and include needed header folders and libraries
if (HB_HAVE_FREETYPE)
  include_directories(AFTER ${FREETYPE_INCLUDE_DIRS})
  add_definitions(-DHAVE_FREETYPE=1)
  list(APPEND project_headers ${PROJECT_SOURCE_DIR}/src/hb-ft.h)
endif ()

## Atomic ops availability detection
file(WRITE "${PROJECT_BINARY_DIR}/try_compile_intel_atomic_primitives.c"
"		void memory_barrier (void) { __sync_synchronize (); }
		int atomic_add (int *i) { return __sync_fetch_and_add (i, 1); }
		int mutex_trylock (int *m) { return __sync_lock_test_and_set (m, 1); }
		void mutex_unlock (int *m) { __sync_lock_release (m); }
		int main () { return 0; }
")
try_compile(HB_HAVE_INTEL_ATOMIC_PRIMITIVES
  ${PROJECT_BINARY_DIR}/try_compile_intel_atomic_primitives
  ${PROJECT_BINARY_DIR}/try_compile_intel_atomic_primitives.c)
if (HB_HAVE_INTEL_ATOMIC_PRIMITIVES)
  add_definitions(-DHAVE_INTEL_ATOMIC_PRIMITIVES)
endif ()

file(WRITE "${PROJECT_BINARY_DIR}/try_compile_solaris_atomic_ops.c"
"		#include <atomic.h>
		/* This requires Solaris Studio 12.2 or newer: */
		#include <mbarrier.h>
		void memory_barrier (void) { __machine_rw_barrier (); }
		int atomic_add (volatile unsigned *i) { return atomic_add_int_nv (i, 1); }
		void *atomic_ptr_cmpxchg (volatile void **target, void *cmp, void *newval) { return atomic_cas_ptr (target, cmp, newval); }
		int main () { return 0; }
")
try_compile(HB_HAVE_SOLARIS_ATOMIC_OPS
  ${PROJECT_BINARY_DIR}/try_compile_solaris_atomic_ops
  ${PROJECT_BINARY_DIR}/try_compile_solaris_atomic_ops.c)
if (HB_HAVE_SOLARIS_ATOMIC_OPS)
  add_definitions(-DHAVE_SOLARIS_ATOMIC_OPS)
endif ()


## Define harfbuzz library
add_library(${PROJECT_NAME} STATIC ${project_sources} ${project_extra_sources} ${project_headers})

set_target_properties(
  ${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}
    C_VISIBILITY_PRESET hidden
    DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
    COMPILE_PDB_NAME ${PROJECT_NAME}
    COMPILE_PDB_NAME_DEBUG "${PROJECT_NAME}${OPENCV_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

if (UNIX OR MINGW)
  # Make symbols link locally
  include (CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG(-Bsymbolic-functions CXX_SUPPORTS_FLAG_BSYMB_FUNCS)
  if (CXX_SUPPORTS_FLAG_BSYMB_FUNCS)
    link_libraries(-Bsymbolic-functions)
  endif ()

  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Make sure we don't link to libstdc++
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")
    set (CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "m") # libm
    set (CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "")
    set_target_properties(harfbuzz PROPERTIES LINKER_LANGUAGE C)
    if (HB_BUILD_SUBSET)
      set_target_properties(harfbuzz-subset PROPERTIES LINKER_LANGUAGE C)
    endif ()

    # No threadsafe statics as we do it ourselves
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-threadsafe-statics")
  endif ()

  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  if (COMPILER_SUPPORTS_CXX11)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
  endif()
endif ()

set(HARFBUZZ_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" PARENT_SCOPE)
set(HARFBUZZ_LIBRARIES harfbuzz PARENT_SCOPE)

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "3rdparty")
endif()

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(${PROJECT_NAME} EXPORT OpenCVModules ARCHIVE DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

ocv_install_3rdparty_licenses(${PROJECT_NAME} harfbuzz_COPYING.txt)

cmake_minimum_required(VERSION 3.1)

if(" ${HALIDE_ROOT_DIR}" STREQUAL " ")
  unset(HALIDE_ROOT_DIR CACHE)
endif()
ocv_check_environment_variables(HALIDE_ROOT_DIR)
set(HALIDE_ROOT_DIR "${HALIDE_ROOT_DIR}" CACHE PATH "Halide root directory")

if(NOT HAVE_HALIDE)
  find_package(Halide QUIET) # Try CMake-based config files
  if(Halide_FOUND)
    set(HALIDE_INCLUDE_DIRS "${Halide_INCLUDE_DIRS}" CACHE PATH "Halide include directories" FORCE)
    set(HALIDE_LIBRARIES "${Halide_LIBRARIES}" CACHE PATH "Halide libraries" FORCE)
    set(HAVE_HALIDE TRUE)
  endif()
endif()

if(NOT HAVE_HALIDE AND HALIDE_ROOT_DIR)
  # Try manual search
  find_library(HALIDE_LIBRARY
      NAMES Halide
      HINTS ${HALIDE_ROOT_DIR}/lib          # Unix
      HINTS ${HALIDE_ROOT_DIR}/lib/Release  # Win32
  )
  find_path(HALIDE_INCLUDE_DIR
      NAMES Halide.h HalideRuntime.h
      HINTS ${HALIDE_ROOT_DIR}/include
  )
  if(HALIDE_LIBRARY AND HALIDE_INCLUDE_DIR)
    # TODO try_compile
    set(HALIDE_INCLUDE_DIRS "${HALIDE_INCLUDE_DIR}" CACHE PATH "Halide include directories" FORCE)
    set(HALIDE_LIBRARIES "${HALIDE_LIBRARY}" CACHE PATH "Halide libraries" FORCE)
    set(HAVE_HALIDE TRUE)
  endif()
  if(NOT HAVE_HALIDE)
    ocv_clear_vars(HALIDE_LIBRARIES HALIDE_INCLUDE_DIRS CACHE)
  endif()
endif()

if(HAVE_HALIDE)
  include_directories(${HALIDE_INCLUDE_DIRS})
  list(APPEND OPENCV_LINKER_LIBS ${HALIDE_LIBRARIES})
else()
  ocv_clear_vars(HALIDE_INCLUDE_DIRS HALIDE_LIBRARIES)
endif()

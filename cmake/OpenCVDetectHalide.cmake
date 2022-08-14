cmake_minimum_required(VERSION ${MIN_VER_CMAKE})

if(" ${HALIDE_ROOT_DIR}" STREQUAL " ")
  unset(HALIDE_ROOT_DIR CACHE)
endif()
ocv_check_environment_variables(HALIDE_ROOT_DIR)
set(HALIDE_ROOT_DIR "${HALIDE_ROOT_DIR}" CACHE PATH "Halide root directory")

if(NOT HAVE_HALIDE)
  find_package(Halide QUIET) # Try CMake-based config files
  if(Halide_FOUND)
    if(TARGET Halide::Halide)  # modern Halide scripts defines imported target
      set(HALIDE_INCLUDE_DIRS "")
      set(HALIDE_LIBRARIES "Halide::Halide")
      set(HAVE_HALIDE TRUE)
    else()
      # using HALIDE_INCLUDE_DIRS / Halide_LIBRARIES
      set(HAVE_HALIDE TRUE)
    endif()
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
    set(HALIDE_INCLUDE_DIRS "${HALIDE_INCLUDE_DIR}")
    set(HALIDE_LIBRARIES "${HALIDE_LIBRARY}")
    set(HAVE_HALIDE TRUE)
  endif()
endif()

if(HAVE_HALIDE)
  if(HALIDE_INCLUDE_DIRS)
    include_directories(${HALIDE_INCLUDE_DIRS})
  endif()
  list(APPEND OPENCV_LINKER_LIBS ${HALIDE_LIBRARIES})
endif()

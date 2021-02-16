if(CMAKE_VERSION VERSION_LESS "3.9")
  message(STATUS "OpenMP detection requires CMake 3.9+")  # OpenMP::OpenMP_CXX target
endif()

find_package(OpenMP)
if(OpenMP_FOUND)
  if(TARGET OpenMP::OpenMP_CXX)
    set(HAVE_OPENMP 1)
    ocv_add_external_target(openmp "" "OpenMP::OpenMP_CXX" "HAVE_OPENMP=1")
  else()
    message(WARNING "OpenMP: missing OpenMP::OpenMP_CXX target")
  endif()
endif()

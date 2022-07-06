set(the_description "Clustering and Search in Multi-Dimensional Spaces")
if(ARM AND CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  # suppress warnings from GCC only on 7.1 and later
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-psabi)
endif()
ocv_define_module(flann opencv_core WRAP python)

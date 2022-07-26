set(the_description "2D Features Framework")

ocv_add_dispatched_file(sift SSE4_1 AVX2 AVX512_SKX)
if(ARM AND CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  # suppress warnings from GCC only on 7.1 and later
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-psabi)
endif()

set(debug_modules "")
if(DEBUG_opencv_features2d)
  list(APPEND debug_modules opencv_highgui)
endif()
ocv_define_module(features2d opencv_imgproc ${debug_modules} OPTIONAL opencv_flann WRAP java objc python js)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/include")
set(deps opencv_lapack zlib flann)
define_opencv_module(core ${deps})

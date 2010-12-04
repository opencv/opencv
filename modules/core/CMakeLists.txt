include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/include")
set(deps opencv_lapack opencv_zlib)
define_opencv_module(core ${deps})

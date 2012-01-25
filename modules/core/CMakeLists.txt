if(ZLIB_FOUND)
    include_directories(${ZLIB_INCLUDE_DIR})
else()
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/zlib")
endif()

define_opencv_module(core ${ZLIB_LIBRARY})

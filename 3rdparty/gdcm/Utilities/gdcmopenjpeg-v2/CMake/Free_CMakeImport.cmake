set(CMAKE_MODULE_PATH "${OPENJPEG_SOURCE_DIR}/CMake")
find_package(FreeImage REQUIRED)
add_definitions ( -DFREEIMAGE_LIB )

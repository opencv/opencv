set(the_description "The ts module")

if(IOS OR WINRT)
  ocv_module_disable(ts)
endif()

set(OPENCV_MODULE_TYPE STATIC)
set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)

ocv_add_module(ts INTERNAL opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui)

ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()

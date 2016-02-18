set(the_description "The ts module")

if(IOS)
  ocv_module_disable(ts)
endif()

set(OPENCV_MODULE_TYPE STATIC)
set(OPENCV_MODULE_IS_PART_OF_WORLD FALSE)

ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef)

<<<<<<< HEAD
ocv_add_module(ts opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui)
=======
ocv_add_module(ts opencv_core opencv_features2d opencv_highgui opencv_imgproc opencv_video)
>>>>>>> a28cde9c3bf69e7839971c29900fbbd4963998bd

ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()

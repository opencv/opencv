message(STATUS "opencv_dnn: filter out ocl4dnn source code")
ocv_list_filterout(OPENCV_MODULE_${the_module}_SOURCES "/ocl4dnn/")
ocv_list_filterout(OPENCV_MODULE_${the_module}_HEADERS "/ocl4dnn/")

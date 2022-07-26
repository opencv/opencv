set(the_description "Object Detection")

ocv_define_module(objdetect
    opencv_core
    opencv_imgproc
    opencv_calib3d
    OPTIONAL
        opencv_dnn
    WRAP
        python
        java
        objc
        js
)

if(ARM AND CV_GCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  # suppress warnings from GCC only on 7.1 and later
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wno-psabi)
endif()

if(HAVE_QUIRC)
    get_property(QUIRC_INCLUDE GLOBAL PROPERTY QUIRC_INCLUDE_DIR)
    ocv_include_directories(${QUIRC_INCLUDE})
    ocv_target_link_libraries(${the_module} quirc)
endif()

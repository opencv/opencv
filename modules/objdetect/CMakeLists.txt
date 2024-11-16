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

if(HAVE_QUIRC)
    get_property(QUIRC_INCLUDE GLOBAL PROPERTY QUIRC_INCLUDE_DIR)
    ocv_include_directories(${QUIRC_INCLUDE})
    ocv_target_link_libraries(${the_module} quirc)
endif()

if(CMAKE_VERSION VERSION_GREATER "3.11")
  find_package(Iconv QUIET)
  if(Iconv_FOUND)
    ocv_target_link_libraries(${the_module} Iconv::Iconv)
  else()
    ocv_target_compile_definitions(${the_module} PRIVATE "NO_ICONV=1")
  endif()
endif()
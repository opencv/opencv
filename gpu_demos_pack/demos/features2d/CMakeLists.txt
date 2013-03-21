list(FIND OpenCV_LIB_COMPONENTS "opencv_nonfree" idx)

if(idx GREATER -1)
    ocv_define_gpu_demo(features2d DESCRIPTION "Features2D")
endif()

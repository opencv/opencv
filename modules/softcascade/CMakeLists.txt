set(the_description "Soft Cascade detection and training")
ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4310 -Wundef -Wmissing-declarations -Wmissing-prototypes)
ocv_define_module(softcascade opencv_core opencv_imgproc opencv_ml)

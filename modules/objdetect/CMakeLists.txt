set(the_description "Object Detection")
ocv_define_module(objdetect
    opencv_core
    opencv_imgproc
    opencv_3d
    opencv_flann
    opencv_features
    OPTIONAL
        opencv_dnn
    WRAP
        python
        java
        objc
        js
)

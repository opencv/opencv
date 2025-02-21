set(the_description "DNN-based algorithms")
ocv_define_module(dnn_algorithms
    opencv_core
    opencv_imgproc
    opencv_dnn
    # opencv_objdetect
    opencv_video
    WRAP
      java
      objc
      python
      js
)

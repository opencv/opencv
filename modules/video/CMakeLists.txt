set(the_description "Video Analysis")
ocv_define_module(video
    opencv_imgproc
    OPTIONAL
      opencv_calib3d
      opencv_dnn
    WRAP
      java
      objc
      python
      js
)

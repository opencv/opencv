Creating your own corner detector {#tutorial_generic_corner_detector}
=================================

@tableofcontents

@prev_tutorial{tutorial_good_features_to_track}
@next_tutorial{tutorial_corner_subpixels}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function @ref cv::cornerEigenValsAndVecs to find the eigenvalues and eigenvectors
    to determine if a pixel is a corner.
-   Use the OpenCV function @ref cv::cornerMinEigenVal to find the minimum eigenvalues for corner
    detection.
-   Implement our own version of the Harris detector as well as the Shi-Tomasi detector, by using
    the two functions above.

Theory
------

Code
----

@add_toggle_cpp
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/5.x/samples/cpp/tutorial_code/TrackingMotion/cornerDetector_Demo.cpp)

@include samples/cpp/tutorial_code/TrackingMotion/cornerDetector_Demo.cpp
@end_toggle

@add_toggle_java
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/5.x/samples/java/tutorial_code/TrackingMotion/generic_corner_detector/CornerDetectorDemo.java)

@include samples/java/tutorial_code/TrackingMotion/generic_corner_detector/CornerDetectorDemo.java
@end_toggle

@add_toggle_python
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/5.x/samples/python/tutorial_code/TrackingMotion/generic_corner_detector/cornerDetector_Demo.py)

@include samples/python/tutorial_code/TrackingMotion/generic_corner_detector/cornerDetector_Demo.py
@end_toggle

Explanation
-----------

Result
------

![](images/My_Harris_corner_detector_Result.jpg)

![](images/My_Shi_Tomasi_corner_detector_Result.jpg)

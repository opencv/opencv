Creating yor own corner detector {#tutorial_generic_corner_detector}
================================

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function @ref cv::cornerEigenValsAndVecs to find the eigenvalues and eigenvectors
    to determine if a pixel is a corner.
-   Use the OpenCV function @ref cv::cornerMinEigenVal to find the minimum eigenvalues for corner
    detection.
-   To implement our own version of the Harris detector as well as the Shi-Tomasi detector, by using
    the two functions above.

Theory
------

Code
----

This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/TrackingMotion/cornerDetector_Demo.cpp)

@include cpp/tutorial_code/TrackingMotion/cornerDetector_Demo.cpp

Explanation
-----------

Result
------

![](images/My_Harris_corner_detector_Result.jpg)

![](images/My_Shi_Tomasi_corner_detector_Result.jpg)

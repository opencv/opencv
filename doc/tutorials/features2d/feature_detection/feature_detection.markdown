Feature Detection {#tutorial_feature_detection}
=================

@tableofcontents

@prev_tutorial{tutorial_corner_subpixels}
@next_tutorial{tutorial_feature_description}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn how to:

-   Use the @ref cv::FeatureDetector interface in order to find interest points. Specifically:
    -   Use the cv::xfeatures2d::SURF and its function cv::xfeatures2d::SURF::detect to perform the
        detection process
    -   Use the function @ref cv::drawKeypoints to draw the detected keypoints

\warning You need the <a href="https://github.com/opencv/opencv_contrib">OpenCV contrib modules</a> to be able to use the SURF features
(alternatives are ORB, KAZE, ... features).

Theory
------

Code
----

@add_toggle_cpp
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/4.x/samples/cpp/tutorial_code/features2D/feature_detection/SURF_detection_Demo.cpp)
@include samples/cpp/tutorial_code/features2D/feature_detection/SURF_detection_Demo.cpp
@end_toggle

@add_toggle_java
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/4.x/samples/java/tutorial_code/features2D/feature_detection/SURFDetectionDemo.java)
@include samples/java/tutorial_code/features2D/feature_detection/SURFDetectionDemo.java
@end_toggle

@add_toggle_python
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/4.x/samples/python/tutorial_code/features2D/feature_detection/SURF_detection_Demo.py)
@include samples/python/tutorial_code/features2D/feature_detection/SURF_detection_Demo.py
@end_toggle

Explanation
-----------

Result
------

-#  Here is the result of the feature detection applied to the `box.png` image:

    ![](images/Feature_Detection_Result_a.jpg)

-#  And here is the result for the `box_in_scene.png` image:

    ![](images/Feature_Detection_Result_b.jpg)

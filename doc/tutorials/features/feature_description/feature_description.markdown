Feature Description {#tutorial_feature_description}
===================

@tableofcontents

@prev_tutorial{tutorial_feature_detection}
@next_tutorial{tutorial_feature_flann_matcher}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn how to:

-   Use the @ref cv::DescriptorExtractor interface in order to find the feature vector correspondent
    to the keypoints. Specifically:
    -   Use cv::xfeatures2d::SURF and its function cv::xfeatures2d::SURF::compute to perform the
        required calculations.
    -   Use a @ref cv::DescriptorMatcher to match the features vector
    -   Use the function @ref cv::drawMatches to draw the detected matches.

\warning You need the <a href="https://github.com/opencv/opencv_contrib">OpenCV contrib modules</a> to be able to use the SURF features
(alternatives are ORB, KAZE, ... features).

Theory
------

Code
----

@add_toggle_cpp
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/5.x/samples/cpp/tutorial_code/features/feature_description/SURF_matching_Demo.cpp)
@include samples/cpp/tutorial_code/features/feature_description/SURF_matching_Demo.cpp
@end_toggle

@add_toggle_java
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/5.x/samples/java/tutorial_code/features/feature_description/SURFMatchingDemo.java)
@include samples/java/tutorial_code/features/feature_description/SURFMatchingDemo.java
@end_toggle

@add_toggle_python
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/5.x/samples/python/tutorial_code/features/feature_description/SURF_matching_Demo.py)
@include samples/python/tutorial_code/features/feature_description/SURF_matching_Demo.py
@end_toggle

Explanation
-----------

Result
------

Here is the result after applying the BruteForce matcher between the two original images:

![](images/Feature_Description_BruteForce_Result.jpg)

Feature Matching with FLANN {#tutorial_feature_flann_matcher}
===========================

@tableofcontents

@prev_tutorial{tutorial_feature_description}
@next_tutorial{tutorial_feature_homography}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn how to:

-   Use the @ref cv::FlannBasedMatcher interface in order to perform a quick and efficient matching
    by using the @ref flann module

\warning You need the <a href="https://github.com/opencv/opencv_contrib">OpenCV contrib modules</a> to be able to use the SURF features
(alternatives are ORB, KAZE, ... features).

Theory
------

Classical feature descriptors (SIFT, SURF, ...) are usually compared and matched using the Euclidean distance (or L2-norm).
Since SIFT and SURF descriptors represent the histogram of oriented gradient (of the Haar wavelet response for SURF)
in a neighborhood, alternatives of the Euclidean distance are histogram-based metrics (\f$ \chi^{2} \f$, Earth Mover’s Distance (EMD), ...).

Arandjelovic et al. proposed in @cite Arandjelovic:2012:TTE:2354409.2355123 to extend to the RootSIFT descriptor:
> a square root (Hellinger) kernel instead of the standard Euclidean distance to measure the similarity between SIFT descriptors
> leads to a dramatic performance boost in all stages of the pipeline.

Binary descriptors (ORB, BRISK, ...) are matched using the <a href="https://en.wikipedia.org/wiki/Hamming_distance">Hamming distance</a>.
This distance is equivalent to count the number of different elements for binary strings (population count after applying a XOR operation):
\f[ d_{hamming} \left ( a,b \right ) = \sum_{i=0}^{n-1} \left ( a_i \oplus b_i \right ) \f]

To filter the matches, Lowe proposed in @cite Lowe04 to use a distance ratio test to try to eliminate false matches.
The distance ratio between the two nearest matches of a considered keypoint is computed and it is a good match when this value is below
a threshold. Indeed, this ratio allows helping to discriminate between ambiguous matches (distance ratio between the two nearest neighbors
is close to one) and well discriminated matches. The figure below from the SIFT paper illustrates the probability that a match is correct
based on the nearest-neighbor distance ratio test.

![](images/Feature_FlannMatcher_Lowe_ratio_test.png)

Alternative or additional filterering tests are:
-   cross check test (good match \f$ \left( f_a, f_b \right) \f$ if feature \f$ f_b \f$ is the best match for \f$ f_a \f$ in \f$ I_b \f$
    and feature \f$ f_a \f$ is the best match for \f$ f_b \f$ in \f$ I_a \f$)
-   geometric test (eliminate matches that do not fit to a geometric model, e.g. RANSAC or robust homography for planar objects)

Code
----

@add_toggle_cpp
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/4.x/samples/cpp/tutorial_code/features2D/feature_flann_matcher/SURF_FLANN_matching_Demo.cpp)
@include samples/cpp/tutorial_code/features2D/feature_flann_matcher/SURF_FLANN_matching_Demo.cpp
@end_toggle

@add_toggle_java
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/4.x/samples/java/tutorial_code/features2D/feature_flann_matcher/SURFFLANNMatchingDemo.java)
@include samples/java/tutorial_code/features2D/feature_flann_matcher/SURFFLANNMatchingDemo.java
@end_toggle

@add_toggle_python
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/4.x/samples/python/tutorial_code/features2D/feature_flann_matcher/SURF_FLANN_matching_Demo.py)
@include samples/python/tutorial_code/features2D/feature_flann_matcher/SURF_FLANN_matching_Demo.py
@end_toggle

Explanation
-----------

Result
------

-   Here is the result of the SURF feature matching using the distance ratio test:

    ![](images/Feature_FlannMatcher_Result_ratio_test.jpg)

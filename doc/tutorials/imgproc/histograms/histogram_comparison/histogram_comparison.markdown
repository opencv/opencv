Histogram Comparison {#tutorial_histogram_comparison}
====================

@tableofcontents

@prev_tutorial{tutorial_histogram_calculation}
@next_tutorial{tutorial_back_projection}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn how to:

-   Use the function @ref cv::compareHist to get a numerical parameter that express how well two
    histograms match with each other.
-   Use different metrics to compare histograms

Theory
------

-   To compare two histograms ( \f$H_{1}\f$ and \f$H_{2}\f$ ), first we have to choose a *metric*
    (\f$d(H_{1}, H_{2})\f$) to express how well both histograms match.
-   OpenCV implements the function @ref cv::compareHist to perform a comparison. It also offers 6
    different metrics to compute the matching:
    -#  **Correlation ( cv::HISTCMP_CORREL )**
        \f[d(H_1,H_2) =  \frac{\sum_I (H_1(I) - \bar{H_1}) (H_2(I) - \bar{H_2})}{\sqrt{\sum_I(H_1(I) - \bar{H_1})^2 \sum_I(H_2(I) - \bar{H_2})^2}}\f]
        where
        \f[\bar{H_k} =  \frac{1}{N} \sum _J H_k(J)\f]
        and \f$N\f$ is the total number of histogram bins.

    -#  **Chi-Square ( cv::HISTCMP_CHISQR )**
        \f[d(H_1,H_2) =  \sum _I  \frac{\left(H_1(I)-H_2(I)\right)^2}{H_1(I)}\f]

    -#  **Intersection ( cv::HISTCMP_INTERSECT )**
        \f[d(H_1,H_2) =  \sum _I  \min (H_1(I), H_2(I))\f]

    -#  **Bhattacharyya distance ( cv::HISTCMP_BHATTACHARYYA )**
        \f[d(H_1,H_2) =  \sqrt{1 - \frac{1}{\sqrt{\bar{H_1} \bar{H_2} N^2}} \sum_I \sqrt{H_1(I) \cdot H_2(I)}}\f]

    -#  **Alternative Chi-Square ( cv::HISTCMP_CHISQR_ALT )**
        \f[d(H_1,H_2) =  2 * \sum _I  \frac{\left(H_1(I)-H_2(I)\right)^2}{H_1(I)+H_2(I)}\f]

    -#  **Kullback-Leibler divergence ( cv::HISTCMP_KL_DIV )**
        \f[d(H_1,H_2) =  \sum _I H_1(I) \log \left(\frac{H_1(I)}{H_2(I)}\right)\f] */

Code
----

-   **What does this program do?**
    -   Loads a *base image* and 2 *test images* to be compared with it.
    -   Generate 1 image that is the lower half of the *base image*
    -   Convert the images to HSV format
    -   Calculate the H-S histogram for all the images and normalize them in order to compare them.
    -   Compare the histogram of the *base image* with respect to the 2 test histograms, the
        histogram of the lower half base image and with the same base image histogram.
    -   Display the numerical matching parameters obtained.

@add_toggle_cpp
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/4.x/samples/cpp/tutorial_code/Histograms_Matching/compareHist_Demo.cpp)

-   **Code at glance:**
    @include samples/cpp/tutorial_code/Histograms_Matching/compareHist_Demo.cpp
@end_toggle

@add_toggle_java
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/4.x/samples/java/tutorial_code/Histograms_Matching/histogram_comparison/CompareHistDemo.java)

-   **Code at glance:**
    @include samples/java/tutorial_code/Histograms_Matching/histogram_comparison/CompareHistDemo.java
@end_toggle

@add_toggle_python
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/4.x/samples/python/tutorial_code/Histograms_Matching/histogram_comparison/compareHist_Demo.py)

-   **Code at glance:**
    @include samples/python/tutorial_code/Histograms_Matching/histogram_comparison/compareHist_Demo.py
@end_toggle

Explanation
-----------

-   Load the base image (src_base) and the other two test images:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/compareHist_Demo.cpp Load three images with different environment settings
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/histogram_comparison/CompareHistDemo.java Load three images with different environment settings
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/histogram_comparison/compareHist_Demo.py Load three images with different environment settings
    @end_toggle

-   Convert them to HSV format:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/compareHist_Demo.cpp Convert to HSV
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/histogram_comparison/CompareHistDemo.java Convert to HSV
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/histogram_comparison/compareHist_Demo.py Convert to HSV
    @end_toggle

-   Also, create an image of half the base image (in HSV format):

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/compareHist_Demo.cpp Convert to HSV half
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/histogram_comparison/CompareHistDemo.java Convert to HSV half
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/histogram_comparison/compareHist_Demo.py Convert to HSV half
    @end_toggle

-   Initialize the arguments to calculate the histograms (bins, ranges and channels H and S).

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/compareHist_Demo.cpp Using 50 bins for hue and 60 for saturation
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/histogram_comparison/CompareHistDemo.java Using 50 bins for hue and 60 for saturation
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/histogram_comparison/compareHist_Demo.py Using 50 bins for hue and 60 for saturation
    @end_toggle

-   Calculate the Histograms for the base image, the 2 test images and the half-down base image:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/compareHist_Demo.cpp Calculate the histograms for the HSV images
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/histogram_comparison/CompareHistDemo.java Calculate the histograms for the HSV images
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/histogram_comparison/compareHist_Demo.py Calculate the histograms for the HSV images
    @end_toggle

-   Apply sequentially the 6 comparison methods between the histogram of the base image (hist_base)
    and the other histograms:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/compareHist_Demo.cpp Apply the histogram comparison methods
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/histogram_comparison/CompareHistDemo.java Apply the histogram comparison methods
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/histogram_comparison/compareHist_Demo.py Apply the histogram comparison methods
    @end_toggle

Results
-------

-#  We use as input the following images:
    ![Base_0](images/Histogram_Comparison_Source_0.jpg)
    ![Test_1](images/Histogram_Comparison_Source_1.jpg)
    ![Test_2](images/Histogram_Comparison_Source_2.jpg)
    where the first one is the base (to be compared to the others), the other 2 are the test images.
    We will also compare the first image with respect to itself and with respect of half the base
    image.

-#  We should expect a perfect match when we compare the base image histogram with itself. Also,
    compared with the histogram of half the base image, it should present a high match since both
    are from the same source. For the other two test images, we can observe that they have very
    different lighting conditions, so the matching should not be very good:

-#  Here the numeric results we got with OpenCV 4.12.0:
      *Method*          |  Base - Base |  Base - Half |  Base - Test 1 |  Base - Test 2
    ------------------- | ------------ | ------------ | -------------- | ---------------
      *Correlation*     |  1.000000    |  0.880438    |  0.20457       |  0.065752
      *Chi-square*      |  0.000000    |  0.328307    |  181.674       |  80.1494
      *Intersection*    |  1.000000    |  0.75005     |  0.315061      |  0.0908022
      *Bhattacharyya*   |  0.000000    |  0.237866    |  0.679825      |  0.873709
      *Chi-Square alt.* |  0.000000    |  0.395046    |  2.31572       |  3.41024
      *KL divergence*   |  0.000000    |  0.321064    |  2.6616        |  9.55412

    For the *Correlation* and *Intersection* methods, the higher the metric, the more accurate the
    match. As we can see, the match *base-base* is the highest of all as expected. Also we can observe
    that the match *base-half* is the second best match (as we predicted). For the other four metrics,
    the less the result, the better the match.

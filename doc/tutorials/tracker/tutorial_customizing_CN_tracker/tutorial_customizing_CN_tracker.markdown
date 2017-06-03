Customizing the CN Tracker {#tutorial_customizing_CN_tracker}
======================

Goal
----

In this tutorial you will learn how to

-   Set custom parameters for CN tracker.
-   Use your own feature-extractor function for the CN tracker.

This document contains tutorial for @ref cv::TrackerKCF.

Source Code
-----------

@include cpp/tutorial_code/tracker/customizing_CN_tracker/customizing_CN_tracker.cpp

Explanation
-----------

This part explains how to set custom parameters and use your own feature-extractor function for the CN tracker.
If you need more detailed tutorial about the basic things, please refer to @ref tutorial_introduction_to_tracker.

-#  **Set Custom Parameters**

    @snippet cpp/tutorial_code/tracker/customizing_CN_tracker/customizing_CN_tracker.cpp param

    To set custom paramters, an object should be created. Each tracker algorithm has their own parameter format.
    So, in this case we should use parameter from @ref cv::TrackerKCF since we are isterested in modifying the parameter of this tracker algorithm.

    There are several parameters can be configured as detailly explained in @ref cv::TrackerKCF::Params.
    For this tutorial, we focussed on the feature extractor functions.

    Several feature types can be used in @ref cv::TrackerKCF.
    In this case, the grayscale value (1 dimension) and color-names features (10 dimension),
    will be merged as 11 dimension feature and then compressed into 2 dimension as specified in the code.

    If you want to use another type of pre-defined feature-extractor function, you can check in @ref cv::TrackerKCF::MODE.
    We will leave the non-compressed feature as 0 since we want to use a customized function.

-#  **Using a custom function**

    You can define your own feature-extractor function for the CN tracker.
    However, you need to take care about several things:
    - The extracted feature should have same size with the size of the given bounding box (width and height).
      For the number of channels you can check the limitation in @ref cv::Mat.
    - You can only use feature that can be compared using euclidean distance.
      Feature like local binary pattern (LBP) may not suitable since it should be compared using hamming distance.

    Since the size of the extracted feature should be in the same size with the given bounding box,
    we need to take care whenever the given bounding box is partially out of range.
    In this case, we can copy part of image contained in the bounding box as shown in the snippet below.

    @snippet cpp/tutorial_code/tracker/customizing_CN_tracker/customizing_CN_tracker.cpp insideimage

    Whenever the copied image is smaller than the given bounding box,
    padding should be given to the sides where the bounding box is partially out of frame.

    @snippet cpp/tutorial_code/tracker/customizing_CN_tracker/customizing_CN_tracker.cpp padding

-#  **Defining the feature**

    In this tutorial, the extracted feature is reponse of Sobel filter in x and y direction.
    Those Sobel filter response are concatenated, resulting a feature with 2 channels.

    @snippet cpp/tutorial_code/tracker/customizing_CN_tracker/customizing_CN_tracker.cpp sobel

-#  **Post processing**

    Make sure to convert the feature into @ref cv::CV_64F data format and normalize its value with range -0.5 to 0.5

    @snippet cpp/tutorial_code/tracker/customizing_CN_tracker/customizing_CN_tracker.cpp postprocess

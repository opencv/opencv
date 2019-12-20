Thresholding Operations using inRange {#tutorial_threshold_inRange}
=====================================

@prev_tutorial{tutorial_threshold}
@next_tutorial{tutorial_filter_2d}

Goal
----

In this tutorial you will learn how to:

-   Perform basic thresholding operations using OpenCV @ref cv::inRange function.
-   Detect an object based on the range of pixel values in the HSV colorspace.

Theory
------
-   In the previous tutorial, we learnt how to perform thresholding using @ref cv::threshold function.
-   In this tutorial, we will learn how to do it using @ref cv::inRange function.
-   The concept remains the same, but now we add a range of pixel values we need.

HSV colorspace
--------------

<a href="https://en.wikipedia.org/wiki/HSL_and_HSV">HSV</a> (hue, saturation, value) colorspace
is a model to represent the colorspace similar to the RGB color model. Since the hue channel
models the color type, it is very useful in image processing tasks that need to segment objects
based on its color. Variation of the saturation goes from unsaturated to represent shades of gray and
fully saturated (no white component). Value channel describes the brightness or the intensity of the
color. Next image shows the HSV cylinder.

![By SharkDderivative work: SharkD [CC BY-SA 3.0 or GFDL], via Wikimedia Commons](images/Threshold_inRange_HSV_colorspace.jpg)

Since colors in the RGB colorspace are coded using the three channels, it is more difficult to segment
an object in the image based on its color.

![By SharkD [GFDL or CC BY-SA 4.0], from Wikimedia Commons](images/Threshold_inRange_RGB_colorspace.jpg)

Formulas used to convert from one colorspace to another colorspace using @ref cv::cvtColor function
are described in @ref imgproc_color_conversions

Code
----

@add_toggle_cpp
The tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp)
@include samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp
@end_toggle

@add_toggle_java
The tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/java/tutorial_code/ImgProc/threshold_inRange/ThresholdInRange.java)
@include samples/java/tutorial_code/ImgProc/threshold_inRange/ThresholdInRange.java
@end_toggle

@add_toggle_python
The tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/python/tutorial_code/imgProc/threshold_inRange/threshold_inRange.py)
@include samples/python/tutorial_code/imgProc/threshold_inRange/threshold_inRange.py
@end_toggle

Explanation
-----------

Let's check the general structure of the program:
-   Capture the video stream from default or supplied capturing device.

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp cap
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgProc/threshold_inRange/ThresholdInRange.java cap
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/threshold_inRange/threshold_inRange.py cap
    @end_toggle

-   Create a window to display the default frame and the threshold frame.

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp window
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgProc/threshold_inRange/ThresholdInRange.java window
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/threshold_inRange/threshold_inRange.py window
    @end_toggle

-   Create the trackbars to set the range of HSV values

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp trackbar
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgProc/threshold_inRange/ThresholdInRange.java trackbar
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/threshold_inRange/threshold_inRange.py trackbar
    @end_toggle

-   Until the user want the program to exit do the following

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp while
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgProc/threshold_inRange/ThresholdInRange.java while
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/threshold_inRange/threshold_inRange.py while
    @end_toggle

-   Show the images

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp show
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgProc/threshold_inRange/ThresholdInRange.java show
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/threshold_inRange/threshold_inRange.py show
    @end_toggle

-   For a trackbar which controls the lower range, say for example hue value:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp low
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgProc/threshold_inRange/ThresholdInRange.java low
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/threshold_inRange/threshold_inRange.py low
    @end_toggle
    @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp low

-   For a trackbar which controls the upper range, say for example hue value:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp high
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgProc/threshold_inRange/ThresholdInRange.java high
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/threshold_inRange/threshold_inRange.py high
    @end_toggle

-   It is necessary to find the maximum and minimum value to avoid discrepancies such as
    the high value of threshold becoming less than the low value.

Results
-------

-  After compiling this program, run it. The program will open two windows

-  As you set the range values from the trackbar, the resulting frame will be visible in the other window.

    ![](images/Threshold_inRange_Tutorial_Result_input.jpeg)
    ![](images/Threshold_inRange_Tutorial_Result_output.jpeg)

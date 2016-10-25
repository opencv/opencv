Thresholding Operations using inRange {#tutorial_threshold_inRange}
=============================

Goal
----

In this tutorial you will learn how to:

-   Perform basic thresholding operations using OpenCV function @ref cv::inRange
-   Detect an object based on the range of pixel values it has

Theory
-----------
-   In the previous tutorial, we learnt how perform thresholding using @ref cv::threshold function.
-   In this tutorial, we will learn how to do it using @ref cv::inRange function.
-   The concept remains same, but now we add a range of pixel values we need.

Code
----

The tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp)
@include samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp

Explanation
-----------

-#  Let's check the general structure of the program:
    -   Create two Matrix elements to store the frames
        @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp mat
    -   Capture the video stream from default capturing device.
        @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp cap
    -   Create a window to display the default frame and the threshold frame.
        @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp window
    -   Create trackbars to set the range of RGB values
        @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp trackbar
    -   Until the user want the program to exit do the following
        @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp while
    -   Show the images
        @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp show
    -   For a trackbar which controls the lower range, say for example Red value:
        @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp low
    -   For a trackbar which controls the upper range, say for example Red value:
        @snippet samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp high
    -   It is necessary to find the maximum and minimum value to avoid discrepancies such as
        the high value of threshold becoming less the low value.

Results
-------

-#  After compiling this program, run it. The program will open two windows

-#  As you set the RGB range values from the trackbar, the resulting frame will be visible in the other window.

    ![](images/Threshold_inRange_Tutorial_Result_input.jpeg)
    ![](images/Threshold_inRange_Tutorial_Result_output.jpeg)

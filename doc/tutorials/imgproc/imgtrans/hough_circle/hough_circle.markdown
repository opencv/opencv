Hough Circle Transform {#tutorial_hough_circle}
======================

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function @ref cv::HoughCircles to detect circles in an image.

Theory
------

### Hough Circle Transform

-   The Hough Circle Transform works in a *roughly* analogous way to the Hough Line Transform
    explained in the previous tutorial.
-   In the line detection case, a line was defined by two parameters \f$(r, \theta)\f$. In the circle
    case, we need three parameters to define a circle:

    \f[C : ( x_{center}, y_{center}, r )\f]

    where \f$(x_{center}, y_{center})\f$ define the center position (green point) and \f$r\f$ is the radius,
    which allows us to completely define a circle, as it can be seen below:

    ![](images/Hough_Circle_Tutorial_Theory_0.jpg)

-   For sake of efficiency, OpenCV implements a detection method slightly trickier than the standard
    Hough Transform: *The Hough gradient method*, which is made up of two main stages. The first
    stage involves edge detection and finding the possible circle centers and the second stage finds
    the best radius for each candidate center. For more details, please check the book *Learning
    OpenCV* or your favorite Computer Vision bibliography

Code
----

-#  **What does this program do?**
    -   Loads an image and blur it to reduce the noise
    -   Applies the *Hough Circle Transform* to the blurred image .
    -   Display the detected circle in a window.

-#  The sample code that we will explain can be downloaded from [here](https://github.com/opencv/opencv/tree/master/samples/cpp/houghcircles.cpp).
    A slightly fancier version (which shows trackbars for
    changing the threshold values) can be found [here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/ImgTrans/HoughCircle_Demo.cpp).
    @include samples/cpp/houghcircles.cpp

Explanation
-----------

-#  Load an image
    @snippet samples/cpp/houghcircles.cpp load
-#  Convert it to grayscale:
    @snippet samples/cpp/houghcircles.cpp convert_to_gray
-#  Apply a Median blur to reduce noise and avoid false circle detection:
    @snippet samples/cpp/houghcircles.cpp reduce_noise
-#  Proceed to apply Hough Circle Transform:
    @snippet samples/cpp/houghcircles.cpp houghcircles
    with the arguments:

    -   *gray*: Input image (grayscale).
    -   *circles*: A vector that stores sets of 3 values: \f$x_{c}, y_{c}, r\f$ for each detected
        circle.
    -   *HOUGH_GRADIENT*: Define the detection method. Currently this is the only one available in
        OpenCV.
    -   *dp = 1*: The inverse ratio of resolution.
    -   *min_dist = gray.rows/16*: Minimum distance between detected centers.
    -   *param_1 = 200*: Upper threshold for the internal Canny edge detector.
    -   *param_2* = 100\*: Threshold for center detection.
    -   *min_radius = 0*: Minimum radio to be detected. If unknown, put zero as default.
    -   *max_radius = 0*: Maximum radius to be detected. If unknown, put zero as default.

-#  Draw the detected circles:
    @snippet samples/cpp/houghcircles.cpp draw
    You can see that we will draw the circle(s) on red and the center(s) with a small green dot

-#  Display the detected circle(s) and wait for the user to exit the program:
    @snippet samples/cpp/houghcircles.cpp display

Result
------

The result of running the code above with a test image is shown below:

![](images/Hough_Circle_Tutorial_Result.jpg)

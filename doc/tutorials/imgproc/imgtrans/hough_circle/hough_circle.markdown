Hough Circle Transform {#tutorial_hough_circle}
======================

@prev_tutorial{tutorial_hough_lines}
@next_tutorial{tutorial_remap}

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function **HoughCircles()** to detect circles in an image.

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

####  What does this program do?
-   Loads an image and blur it to reduce the noise
-   Applies the *Hough Circle Transform* to the blurred image .
-   Display the detected circle in a window.

Code
----

@add_toggle_cpp
The sample code that we will explain can be downloaded from
[here](https://raw.githubusercontent.com/opencv/opencv/master/samples/cpp/tutorial_code/ImgTrans/houghcircles.cpp).
A slightly fancier version (which shows trackbars for changing the threshold values) can be found
[here](https://raw.githubusercontent.com/opencv/opencv/master/samples/cpp/tutorial_code/ImgTrans/HoughCircle_Demo.cpp).
@include samples/cpp/tutorial_code/ImgTrans/houghcircles.cpp
@end_toggle

@add_toggle_java
The sample code that we will explain can be downloaded from
[here](https://raw.githubusercontent.com/opencv/opencv/master/samples/java/tutorial_code/ImgTrans/HoughCircle/HoughCircles.java).
@include samples/java/tutorial_code/ImgTrans/HoughCircle/HoughCircles.java
@end_toggle

@add_toggle_python
The sample code that we will explain can be downloaded from
[here](https://raw.githubusercontent.com/opencv/opencv/master/samples/python/tutorial_code/ImgTrans/HoughCircle/hough_circle.py).
@include samples/python/tutorial_code/ImgTrans/HoughCircle/hough_circle.py
@end_toggle

Explanation
-----------

The image we used can be found [here](https://raw.githubusercontent.com/opencv/opencv/master/samples/data/smarties.png)

####  Load an image:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghcircles.cpp load
@end_toggle

@add_toggle_java
@snippet samples/python/tutorial_code/ImgTrans/HoughCircle/hough_circle.py load
@end_toggle

@add_toggle_python
@snippet samples/java/tutorial_code/ImgTrans/HoughCircle/HoughCircles.java load
@end_toggle

####  Convert it to grayscale:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghcircles.cpp convert_to_gray
@end_toggle

@add_toggle_java
@snippet samples/python/tutorial_code/ImgTrans/HoughCircle/hough_circle.py convert_to_gray
@end_toggle

@add_toggle_python
@snippet samples/java/tutorial_code/ImgTrans/HoughCircle/HoughCircles.java convert_to_gray
@end_toggle

#### Apply a Median blur to reduce noise and avoid false circle detection:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghcircles.cpp reduce_noise
@end_toggle

@add_toggle_java
@snippet samples/python/tutorial_code/ImgTrans/HoughCircle/hough_circle.py reduce_noise
@end_toggle

@add_toggle_python
@snippet samples/java/tutorial_code/ImgTrans/HoughCircle/HoughCircles.java reduce_noise
@end_toggle

#### Proceed to apply Hough Circle Transform:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghcircles.cpp houghcircles
@end_toggle

@add_toggle_java
@snippet samples/python/tutorial_code/ImgTrans/HoughCircle/hough_circle.py houghcircles
@end_toggle

@add_toggle_python
@snippet samples/java/tutorial_code/ImgTrans/HoughCircle/HoughCircles.java houghcircles
@end_toggle

-   with the arguments:

    -   *gray*: Input image (grayscale).
    -   *circles*: A vector that stores sets of 3 values: \f$x_{c}, y_{c}, r\f$ for each detected
        circle.
    -   *HOUGH_GRADIENT*: Define the detection method. Currently this is the only one available in
        OpenCV.
    -   *dp = 1*: The inverse ratio of resolution.
    -   *min_dist = gray.rows/16*: Minimum distance between detected centers.
    -   *param_1 = 200*: Upper threshold for the internal Canny edge detector.
    -   *param_2* = 100\*: Threshold for center detection.
    -   *min_radius = 0*: Minimum radius to be detected. If unknown, put zero as default.
    -   *max_radius = 0*: Maximum radius to be detected. If unknown, put zero as default.

####  Draw the detected circles:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghcircles.cpp draw
@end_toggle

@add_toggle_java
@snippet samples/python/tutorial_code/ImgTrans/HoughCircle/hough_circle.py draw
@end_toggle

@add_toggle_python
@snippet samples/java/tutorial_code/ImgTrans/HoughCircle/HoughCircles.java draw
@end_toggle

You can see that we will draw the circle(s) on red and the center(s) with a small green dot

####  Display the detected circle(s) and wait for the user to exit the program:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/houghcircles.cpp display
@end_toggle

@add_toggle_java
@snippet samples/python/tutorial_code/ImgTrans/HoughCircle/hough_circle.py display
@end_toggle

@add_toggle_python
@snippet samples/java/tutorial_code/ImgTrans/HoughCircle/HoughCircles.java display
@end_toggle

Result
------

The result of running the code above with a test image is shown below:

![](images/Hough_Circle_Tutorial_Result.png)

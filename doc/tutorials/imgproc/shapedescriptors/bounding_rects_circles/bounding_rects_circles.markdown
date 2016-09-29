Creating Bounding boxes and circles for contours {#tutorial_bounding_rects_circles}
================================================

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function @ref cv::boundingRect
-   Use the OpenCV function @ref cv::minEnclosingCircle

Theory
------

Code
----

This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp)
@include samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp

Explanation
-----------

The main function is rather simple, as follows from the comments we do the following:
-#  Open the image, convert it into grayscale and blur it to get rid of the noise.
    @snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp setup
-# Create a window with header "Source" and display the source file in it.
    @snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp createWindow
-# Create a trackbar on the source_window and assign a callback function to it
   In general callback functions are used to react to some kind of signal, in our
   case it's trackbar's state change.
   @snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp taskbar
-# Explicit one-time call of `thresh_callback` is necessary to display
   the "Contours" window simultaniously with the "Source" window.
   @snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp callback00
-# Wait for user to close the windows.
   @snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp waitForIt


The callback function `thresh_callback` does all the interesting job.


-# Writes to `threshold_output` the threshold of the grayscale picture (you can check out about thresholding @ref tutorial_threshold "here").
   @snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp threshold
-# Finds contours and saves them to the vectors `contour` and `hierarchy`.
    @snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp findContours
-# For every found contour we now apply approximation to polygons
   with accuracy +-3 and stating that the curve must me closed.

   After that we find a bounding rect for every polygon and save it to `boundRect`.

   At last we find a minimum enclosing circle for every polygon and
   save it to `center` and `radius` vectors.
   @snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp allthework

We found everything we need, all we have to do is to draw.

-# Create new Mat of unsigned 8-bit chars, filled with zeros.
   It will contain all the drawings we are going to make (rects and circles).
   @snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp zeroMat
-# For every contour: pick a random color, draw the contour, the bounding rectangle and
   the minimal enclosing circle with it,
   @snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp forContour
-# Display the results: create a new window "Contours" and show everything we added to drawings on it.
   @snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp showDrawings

Result
------

Here it is:
![](images/Bounding_Rects_Circles_Source_Image.jpg)
![](images/Bounding_Rects_Circles_Result.jpg)

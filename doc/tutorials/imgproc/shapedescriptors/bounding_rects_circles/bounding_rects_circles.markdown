Creating Bounding boxes and circles for contours {#tutorial_bounding_rects_circles}
================================================

@prev_tutorial{tutorial_hull}
@next_tutorial{tutorial_bounding_rotated_ellipses}

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function @ref cv::boundingRect
-   Use the OpenCV function @ref cv::minEnclosingCircle

Theory
------

Code
----

@add_toggle_cpp
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp)
@include samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp
@end_toggle

@add_toggle_java
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/java/tutorial_code/ShapeDescriptors/bounding_rects_circles/GeneralContoursDemo1.java)
@include samples/java/tutorial_code/ShapeDescriptors/bounding_rects_circles/GeneralContoursDemo1.java
@end_toggle

@add_toggle_python
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/python/tutorial_code/ShapeDescriptors/bounding_rects_circles/generalContours_demo1.py)
@include samples/python/tutorial_code/ShapeDescriptors/bounding_rects_circles/generalContours_demo1.py
@end_toggle

Explanation
-----------

The main function is rather simple, as follows from the comments we do the following:
-   Open the image, convert it into grayscale and blur it to get rid of the noise.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp setup
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ShapeDescriptors/bounding_rects_circles/GeneralContoursDemo1.java setup
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ShapeDescriptors/bounding_rects_circles/generalContours_demo1.py setup
@end_toggle

-  Create a window with header "Source" and display the source file in it.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp createWindow
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ShapeDescriptors/bounding_rects_circles/GeneralContoursDemo1.java createWindow
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ShapeDescriptors/bounding_rects_circles/generalContours_demo1.py createWindow
@end_toggle

-  Create a trackbar on the `source_window` and assign a callback function to it.
   In general callback functions are used to react to some kind of signal, in our
   case it's trackbar's state change.
   Explicit one-time call of `thresh_callback` is necessary to display
   the "Contours" window simultaniously with the "Source" window.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp trackbar
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ShapeDescriptors/bounding_rects_circles/GeneralContoursDemo1.java trackbar
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ShapeDescriptors/bounding_rects_circles/generalContours_demo1.py trackbar
@end_toggle

The callback function does all the interesting job.

-  Use @ref cv::Canny to detect edges in the images.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp Canny
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ShapeDescriptors/bounding_rects_circles/GeneralContoursDemo1.java Canny
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ShapeDescriptors/bounding_rects_circles/generalContours_demo1.py Canny
@end_toggle

-  Finds contours and saves them to the vectors `contour` and `hierarchy`.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp findContours
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ShapeDescriptors/bounding_rects_circles/GeneralContoursDemo1.java findContours
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ShapeDescriptors/bounding_rects_circles/generalContours_demo1.py findContours
@end_toggle

-  For every found contour we now apply approximation to polygons
   with accuracy +-3 and stating that the curve must be closed.
   After that we find a bounding rect for every polygon and save it to `boundRect`.
   At last we find a minimum enclosing circle for every polygon and
   save it to `center` and `radius` vectors.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp allthework
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ShapeDescriptors/bounding_rects_circles/GeneralContoursDemo1.java allthework
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ShapeDescriptors/bounding_rects_circles/generalContours_demo1.py allthework
@end_toggle

We found everything we need, all we have to do is to draw.

-  Create new Mat of unsigned 8-bit chars, filled with zeros.
   It will contain all the drawings we are going to make (rects and circles).

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp zeroMat
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ShapeDescriptors/bounding_rects_circles/GeneralContoursDemo1.java zeroMat
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ShapeDescriptors/bounding_rects_circles/generalContours_demo1.py zeroMat
@end_toggle

-  For every contour: pick a random color, draw the contour, the bounding rectangle and
   the minimal enclosing circle with it.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp forContour
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ShapeDescriptors/bounding_rects_circles/GeneralContoursDemo1.java forContour
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ShapeDescriptors/bounding_rects_circles/generalContours_demo1.py forContour
@end_toggle

-  Display the results: create a new window "Contours" and show everything we added to drawings on it.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo1.cpp showDrawings
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ShapeDescriptors/bounding_rects_circles/GeneralContoursDemo1.java showDrawings
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ShapeDescriptors/bounding_rects_circles/generalContours_demo1.py showDrawings
@end_toggle

Result
------

Here it is:
![](images/Bounding_Rects_Circles_Source_Image.jpg)
![](images/Bounding_Rects_Circles_Result.jpg)

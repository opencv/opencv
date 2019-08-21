Creating Bounding rotated boxes and ellipses for contours {#tutorial_bounding_rotated_ellipses}
=========================================================

@prev_tutorial{tutorial_bounding_rects_circles}
@next_tutorial{tutorial_moments}

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function @ref cv::minAreaRect
-   Use the OpenCV function @ref cv::fitEllipse

Theory
------

Code
----

@add_toggle_cpp
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo2.cpp)
@include samples/cpp/tutorial_code/ShapeDescriptors/generalContours_demo2.cpp
@end_toggle

@add_toggle_java
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/java/tutorial_code/ShapeDescriptors/bounding_rotated_ellipses/GeneralContoursDemo2.java)
@include samples/java/tutorial_code/ShapeDescriptors/bounding_rotated_ellipses/GeneralContoursDemo2.java
@end_toggle

@add_toggle_python
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/python/tutorial_code/ShapeDescriptors/bounding_rotated_ellipses/generalContours_demo2.py)
@include samples/python/tutorial_code/ShapeDescriptors/bounding_rotated_ellipses/generalContours_demo2.py
@end_toggle

Explanation
-----------

Result
------

Here it is:
![](images/Bounding_Rotated_Ellipses_Source_Image.jpg)
![](images/Bounding_Rotated_Ellipses_Result.jpg)

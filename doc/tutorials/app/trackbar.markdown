Adding a Trackbar to our applications! {#tutorial_trackbar}
======================================

@tableofcontents

@next_tutorial{tutorial_raster_io_gdal}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |


-   In the previous tutorials (about @ref tutorial_adding_images and the @ref tutorial_basic_linear_transform)
    you might have noted that we needed to give some **input** to our programs, such
    as \f$\alpha\f$ and \f$beta\f$. We accomplished that by entering this data using the Terminal.
-   Well, it is time to use some fancy GUI tools. OpenCV provides some GUI utilities (**highgui** module)
    for you. An example of this is a **Trackbar**.

    ![](images/Adding_Trackbars_Tutorial_Trackbar.png)

-   In this tutorial we will just modify our two previous programs so that they get the input
    information from the trackbar.

Goals
-----

In this tutorial you will learn how to:

-   Add a Trackbar in an OpenCV window by using @ref cv::createTrackbar

Code
----

Let's modify the program made in the tutorial @ref tutorial_adding_images. We will let the user enter the
\f$\alpha\f$ value by using the Trackbar.

@add_toggle_cpp
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/HighGUI/AddingImagesTrackbar.cpp)
@include cpp/tutorial_code/HighGUI/AddingImagesTrackbar.cpp
@end_toggle

@add_toggle_java
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/java/tutorial_code/highgui/trackbar/AddingImagesTrackbar.java)
@include java/tutorial_code/highgui/trackbar/AddingImagesTrackbar.java
@end_toggle

@add_toggle_python
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/python/tutorial_code/highgui/trackbar/AddingImagesTrackbar.py)
@include python/tutorial_code/highgui/trackbar/AddingImagesTrackbar.py
@end_toggle

Explanation
-----------

We only analyze the code that is related to Trackbar:

-  First, we load two images, which are going to be blended.

@add_toggle_cpp
@snippet cpp/tutorial_code/HighGUI/AddingImagesTrackbar.cpp load
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/highgui/trackbar/AddingImagesTrackbar.java load
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/highgui/trackbar/AddingImagesTrackbar.py load
@end_toggle

-  To create a trackbar, first we have to create the window in which it is going to be located. So:

@add_toggle_cpp
@snippet cpp/tutorial_code/HighGUI/AddingImagesTrackbar.cpp window
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/highgui/trackbar/AddingImagesTrackbar.java window
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/highgui/trackbar/AddingImagesTrackbar.py window
@end_toggle

-  Now we can create the Trackbar:

@add_toggle_cpp
@snippet cpp/tutorial_code/HighGUI/AddingImagesTrackbar.cpp create_trackbar
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/highgui/trackbar/AddingImagesTrackbar.java create_trackbar
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/highgui/trackbar/AddingImagesTrackbar.py create_trackbar
@end_toggle

Note the following (C++ code):
    -   Our Trackbar has a label **TrackbarName**
    -   The Trackbar is located in the window named **Linear Blend**
    -   The Trackbar values will be in the range from \f$0\f$ to **alpha_slider_max** (the minimum
        limit is always **zero**).
    -   The numerical value of Trackbar is stored in **alpha_slider**
    -   Whenever the user moves the Trackbar, the callback function **on_trackbar** is called

Finally, we have to define the callback function **on_trackbar** for C++ and Python code, using an anonymous inner class listener in Java

@add_toggle_cpp
@snippet cpp/tutorial_code/HighGUI/AddingImagesTrackbar.cpp on_trackbar
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/highgui/trackbar/AddingImagesTrackbar.java on_trackbar
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/highgui/trackbar/AddingImagesTrackbar.py on_trackbar
@end_toggle

Note that (C++ code):
    -   We use the value of **alpha_slider** (integer) to get a double value for **alpha**.
    -   **alpha_slider** is updated each time the trackbar is displaced by the user.
    -   We define *src1*, *src2*, *dist*, *alpha*, *alpha_slider* and *beta* as global variables,
        so they can be used everywhere.

Result
------

-   Our program produces the following output:

    ![](images/Adding_Trackbars_Tutorial_Result_0.jpg)

-   As a manner of practice, you can also add two trackbars for the program made in
    @ref tutorial_basic_linear_transform. One trackbar to set \f$\alpha\f$ and another for set \f$\beta\f$. The output might
    look like:

    ![](images/Adding_Trackbars_Tutorial_Result_1.jpg)

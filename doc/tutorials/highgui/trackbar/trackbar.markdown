Adding a Trackbar to our applications! {#tutorial_trackbar}
======================================

-   In the previous tutorials (about *linear blending* and the *brightness and contrast
    adjustments*) you might have noted that we needed to give some **input** to our programs, such
    as \f$\alpha\f$ and \f$beta\f$. We accomplished that by entering this data using the Terminal
-   Well, it is time to use some fancy GUI tools. OpenCV provides some GUI utilities (*highgui.hpp*)
    for you. An example of this is a **Trackbar**

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
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/HighGUI/AddingImagesTrackbar.cpp)
@include cpp/tutorial_code/HighGUI/AddingImagesTrackbar.cpp

Explanation
-----------

We only analyze the code that is related to Trackbar:

-#  First, we load two images, which are going to be blended.
    @snippet cpp/tutorial_code/HighGUI/AddingImagesTrackbar.cpp load

-#  To create a trackbar, first we have to create the window in which it is going to be located. So:
    @snippet cpp/tutorial_code/HighGUI/AddingImagesTrackbar.cpp window

-#  Now we can create the Trackbar:
    @snippet cpp/tutorial_code/HighGUI/AddingImagesTrackbar.cpp create_trackbar

    Note the following:

    -   Our Trackbar has a label **TrackbarName**
    -   The Trackbar is located in the window named **Linear Blend**
    -   The Trackbar values will be in the range from \f$0\f$ to **alpha_slider_max** (the minimum
        limit is always **zero**).
    -   The numerical value of Trackbar is stored in **alpha_slider**
    -   Whenever the user moves the Trackbar, the callback function **on_trackbar** is called

-#  Finally, we have to define the callback function **on_trackbar**
    @snippet cpp/tutorial_code/HighGUI/AddingImagesTrackbar.cpp on_trackbar

    Note that:
    -   We use the value of **alpha_slider** (integer) to get a double value for **alpha**.
    -   **alpha_slider** is updated each time the trackbar is displaced by the user.
    -   We define *src1*, *src2*, *dist*, *alpha*, *alpha_slider* and *beta* as global variables,
        so they can be used everywhere.

Result
------

-   Our program produces the following output:

    ![](images/Adding_Trackbars_Tutorial_Result_0.jpg)

-   As a manner of practice, you can also add two trackbars for the program made in
    @ref tutorial_basic_linear_transform. One trackbar to set \f$\alpha\f$ and another for \f$\beta\f$. The output might
    look like:

    ![](images/Adding_Trackbars_Tutorial_Result_1.jpg)

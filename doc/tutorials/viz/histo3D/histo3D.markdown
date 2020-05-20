Creating a 3D histogram {#tutorial_histo3D}
================

@prev_tutorial{tutorial_creating_widgets}

Goal
----

In this tutorial you will learn how to

-   Create your own callback keyboard function for viz window.
-   Show your 3D histogram in a viz window.

Code
----

You can download the code from [here ](https://github.com/opencv/opencv/tree/3.4/samples/cpp/tutorial_code/viz/histo3D.cpp).
@include samples/cpp/tutorial_code/viz/histo3D.cpp

Explanation
-----------

Here is the general structure of the program:

-   You can give full path to an image in command line
    @snippet histo3D.cpp command_line_parser

    or without path, a synthetic image is generated with pixel values are a gaussian distribution @ref cv::RNG::fill center(60+/-10,40+/-5,50+/-20) in first quadrant,
    (160+/-20,10+/-5,50+/-10) in second quadrant, (90+/-10,100+/-20,50+/-20) in third quadrant, (100+/-10,10+/-5,150+/-40) in last quadrant.
    @snippet histo3D.cpp synthetic_image
    Image tridimensional histogram is calculated using opencv @ref cv::calcHist and @ref cv::normalize between 0 and 100.
    @snippet histo3D.cpp calchist_for_histo3d
    channel are 2, 1 and 0 to synchronise color with Viz axis color in objetc cv::viz::WCoordinateSystem.

    A slidebar is inserted in image window. Init slidebar value is 90, it means that only histogram cell greater than 9/100000.0 (23 pixels for an 512X512 pixels) will be display.
    @snippet histo3D.cpp slide_bar_for_thresh
    We are ready to open a viz window with a callback function to capture keyboard event in viz window. Using @ref cv::viz::Viz3d::spinOnce enable keyboard event to be capture in @ref cv::imshow window too.
    @snippet histo3D.cpp manage_viz_imshow_window
    The function DrawHistogram3D processes histogram Mat to display it in a Viz window. Number of plan, row and column in [three dimensional Mat](@ref CVMat_Details ) can be found using  this code :
    @snippet histo3D.cpp get_cube_size
    To get histogram value at a specific location we use @ref cv::Mat::at(int i0,int i1, int i2)  method with three arguments k, i and j where k is plane number, i row number and j column number.
    @snippet histo3D.cpp get_cube_values

-   Callback function
    Principle are as mouse callback function. Key code pressed is in field code of class @ref cv::viz::KeyboardEvent.
    @snippet histo3D.cpp viz_keyboard_callback

Results
-------

Here is the result of the program with no argument and threshold equal to 50.

![](images/histo50.png)

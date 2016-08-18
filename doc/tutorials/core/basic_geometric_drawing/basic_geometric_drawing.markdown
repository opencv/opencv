Basic Drawing {#tutorial_basic_geometric_drawing}
=============

Goals
-----

In this tutorial you will learn how to:

-   Use @ref cv::Point to define 2D points in an image.
-   Use @ref cv::Scalar and why it is useful
-   Draw a **line** by using the OpenCV function @ref cv::line
-   Draw an **ellipse** by using the OpenCV function @ref cv::ellipse
-   Draw a **rectangle** by using the OpenCV function @ref cv::rectangle
-   Draw a **circle** by using the OpenCV function @ref cv::circle
-   Draw a **filled polygon** by using the OpenCV function @ref cv::fillPoly

OpenCV Theory
-------------

For this tutorial, we will heavily use two structures: @ref cv::Point and @ref cv::Scalar :

### Point

It represents a 2D point, specified by its image coordinates \f$x\f$ and \f$y\f$. We can define it as:
@code{.cpp}
Point pt;
pt.x = 10;
pt.y = 8;
@endcode
or
@code{.cpp}
Point pt =  Point(10, 8);
@endcode
### Scalar

-   Represents a 4-element vector. The type Scalar is widely used in OpenCV for passing pixel
    values.
-   In this tutorial, we will use it extensively to represent BGR color values (3 parameters). It is
    not necessary to define the last argument if it is not going to be used.
-   Let's see an example, if we are asked for a color argument and we give:
    @code{.cpp}
    Scalar( a, b, c )
    @endcode
    We would be defining a BGR color such as: *Blue = a*, *Green = b* and *Red = c*

Code
----

-   This code is in your OpenCV sample folder. Otherwise you can grab it from
    [here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/core/Matrix/Drawing_1.cpp)
    @include samples/cpp/tutorial_code/core/Matrix/Drawing_1.cpp

Explanation
-----------

-#  Since we plan to draw two examples (an atom and a rook), we have to create two images and two
    windows to display them.
    @snippet cpp/tutorial_code/core/Matrix/Drawing_1.cpp create_images

-#  We created functions to draw different geometric shapes. For instance, to draw the atom we used
    *MyEllipse* and *MyFilledCircle*:
    @snippet cpp/tutorial_code/core/Matrix/Drawing_1.cpp draw_atom

-#  And to draw the rook we employed *MyLine*, *rectangle* and a *MyPolygon*:
    @snippet cpp/tutorial_code/core/Matrix/Drawing_1.cpp draw_rook

-#  Let's check what is inside each of these functions:
    -   *MyLine*
        @snippet cpp/tutorial_code/core/Matrix/Drawing_1.cpp myline

        As we can see, *MyLine* just call the function @ref cv::line , which does the following:

        -   Draw a line from Point **start** to Point **end**
        -   The line is displayed in the image **img**
        -   The line color is defined by **Scalar( 0, 0, 0)** which is the RGB value correspondent
            to **Black**
        -   The line thickness is set to **thickness** (in this case 2)
        -   The line is a 8-connected one (**lineType** = 8)
    -   *MyEllipse*
        @snippet cpp/tutorial_code/core/Matrix/Drawing_1.cpp myellipse

        From the code above, we can observe that the function @ref cv::ellipse draws an ellipse such
        that:

        -   The ellipse is displayed in the image **img**
        -   The ellipse center is located in the point **(w/2, w/2)** and is enclosed in a box
            of size **(w/4, w/16)**
        -   The ellipse is rotated **angle** degrees
        -   The ellipse extends an arc between **0** and **360** degrees
        -   The color of the figure will be **Scalar( 255, 0, 0)** which means blue in BGR value.
        -   The ellipse's **thickness** is 2.
    -   *MyFilledCircle*
        @snippet cpp/tutorial_code/core/Matrix/Drawing_1.cpp myfilledcircle

        Similar to the ellipse function, we can observe that *circle* receives as arguments:

        -   The image where the circle will be displayed (**img**)
        -   The center of the circle denoted as the Point **center**
        -   The radius of the circle: **w/32**
        -   The color of the circle: **Scalar(0, 0, 255)** which means *Red* in BGR
        -   Since **thickness** = -1, the circle will be drawn filled.
    -   *MyPolygon*
        @snippet cpp/tutorial_code/core/Matrix/Drawing_1.cpp mypolygon

        To draw a filled polygon we use the function @ref cv::fillPoly . We note that:

        -   The polygon will be drawn on **img**
        -   The vertices of the polygon are the set of points in **ppt**
        -   The total number of vertices to be drawn are **npt**
        -   The number of polygons to be drawn is only **1**
        -   The color of the polygon is defined by **Scalar( 255, 255, 255)**, which is the BGR
            value for *white*
    -   *rectangle*
        @snippet cpp/tutorial_code/core/Matrix/Drawing_1.cpp rectangle

        Finally we have the @ref cv::rectangle function (we did not create a special function for
        this guy). We note that:

        -   The rectangle will be drawn on **rook_image**
        -   Two opposite vertices of the rectangle are defined by *\* Point( 0, 7*w/8 )*\*
            andPoint( w, w)*\*
        -   The color of the rectangle is given by **Scalar(0, 255, 255)** which is the BGR value
            for *yellow*
        -   Since the thickness value is given by **FILLED (-1)**, the rectangle will be filled.

Result
------

Compiling and running your program should give you a result like this:

![](images/Drawing_1_Tutorial_Result_0.png)

Basic Drawing {#tutorial_basic_geometric_drawing}
=============

@tableofcontents

@next_tutorial{tutorial_random_generator_and_text}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

Goals
-----

In this tutorial you will learn how to:

-   Draw a **line** by using the OpenCV function **line()**
-   Draw an **ellipse** by using the OpenCV function **ellipse()**
-   Draw a **rectangle** by using the OpenCV function **rectangle()**
-   Draw a **circle** by using the OpenCV function **circle()**
-   Draw a **filled polygon** by using the OpenCV function **fillPoly()**

@add_toggle_cpp
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
@end_toggle

@add_toggle_java
OpenCV Theory
-------------

For this tutorial, we will heavily use two structures: @ref cv::Point and @ref cv::Scalar :

### Point

It represents a 2D point, specified by its image coordinates \f$x\f$ and \f$y\f$. We can define it as:
@code{.java}
Point pt = new Point();
pt.x = 10;
pt.y = 8;
@endcode
or
@code{.java}
Point pt = new Point(10, 8);
@endcode
### Scalar

-   Represents a 4-element vector. The type Scalar is widely used in OpenCV for passing pixel
    values.
-   In this tutorial, we will use it extensively to represent BGR color values (3 parameters). It is
    not necessary to define the last argument if it is not going to be used.
-   Let's see an example, if we are asked for a color argument and we give:
    @code{.java}
    Scalar( a, b, c )
    @endcode
    We would be defining a BGR color such as: *Blue = a*, *Green = b* and *Red = c*
@end_toggle

Code
----

@add_toggle_cpp
-   This code is in your OpenCV sample folder. Otherwise you can grab it from
    [here](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_1.cpp)
    @include samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_1.cpp
@end_toggle

@add_toggle_java
-   This code is in your OpenCV sample folder. Otherwise you can grab it from
    [here](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/java/tutorial_code/ImgProc/BasicGeometricDrawing/BasicGeometricDrawing.java)
    @include samples/java/tutorial_code/ImgProc/BasicGeometricDrawing/BasicGeometricDrawing.java
@end_toggle

@add_toggle_python
-   This code is in your OpenCV sample folder. Otherwise you can grab it from
    [here](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/python/tutorial_code/imgProc/BasicGeometricDrawing/basic_geometric_drawing.py)
    @include samples/python/tutorial_code/imgProc/BasicGeometricDrawing/basic_geometric_drawing.py
@end_toggle

Explanation
-----------

Since we plan to draw two examples (an atom and a rook), we have to create two images and two
windows to display them.
@add_toggle_cpp
@snippet cpp/tutorial_code/ImgProc/basic_drawing/Drawing_1.cpp create_images
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgProc/BasicGeometricDrawing/BasicGeometricDrawing.java create_images
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgProc/BasicGeometricDrawing/basic_geometric_drawing.py create_images
@end_toggle

We created functions to draw different geometric shapes. For instance, to draw the atom we used
**MyEllipse** and **MyFilledCircle**:
@add_toggle_cpp
@snippet cpp/tutorial_code/ImgProc/basic_drawing/Drawing_1.cpp draw_atom
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgProc/BasicGeometricDrawing/BasicGeometricDrawing.java draw_atom
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgProc/BasicGeometricDrawing/basic_geometric_drawing.py draw_atom
@end_toggle

And to draw the rook we employed **MyLine**, **rectangle** and a **MyPolygon**:
@add_toggle_cpp
@snippet cpp/tutorial_code/ImgProc/basic_drawing/Drawing_1.cpp draw_rook
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgProc/BasicGeometricDrawing/BasicGeometricDrawing.java draw_rook
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgProc/BasicGeometricDrawing/basic_geometric_drawing.py draw_rook
@end_toggle


Let's check what is inside each of these functions:
@add_toggle_cpp
@end_toggle

<H4>MyLine</H4>
@add_toggle_cpp
@snippet cpp/tutorial_code/ImgProc/basic_drawing/Drawing_1.cpp my_line
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgProc/BasicGeometricDrawing/BasicGeometricDrawing.java my_line
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgProc/BasicGeometricDrawing/basic_geometric_drawing.py my_line
@end_toggle

-   As we can see, **MyLine** just call the function **line()** , which does the following:
    -   Draw a line from Point **start** to Point **end**
    -   The line is displayed in the image **img**
    -   The line color is defined by <B>( 0, 0, 0 )</B> which is the RGB value correspondent
        to **Black**
    -   The line thickness is set to **thickness** (in this case 2)
    -   The line is a 8-connected one (**lineType** = 8)

<H4>MyEllipse</H4>
@add_toggle_cpp
@snippet cpp/tutorial_code/ImgProc/basic_drawing/Drawing_1.cpp my_ellipse
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgProc/BasicGeometricDrawing/BasicGeometricDrawing.java my_ellipse
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgProc/BasicGeometricDrawing/basic_geometric_drawing.py my_ellipse
@end_toggle

-   From the code above, we can observe that the function **ellipse()** draws an ellipse such
    that:

    -   The ellipse is displayed in the image **img**
    -   The ellipse center is located in the point <B>(w/2, w/2)</B> and is enclosed in a box
        of size <B>(w/4, w/16)</B>
    -   The ellipse is rotated **angle** degrees
    -   The ellipse extends an arc between **0** and **360** degrees
    -   The color of the figure will be <B>( 255, 0, 0 )</B> which means blue in BGR value.
    -   The ellipse's **thickness** is 2.

<H4>MyFilledCircle</H4>
@add_toggle_cpp
@snippet cpp/tutorial_code/ImgProc/basic_drawing/Drawing_1.cpp my_filled_circle
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgProc/BasicGeometricDrawing/BasicGeometricDrawing.java my_filled_circle
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgProc/BasicGeometricDrawing/basic_geometric_drawing.py my_filled_circle
@end_toggle

-   Similar to the ellipse function, we can observe that *circle* receives as arguments:

    -   The image where the circle will be displayed (**img**)
    -   The center of the circle denoted as the point **center**
    -   The radius of the circle: **w/32**
    -   The color of the circle: <B>( 0, 0, 255 )</B> which means *Red* in BGR
    -   Since **thickness** = -1, the circle will be drawn filled.

<H4>MyPolygon</H4>
@add_toggle_cpp
@snippet cpp/tutorial_code/ImgProc/basic_drawing/Drawing_1.cpp my_polygon
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgProc/BasicGeometricDrawing/BasicGeometricDrawing.java my_polygon
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgProc/BasicGeometricDrawing/basic_geometric_drawing.py my_polygon
@end_toggle

-   To draw a filled polygon we use the function **fillPoly()** . We note that:

    -   The polygon will be drawn on **img**
    -   The vertices of the polygon are the set of points in **ppt**
    -   The color of the polygon is defined by <B>( 255, 255, 255 )</B>, which is the BGR
        value for *white*

<H4>rectangle</H4>
@add_toggle_cpp
@snippet cpp/tutorial_code/ImgProc/basic_drawing/Drawing_1.cpp rectangle
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgProc/BasicGeometricDrawing/BasicGeometricDrawing.java rectangle
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgProc/BasicGeometricDrawing/basic_geometric_drawing.py rectangle
@end_toggle

-   Finally we have the @ref cv::rectangle function (we did not create a special function for
    this guy). We note that:

    -   The rectangle will be drawn on **rook_image**
    -   Two opposite vertices of the rectangle are defined by <B>( 0, 7*w/8 )</B>
        and <B>( w, w )</B>
    -   The color of the rectangle is given by <B>( 0, 255, 255 )</B> which is the BGR value
        for *yellow*
    -   Since the thickness value is given by **FILLED (-1)**, the rectangle will be filled.

Result
------

Compiling and running your program should give you a result like this:

![](images/Drawing_1_Tutorial_Result_0.png)

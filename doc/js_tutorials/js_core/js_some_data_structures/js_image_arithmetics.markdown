Some Data Structures {#tutorial_js_some_data_structures}
===============================

Goal
----

-   You will learn some data structures : **Point**, **Scalar**, **Size**,  **Circle**, **Rect**, **RotatedRect**  etc.

Scalar is array type in javascript. Point, Size, Circle, Rect and RotatedRect are object type in javascript.

Point
--------------
@code{.js}
let point = new cv.Point(x, y);
@endcode

@param x      x coordinate of the point.(the origin is the top left corner of the image)
@param y      y coordinate of the point.

Scalar
--------------
@code{.js}
let scalar = new cv.Scalar(R, G, B, Alpha);
@endcode

@param R     pixel value of red channel.
@param G     pixel value of green channel.
@param B     pixel value of blue channel.
@param Alpha pixel value of alpha channel.

Size
------------------
@code{.js}
let size = new cv.Size(width, height);
@endcode

@param width    the width of the size.
@param height   the height of the size.

Circle
------------------
@code{.js}
let circle = new cv.Circle(center, radius);
@endcode

@param center    the center of the circle.
@param radius    the radius of the circle.

Rect
------------------
@code{.js}
let rect = new cv.Rect(x, y, width, height);
@endcode

@param x        x coordinate of the vertex which is the top left corner of the rectangle.
@param y        y coordinate of the vertex which is the top left corner of the rectangle.
@param width    the width of the rectangle.
@param height   the height of the rectangle.

RotatedRect
------------------
@code{.js}
let rotatedRect = new cv.RotatedRect(center, size, angle);
@endcode

@param center  the rectangle mass center.
@param size    width and height of the rectangle.
@param angle   the rotation angle in a clockwise direction. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle.

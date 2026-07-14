Some Data Structures {#tutorial_js_some_data_structures}
===============================

Goal
----

-   You will learn some data structures : **Point**, **Scalar**, **Size**,  **Circle**, **Rect**, **RotatedRect**  etc.

Scalar is array type in Javascript. Point, Size, Circle, Rect and RotatedRect are object type in JavaScript.

Point
--------------

There are 2 ways to construct a Point and they are the same:
@code{.js}
// The first way
let point = new cv.Point(x, y);
// The second way
let point = {x: x, y: y};
@endcode

@param x      x coordinate of the point.(the origin is the top left corner of the image)
@param y      y coordinate of the point.

Scalar
--------------

There are 2 ways to construct a Scalar and they are the same:
@code{.js}
// The first way
let scalar = new cv.Scalar(R, G, B, Alpha);
// The second way
let scalar = [R, G, B, Alpha];
@endcode

@param R     pixel value of red channel.
@param G     pixel value of green channel.
@param B     pixel value of blue channel.
@param Alpha pixel value of alpha channel.

Size
------------------

There are 2 ways to construct a Size and they are the same:
@code{.js}
// The first way
let size = new cv.Size(width, height);
// The second way
let size = {width : width, height : height};
@endcode

@param width    the width of the size.
@param height   the height of the size.

Circle
------------------

There are 2 ways to construct a Circle and they are the same:
@code{.js}
// The first way
let circle = new cv.Circle(center, radius);
// The second way
let circle = {center : center, radius : radius};
@endcode

@param center    the center of the circle.
@param radius    the radius of the circle.

Rect
------------------

There are 2 ways to construct a Rect and they are the same:
@code{.js}
// The first way
let rect = new cv.Rect(x, y, width, height);
// The second way
let rect = {x : x, y : y, width : width, height : height};
@endcode

@param x        x coordinate of the vertex which is the top left corner of the rectangle.
@param y        y coordinate of the vertex which is the top left corner of the rectangle.
@param width    the width of the rectangle.
@param height   the height of the rectangle.

RotatedRect
------------------

There are 2 ways to construct a RotatedRect and they are the same:
@code{.js}
// The first way
let rotatedRect = new cv.RotatedRect(center, size, angle);
// The second way
let rotatedRect = {center : center, size : size, angle : angle};
@endcode

@param center  the rectangle mass center.
@param size    width and height of the rectangle.
@param angle   the rotation angle in a clockwise direction. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle.

Learn how to get the vertices from rotatedRect:

We use the function: **cv.RotatedRect.points(rotatedRect)**
@param rotatedRect       rotated rectangle

@code{.js}
let vertices = cv.RotatedRect.points(rotatedRect);
let point1 = vertices[0];
let point2 = vertices[1];
let point3 = vertices[2];
let point4 = vertices[3];
@endcode

Learn how to get the bounding rectangle from rotatedRect:

We use the function: **cv.RotatedRect.boundingRect(rotatedRect)**
@param rotatedRect       rotated rectangle

@code{.js}
let boundingRect = cv.RotatedRect.boundingRect(rotatedRect);
@endcode
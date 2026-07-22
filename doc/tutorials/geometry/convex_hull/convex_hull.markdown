Convex Hull {#tutorial_geometry_convex_hull}
===========

@tableofcontents

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 5.0 |

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function @ref cv::convexHull (part of the **geometry** module in OpenCV 5.0)
    to find the convex hull of a 2D point set or contour.
-   Link the new `opencv_geometry` module in your CMake project.

Theory
------

The convex hull of a set of points is the smallest convex polygon that contains all the
points. To visualize this, imagine a rubber band stretched open to encompass all the given
points; when released, it snaps around the outermost points, taking the shape of the convex
hull.

### The algorithm

OpenCV computes the hull using Sklansky's algorithm, which is highly efficient for 2D points:

-   If the points are unsorted, the time complexity is \f$O(N \log N)\f$.
-   If the points are already sorted (for example, ordered contour points from
    @ref cv::findContours), the complexity drops to \f$O(N)\f$.

@note In OpenCV 5.0 the computational-geometry algorithms were reorganized: functions such as
@ref cv::convexHull moved from `imgproc` to the new **geometry** module. C++ code must now
include `<opencv2/geometry.hpp>`.

Project configuration (CMake)
-----------------------------

Because OpenCV 5.0 moved these functions into a dedicated module, link `opencv_geometry` in
your `CMakeLists.txt` alongside the standard modules:

@code{.cmake}
cmake_minimum_required(VERSION 3.1)
project(ConvexHullDemo)

find_package(OpenCV REQUIRED)

add_executable(ConvexHullDemo convex_hull_demo.cpp)
target_link_libraries(ConvexHullDemo ${OpenCV_LIBS} opencv_geometry)
@endcode

Code
----

@add_toggle_cpp
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/5.x/samples/cpp/tutorial_code/ShapeDescriptors/hull_demo.cpp)
@include samples/cpp/tutorial_code/ShapeDescriptors/hull_demo.cpp
@end_toggle

@add_toggle_java
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/5.x/samples/java/tutorial_code/ShapeDescriptors/hull/HullDemo.java)
@include samples/java/tutorial_code/ShapeDescriptors/hull/HullDemo.java
@end_toggle

@add_toggle_python
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/5.x/samples/python/tutorial_code/ShapeDescriptors/hull/hull_demo.py)
@include samples/python/tutorial_code/ShapeDescriptors/hull/hull_demo.py
@end_toggle

Explanation
-----------

-   **Include the headers.** The demo includes `imgproc.hpp` for edge and contour detection
    and the new `geometry.hpp` for the convex-hull algorithm.
-   **Edge detection and contour extraction.** Inside the trackbar callback the Canny edge
    detector finds the raw edges, and @ref cv::findContours retrieves the boundary shapes.
-   **Compute the convex hull.** For each contour, @ref cv::convexHull computes its convex
    polygon. By default it takes the input point set and outputs the hull coordinates.
-   **Draw the results.** Both the original contours and their bounding hulls are drawn with
    @ref cv::drawContours on a blank image, each contour/hull pair sharing a random color.

@note `cv::convexHull(points, hull, clockwise, returnPoints)` accepts two useful flags.
`clockwise` orders the output points clockwise when `true` (default `false`). `returnPoints`
returns hull coordinates when `true` (default); set it to `false` to get the **indices** of
the original points instead — required if you later call @ref cv::convexityDefects.

Result
------

After compiling and running with an input image, a window with a trackbar appears. As you
adjust the Canny threshold, the contours are recomputed dynamically and drawn alongside their
convex hulls, each matching pair sharing the same randomized color.

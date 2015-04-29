Creating Widgets {#tutorial_creating_widgets}
================

Goal
----

In this tutorial you will learn how to

-   Create your own widgets using WidgetAccessor and VTK.
-   Show your widget in the visualization window.

Code
----

You can download the code from [here ](https://github.com/Itseez/opencv/tree/master/samples/cpp/tutorial_code/viz/creating_widgets.cpp).
@include samples/cpp/tutorial_code/viz/creating_widgets.cpp

Explanation
-----------

Here is the general structure of the program:

-   Extend Widget3D class to create a new 3D widget.
    @code{.cpp}
    class WTriangle : public viz::Widget3D
    {
        public:
            WTriangle(const Point3f &pt1, const Point3f &pt2, const Point3f &pt3, const viz::Color & color = viz::Color::white());
    };
    @endcode
-   Assign a VTK actor to the widget.
    @code{.cpp}
    // Store this actor in the widget in order that visualizer can access it
    viz::WidgetAccessor::setProp(*this, actor);
    @endcode
-   Set color of the widget.
    @code{.cpp}
    // Set the color of the widget. This has to be called after WidgetAccessor.
    setColor(color);
    @endcode
-   Construct a triangle widget and display it in the window.
    @code{.cpp}
    /// Create a triangle widget
    WTriangle tw(Point3f(0.0,0.0,0.0), Point3f(1.0,1.0,1.0), Point3f(0.0,1.0,0.0), viz::Color::red());

    /// Show widget in the visualizer window
    myWindow.showWidget("TRIANGLE", tw);
    @endcode

Results
-------

Here is the result of the program.

![](images/red_triangle.png)

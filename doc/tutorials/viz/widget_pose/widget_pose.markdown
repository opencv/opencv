Pose of a widget {#tutorial_widget_pose}
================

Goal
----

In this tutorial you will learn how to

-   Add widgets to the visualization window
-   Use Affine3 to set pose of a widget
-   Rotating and translating a widget along an axis

Code
----

You can download the code from [here ](https://github.com/opencv/opencv/tree/3.4/samples/cpp/tutorial_code/viz/widget_pose.cpp).
@include samples/cpp/tutorial_code/viz/widget_pose.cpp

Explanation
-----------

Here is the general structure of the program:

-   Create a visualization window.
    @code{.cpp}
    /// Create a window
    viz::Viz3d myWindow("Coordinate Frame");
    @endcode
-   Show coordinate axes in the window using CoordinateSystemWidget.
    @code{.cpp}
    /// Add coordinate axes
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
    @endcode
-   Display a line representing the axis (1,1,1).
    @code{.cpp}
    /// Add line to represent (1,1,1) axis
    viz::WLine axis(Point3f(-1.0f,-1.0f,-1.0f), Point3f(1.0f,1.0f,1.0f));
    axis.setRenderingProperty(viz::LINE_WIDTH, 4.0);
    myWindow.showWidget("Line Widget", axis);
    @endcode
-   Construct a cube.
    @code{.cpp}
    /// Construct a cube widget
    viz::WCube cube_widget(Point3f(0.5,0.5,0.0), Point3f(0.0,0.0,-0.5), true, viz::Color::blue());
    cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);
    myWindow.showWidget("Cube Widget", cube_widget);
    @endcode
-   Create rotation matrix from rodrigues vector
    @code{.cpp}
    /// Rotate around (1,1,1)
    rot_vec.at<float>(0,0) += CV_PI * 0.01f;
    rot_vec.at<float>(0,1) += CV_PI * 0.01f;
    rot_vec.at<float>(0,2) += CV_PI * 0.01f;

    ...

    Mat rot_mat;
    Rodrigues(rot_vec, rot_mat);
    @endcode
-   Use Affine3f to set pose of the cube.
    @code{.cpp}
    /// Construct pose
    Affine3f pose(rot_mat, Vec3f(translation, translation, translation));
    myWindow.setWidgetPose("Cube Widget", pose);
    @endcode
-   Animate the rotation using wasStopped and spinOnce
    @code{.cpp}
    while(!myWindow.wasStopped())
    {
        ...

        myWindow.spinOnce(1, true);
    }
    @endcode

Results
-------

Here is the result of the program.

\htmlonly
<div align="center">
<iframe width="420" height="315" src="https://www.youtube.com/embed/22HKMN657U0" frameborder="0" allowfullscreen></iframe>
</div>
\endhtmlonly

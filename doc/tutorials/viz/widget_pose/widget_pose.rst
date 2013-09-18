.. _widget_pose:

Pose of a widget
****************

Goal
====

In this tutorial you will learn how to

.. container:: enumeratevisibleitemswithsquare

  * Add widgets to the visualization window
  * Use Affine3 to set pose of a widget
  * Rotating and translating a widget along an axis

Code
====

You can download the code from :download:`here <../../../../samples/cpp/tutorial_code/viz/widget_pose.cpp>`.

.. code-block:: cpp

    #include <opencv2/viz.hpp>
    #include <opencv2/calib3d.hpp>
    #include <iostream>

    using namespace cv;
    using namespace std;

    /**
     * @function main
     */
    int main()
    {
        /// Create a window
        viz::Viz3d myWindow("Coordinate Frame");

        /// Add coordinate axes
        myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

        /// Add line to represent (1,1,1) axis
        viz::WLine axis(Point3f(-1.0f,-1.0f,-1.0f), Point3f(1.0f,1.0f,1.0f));
        axis.setRenderingProperty(viz::LINE_WIDTH, 4.0);
        myWindow.showWidget("Line Widget", axis);

        /// Construct a cube widget
        viz::WCube cube_widget(Point3f(0.5,0.5,0.0), Point3f(0.0,0.0,-0.5), true, viz::Color::blue());
        cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);

        /// Display widget (update if already displayed)
        myWindow.showWidget("Cube Widget", cube_widget);

        /// Rodrigues vector
        Mat rot_vec = Mat::zeros(1,3,CV_32F);
        float translation_phase = 0.0, translation = 0.0;
        while(!myWindow.wasStopped())
        {
            /* Rotation using rodrigues */
            /// Rotate around (1,1,1)
            rot_vec.at<float>(0,0) += CV_PI * 0.01f;
            rot_vec.at<float>(0,1) += CV_PI * 0.01f;
            rot_vec.at<float>(0,2) += CV_PI * 0.01f;

            /// Shift on (1,1,1)
            translation_phase += CV_PI * 0.01f;
            translation = sin(translation_phase);

            Mat rot_mat;
            Rodrigues(rot_vec, rot_mat);

            /// Construct pose
            Affine3f pose(rot_mat, Vec3f(translation, translation, translation));

            myWindow.setWidgetPose("Cube Widget", pose);

            myWindow.spinOnce(1, true);
        }

        return 0;
    }

Explanation
===========

Here is the general structure of the program:

* Create a visualization window.

.. code-block:: cpp

    /// Create a window
    viz::Viz3d myWindow("Coordinate Frame");

* Show coordinate axes in the window using CoordinateSystemWidget.

.. code-block:: cpp

    /// Add coordinate axes
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

* Display a line representing the axis (1,1,1).

.. code-block:: cpp

    /// Add line to represent (1,1,1) axis
    viz::WLine axis(Point3f(-1.0f,-1.0f,-1.0f), Point3f(1.0f,1.0f,1.0f));
    axis.setRenderingProperty(viz::LINE_WIDTH, 4.0);
    myWindow.showWidget("Line Widget", axis);

* Construct a cube.

.. code-block:: cpp

    /// Construct a cube widget
    viz::WCube cube_widget(Point3f(0.5,0.5,0.0), Point3f(0.0,0.0,-0.5), true, viz::Color::blue());
    cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);
    myWindow.showWidget("Cube Widget", cube_widget);

* Create rotation matrix from rodrigues vector

.. code-block:: cpp

    /// Rotate around (1,1,1)
    rot_vec.at<float>(0,0) += CV_PI * 0.01f;
    rot_vec.at<float>(0,1) += CV_PI * 0.01f;
    rot_vec.at<float>(0,2) += CV_PI * 0.01f;

    ...

    Mat rot_mat;
    Rodrigues(rot_vec, rot_mat);

* Use Affine3f to set pose of the cube.

.. code-block:: cpp

    /// Construct pose
    Affine3f pose(rot_mat, Vec3f(translation, translation, translation));
    myWindow.setWidgetPose("Cube Widget", pose);

* Animate the rotation using wasStopped and spinOnce

.. code-block:: cpp

    while(!myWindow.wasStopped())
    {
        ...

        myWindow.spinOnce(1, true);
    }

Results
=======

Here is the result of the program.

.. raw:: html

  <div align="center">
  <iframe width="420" height="315" src="https://www.youtube.com/embed/Jo47zc6-hvI" frameborder="0" allowfullscreen></iframe>
  </div>

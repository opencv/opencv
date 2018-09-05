Transformations {#tutorial_transformations}
===============

Goal
----

In this tutorial you will learn how to

-   How to use makeTransformToGlobal to compute pose
-   How to use makeCameraPose and Viz3d::setViewerPose
-   How to visualize camera position by axes and by viewing frustum

Code
----

You can download the code from [here ](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/viz/transformations.cpp).
@include samples/cpp/tutorial_code/viz/transformations.cpp

Explanation
-----------

Here is the general structure of the program:

-   Create a visualization window.
    @code{.cpp}
    /// Create a window
    viz::Viz3d myWindow("Transformations");
    @endcode
-   Get camera pose from camera position, camera focal point and y direction.
    @code{.cpp}
    /// Let's assume camera has the following properties
    Point3f cam_pos(3.0f,3.0f,3.0f), cam_focal_point(3.0f,3.0f,2.0f), cam_y_dir(-1.0f,0.0f,0.0f);

    /// We can get the pose of the cam using makeCameraPose
    Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    @endcode
-   Obtain transform matrix knowing the axes of camera coordinate system.
    @code{.cpp}
    /// We can get the transformation matrix from camera coordinate system to global using
    /// - makeTransformToGlobal. We need the axes of the camera
    Affine3f transform = viz::makeTransformToGlobal(Vec3f(0.0f,-1.0f,0.0f), Vec3f(-1.0f,0.0f,0.0f), Vec3f(0.0f,0.0f,-1.0f), cam_pos);
    @endcode
-   Create a cloud widget from bunny.ply file
    @code{.cpp}
    /// Create a cloud widget.
    Mat bunny_cloud = cvcloud_load();
    viz::WCloud cloud_widget(bunny_cloud, viz::Color::green());
    @endcode
-   Given the pose in camera coordinate system, estimate the global pose.
    @code{.cpp}
    /// Pose of the widget in camera frame
    Affine3f cloud_pose = Affine3f().translate(Vec3f(0.0f,0.0f,3.0f));
    /// Pose of the widget in global frame
    Affine3f cloud_pose_global = transform * cloud_pose;
    @endcode
-   If the view point is set to be global, visualize camera coordinate frame and viewing frustum.
    @code{.cpp}
    /// Visualize camera frame
    if (!camera_pov)
    {
        viz::WCameraPosition cpw(0.5); // Coordinate axes
        viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599)); // Camera frustum
        myWindow.showWidget("CPW", cpw, cam_pose);
        myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
    }
    @endcode
-   Visualize the cloud widget with the estimated global pose
    @code{.cpp}
    /// Visualize widget
    myWindow.showWidget("bunny", cloud_widget, cloud_pose_global);
    @endcode
-   If the view point is set to be camera's, set viewer pose to **cam_pose**.
    @code{.cpp}
    /// Set the viewer pose to that of camera
    if (camera_pov)
        myWindow.setViewerPose(cam_pose);
    @endcode

Results
-------

-#  Here is the result from the camera point of view.

    ![](images/camera_view_point.png)

-#  Here is the result from global point of view.

    ![](images/global_view_point.png)

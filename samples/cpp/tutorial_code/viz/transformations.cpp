/**
 * @file transformations.cpp
 * @brief Visualizing cloud in different positions, coordinate frames, camera frustums
 * @author Ozan Cagri Tonkal
 */

#include <opencv2/viz.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/**
 * @function help
 * @brief Display instructions to use this tutorial program
 */
void help()
{
    cout
    << "--------------------------------------------------------------------------"   << endl
    << "This program shows how to use makeTransformToGlobal() to compute required pose,"
    << "how to use makeCameraPose and Viz3d::setViewerPose. You can observe the scene "
    << "from camera point of view (C) or global point of view (G)"                    << endl
    << "Usage:"                                                                       << endl
    << "./coordinate_frame [ G | C ]"                                                 << endl
    << endl;
}

/**
 * @function main
 */
int main(int argn, char **argv)
{
    help();
    
    if (argn < 2)
    {
        cout << "Missing arguments." << endl;
        return 1;
    }
    
    bool camera_pov = (argv[1][0] == 'C');
    
    /// Create a window
    viz::Viz3d myWindow("Coordinate Frame");
    
    /// Add coordinate axes
    myWindow.showWidget("Coordinate Widget", viz::CoordinateSystemWidget());
    
    /// Let's assume camera has the following properties
    Point3f cam_pos(3.0f,3.0f,3.0f), cam_focal_point(3.0f, 3.0f, 2.0f), cam_y_dir(-1.0,0.0,0.0);
    /// We can get the pose of the cam using makeCameraPose
    Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    /// We can get the transformation matrix from camera coordinate system to global using
    /// - makeTransformToGlobal. We need the axes of the camera
    Affine3f transform = viz::makeTransformToGlobal(Vec3f(0.0f,-1.0f,0.0f), Vec3f(-1.0f,0.0f,0.0f), Vec3f(0.0f,0.0f,-1.0f), cam_pos);
    
    /// Create a cloud widget.
    viz::SphereWidget sphere_widget(Point3f(0.0,0.0,0.0), 0.5, 10, viz::Color::red());
    
    /// Pose of the widget in camera frame
    Affine3f sphere_pose = Affine3f().translate(Vec3f(0.0f,0.0f,3.0f));
    /// Pose of the widget in global frame
    Affine3f sphere_pose_global = transform * sphere_pose;
    
    /// Visualize camera frame
    if (!camera_pov)
    {
        viz::CameraPositionWidget cpw(0.5); // Coordinate axes
        viz::CameraPositionWidget cpw_frustum(Vec2f(0.889484, 0.523599)); // Camera frustum
        myWindow.showWidget("CPW", cpw, cam_pose);
        myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
    }
    
    /// Visualize widget
    myWindow.showWidget("sphere", sphere_widget, sphere_pose_global);
    
    /// Set the viewer pose to that of camera
    if (camera_pov)
        myWindow.setViewerPose(cam_pose);
    
    /// Start event loop.
    myWindow.spin();
    
    return 0;
}

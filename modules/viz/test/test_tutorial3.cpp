#include "test_precomp.hpp"

using namespace cv;
using namespace std;

/**
 * @function main
 */
void tutorial3(bool camera_pov)
{
    /// Create a window
    viz::Viz3d myWindow("Coordinate Frame");

    /// Add coordinate axes
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

    /// Let's assume camera has the following properties
    Point3d cam_origin(3.0, 3.0, 3.0), cam_focal_point(3.0, 3.0, 2.0), cam_y_dir(-1.0, 0.0, 0.0);

    /// We can get the pose of the cam using makeCameraPose
    Affine3d camera_pose = viz::makeCameraPose(cam_origin, cam_focal_point, cam_y_dir);

    /// We can get the transformation matrix from camera coordinate system to global using
    /// - makeTransformToGlobal. We need the axes of the camera
    Affine3d transform = viz::makeTransformToGlobal(Vec3d(0.0, -1.0, 0.0), Vec3d(-1.0, 0.0, 0.0), Vec3d(0.0, 0.0, -1.0), cam_origin);

    /// Create a cloud widget.
    Mat dragon_cloud = viz::readCloud(get_dragon_ply_file_path());
    viz::WCloud cloud_widget(dragon_cloud, viz::Color::green());

    /// Pose of the widget in camera frame
    Affine3d cloud_pose = Affine3d().rotate(Vec3d(0.0, CV_PI/2, 0.0)).rotate(Vec3d(0.0, 0.0, CV_PI)).translate(Vec3d(0.0, 0.0, 3.0));
    /// Pose of the widget in global frame
    Affine3d cloud_pose_global = transform * cloud_pose;

    /// Visualize camera frame
    myWindow.showWidget("CPW_FRUSTUM", viz::WCameraPosition(Vec2f(0.889484f, 0.523599f)), camera_pose);
    if (!camera_pov)
        myWindow.showWidget("CPW", viz::WCameraPosition(0.5), camera_pose);

    /// Visualize widget
    myWindow.showWidget("bunny", cloud_widget, cloud_pose_global);

    /// Set the viewer pose to that of camera
    if (camera_pov)
        myWindow.setViewerPose(camera_pose);

    /// Start event loop.
    myWindow.spin();
}

TEST(Viz, tutorial3_global_view)
{
    tutorial3(false);
}

TEST(Viz, tutorial3_camera_view)
{
    tutorial3(true);
}

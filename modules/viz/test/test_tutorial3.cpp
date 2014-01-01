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
    Point3f cam_pos(3.f, 3.f, 3.f), cam_focal_point(3.f, 3.f, 2.f), cam_y_dir(-1.f, 0.f, 0.f);

    /// We can get the pose of the cam using makeCameraPose
    Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);

    /// We can get the transformation matrix from camera coordinate system to global using
    /// - makeTransformToGlobal. We need the axes of the camera
    Affine3f transform = viz::makeTransformToGlobal(Vec3f(0.f, -1.f, 0.f), Vec3f(-1.f, 0.f, 0.f), Vec3f(0.f, 0.f, -1.f), cam_pos);

    /// Create a cloud widget.
    Mat dragon_cloud = viz::readCloud(get_dragon_ply_file_path());
    viz::WCloud cloud_widget(dragon_cloud, viz::Color::green());

    /// Pose of the widget in camera frame
    Affine3f cloud_pose = Affine3f().translate(Vec3f(0.f, 0.f, 3.f));
    /// Pose of the widget in global frame
    Affine3f cloud_pose_global = transform * cloud_pose;

    /// Visualize camera frame
    if (!camera_pov)
    {
        viz::WCameraPosition cpw(0.5); // Coordinate axes
        viz::WCameraPosition cpw_frustum(Vec2f(0.889484f, 0.523599f)); // Camera frustum
        myWindow.showWidget("CPW", cpw, cam_pose);
        myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);
    }

    /// Visualize widget
    myWindow.showWidget("bunny", cloud_widget, cloud_pose_global);

    /// Set the viewer pose to that of camera
    if (camera_pov)
        myWindow.setViewerPose(cam_pose);

    /// Start event loop.
    myWindow.spin();
}

TEST(Viz, DISABLED_tutorial3_global_view)
{
    tutorial3(false);
}

TEST(Viz, DISABLED_tutorial3_camera_view)
{
    tutorial3(true);
}

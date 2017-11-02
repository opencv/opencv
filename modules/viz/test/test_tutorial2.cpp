#include "test_precomp.hpp"

using namespace cv;
using namespace std;

static void tutorial2()
{
    /// Create a window
    viz::Viz3d myWindow("Coordinate Frame");

    /// Add coordinate axes
    myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

    /// Add line to represent (1,1,1) axis
    viz::WLine axis(Point3f(-1.0, -1.0, -1.0), Point3d(1.0, 1.0, 1.0));
    axis.setRenderingProperty(viz::LINE_WIDTH, 4.0);
    myWindow.showWidget("Line Widget", axis);

    /// Construct a cube widget
    viz::WCube cube_widget(Point3d(0.5, 0.5, 0.0), Point3d(0.0, 0.0, -0.5), true, viz::Color::blue());
    cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);

    /// Display widget (update if already displayed)
    myWindow.showWidget("Cube Widget", cube_widget);

    /// Rodrigues vector
    Vec3d rot_vec = Vec3d::all(0);
    double translation_phase = 0.0, translation = 0.0;
    while(!myWindow.wasStopped())
    {
        /* Rotation using rodrigues */
        /// Rotate around (1,1,1)
        rot_vec[0] += CV_PI * 0.01;
        rot_vec[1] += CV_PI * 0.01;
        rot_vec[2] += CV_PI * 0.01;

        /// Shift on (1,1,1)
        translation_phase += CV_PI * 0.01;
        translation = sin(translation_phase);

        /// Construct pose
        Affine3d pose(rot_vec, Vec3d(translation, translation, translation));

        myWindow.setWidgetPose("Cube Widget", pose);

        myWindow.spinOnce(1, true);
    }
}


TEST(Viz, DISABLED_tutorial2_pose_of_widget)
{
    tutorial2();
}

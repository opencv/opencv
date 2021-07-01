// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

Mat loadPoints(const String& path)
{
    Mat points;
    std::ifstream ifs(path);

    int _;
    float x, y, z, r, g, b;
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    int count;

    std::string str;
    std::getline(ifs, str);

    for (count = 0; ifs >> _ && ifs >> x && ifs >> z && ifs >> y && ifs >> r && ifs >> g && ifs >> b; ++count)
    {
        y = -y;
        float data[] = { x, y, z, r / 255.0f, g / 255.0f, b / 255.0f };
        points.push_back(Mat(Size(6, 1), CV_32F, &data));
        cx += x;
        cy += y;
        cz += z;
    }

    cx /= count;
    cy /= count;
    cz /= count;

    for (int i = 0; i < count; ++i)
    {
        points.at<float>(i, 0) -= cx;
        points.at<float>(i, 1) -= cy;
        points.at<float>(i, 2) -= cz;
    }

    return points;
}

int main()
{
    float verts_data[] = {
        -0.5, -0.5, 0.0,
        -0.5, +0.5, 0.0,
        +0.5, +0.5, 0.0,
        +0.5, -0.5, 0.0,
    };

    Mat verts_mat = Mat(Size(3, 4), CV_32F, &verts_data);

    int indices_data[] = {
        0, 1, 2,
        2, 3, 0,
    };

    Mat indices_mat = Mat(Size(3, 2), CV_32S, &indices_data);

    float point_data[] = {
        -0.5, -0.5, 0.0, 1.0f, 0.0f, 0.0f,
        -0.5, +0.5, 0.0, 0.0f, 1.0f, 0.0f,
        +0.5, +0.5, 0.0, 0.0f, 0.0f, 1.0f,
        +0.5, -0.5, 0.0, 1.0f, 1.0f, 1.0f,
    };

    Mat points_mat = Mat(Size(6, 4), CV_32F, &point_data);

    float trajectory_data[] = {
         -0.0, +5.0, +0.0, -1.0f, -1.0f, -1.0f,
         -5.0, +4.0, +2.0, +0.0f, +0.0f, -1.0f,
        -10.0, +3.0, +5.0, -0.5f, +0.2f, -0.5f,
    };

    Mat trajectory_mat = Mat(Size(6, 3), CV_32F, &trajectory_data);

    // Images taken from https://rgbd-dataset.cs.washington.edu
    // Papers that need to be cited to use this data:
    // [1] N. Silberman, D. Hoiem, P. Kohli, R. Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012.
    // [2] A.Janoch, S.Karayev, Y.Jia, J.T.Barron, M.Fritz, K.Saenko, and T.Darrell.A category - level 3 - d object dataset : Putting the kinect to work.In ICCV Workshop on Consumer Depth Cameras for Computer Vision, 2011.
    // [3] J.Xiao, A.Owens, and A.Torralba.SUN3D : A database of big spaces reconstructed using SfMand object labels.In ICCV, 2013
    Mat rgbd_components[] = {
        imread(samples::findFile("rgbd-color.jpg"), IMREAD_COLOR),
        imread(samples::findFile("rgbd-depth.png"), IMREAD_ANYDEPTH | IMREAD_ANYCOLOR)
    };

    rgbd_components[0].convertTo(rgbd_components[0], CV_32F);
    rgbd_components[1].convertTo(rgbd_components[1], CV_32F);

    Mat rgbd;
    merge(rgbd_components, 2, rgbd);
    cvtColor(rgbd, rgbd, COLOR_BGRA2RGBA);

    // Point cloud data taken from https://sketchfab.com/3d-models/anthidium-forcipatum-point-cloud-3493da15a8db4f34929fc38d9d0fcb2c
    Mat bee_mat = loadPoints(samples::findFile("anthidium-forcipatum.csv"));

    // Show two instances of the same example mesh
    viz3d::showMesh("viz3d", "mesh1", verts_mat, indices_mat);
    viz3d::showMesh("viz3d", "mesh2", verts_mat, indices_mat);
    viz3d::setObjectPosition("viz3d", "mesh1", { -2.0f, 0.0f, 0.0f });
    viz3d::setObjectPosition("viz3d", "mesh2", { 2.0f, 0.0f, 0.0f });

    // Show an example point cloud
    viz3d::showPoints("viz3d", "points", points_mat);

    // Show a bee point cloud
    viz3d::showPoints("viz3d", "bee", bee_mat);
    viz3d::setObjectPosition("viz3d", "bee", { 0.0f, 0.0f, 5.0f });

    // Show RGBD image as points
    viz3d::showRGBD("rgbd", "rgbd", rgbd, {
        529.5f, 0.0f, 365.0f,
        0.0f, 529.5f, 265.0f,
        0.0f, 0.0f, 1.0f
    }, 0.1f);
    viz3d::setObjectPosition("rgbd", "rgbd", { 0.0f, 0.0f, 0.0f });
    viz3d::setGridVisible("rgbd", true);

    // Show a solid box
    viz3d::showBox("viz3d", "box1", { 0.25f, 0.25f, 0.25f }, { 0.5f, 1.0f, 0.5f });
    viz3d::setObjectPosition("viz3d", "box1", { -5.0f, 0.0f, -5.0f });

    // Show 3 wireframe boxes
    viz3d::showBox("viz3d", "box2", { 1.0f, 0.5f, 0.5f }, { 1.0f, 0.5f, 0.5f }, viz3d::RENDER_SIMPLE);
    viz3d::showBox("viz3d", "box3", { 0.5f, 1.0f, 0.5f }, { 0.5f, 1.0f, 0.5f }, viz3d::RENDER_WIREFRAME);
    viz3d::showBox("viz3d", "box4", { 0.5f, 0.5f, 1.0f }, { 0.5f, 0.5f, 1.0f}, viz3d::RENDER_SHADING);
    viz3d::setObjectPosition("viz3d", "box2", { 5.0f, 0.0f, 5.0f });
    viz3d::setObjectPosition("viz3d", "box3", { -5.0f, 0.0f, 5.0f });
    viz3d::setObjectPosition("viz3d", "box4", { -5.0f, 0.0f, 0.0f });

    // Show a solid sphere
    viz3d::showSphere("viz3d", "sphere1", 1.0f, { 0.7f, 0.9f, 0.7f }, viz3d::RENDER_SHADING);
    viz3d::setObjectPosition("viz3d", "sphere1", { 0.0f, 0.0f, -5.0f });

    // Show wireframe sphere
    viz3d::showSphere("viz3d", "sphere2", 1.0f, { 0.7f, 0.9f, 0.7f }, viz3d::RENDER_WIREFRAME);
    viz3d::setObjectPosition("viz3d", "sphere2", { 5.0f, 0.0f, 0.0f });

    // Show plane
    viz3d::showPlane("viz3d", "plane1", { 1.5f, 1.0f }, { 0.8f, 0.5f, 0.3f });
    viz3d::setObjectPosition("viz3d", "plane1", { 5.0f, 0.0f, -5.0f });

    viz3d::showCameraTrajectory("viz3d", "trajectory", trajectory_mat, 1.5f, 0.2f);

    float x = 0.0f;
    while (waitKey(16) != 27)
    {
        // Animate objects
        viz3d::setObjectRotation("viz3d", "mesh1", { x, 0.0f, 0.0f });
        viz3d::setObjectRotation("viz3d", "mesh2", { 0.0f, 0.0f, x });
        x += 0.01f;
    }

    return 0;
}

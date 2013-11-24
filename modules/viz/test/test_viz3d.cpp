/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and / or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/
#include "test_precomp.hpp"


using namespace cv;

static cv::Mat cvcloud_load()
{
    cv::Mat cloud(1, 20000, CV_32FC3);
        std::ifstream ifs("/Users/nerei/cloud_dragon.ply");

    std::string str;
    for(size_t i = 0; i < 11; ++i)
        std::getline(ifs, str);

    cv::Point3f* data = cloud.ptr<cv::Point3f>();
    for(size_t i = 0; i < 20000; ++i)
        ifs >> data[i].x >> data[i].y >> data[i].z;

    return cloud;
}

bool constant_cam = true;
cv::viz::Widget cam_1, cam_coordinates;

void keyboard_callback(const viz::KeyboardEvent & event, void * cookie)
{
    if (event.keyDown())
    {
        if (event.getKeySym() == "space")
        {
            viz::Viz3d &viz = *((viz::Viz3d *) cookie);
            constant_cam = !constant_cam;
            if (constant_cam)
            {
                viz.showWidget("cam_1", cam_1);
                viz.showWidget("cam_coordinate", cam_coordinates);
                viz.showWidget("cam_text", viz::WText("Global View", Point2i(5,5), 28));
                viz.resetCamera();
            }
            else
            {
                viz.showWidget("cam_text", viz::WText("Cam View", Point2i(5,5), 28));
                viz.removeWidget("cam_1");
                viz.removeWidget("cam_coordinate");
            }
        }
    }
}

TEST(Viz_viz3d, develop)
{
    cv::viz::Viz3d viz("abc");

    cv::viz::Mesh3d bunny_mesh = cv::viz::Mesh3d::loadMesh("bunny.ply");
    cv::viz::WMesh bunny_widget(bunny_mesh);
    bunny_widget.setColor(cv::viz::Color::cyan());

    cam_1 = cv::viz::WCameraPosition(cv::Vec2f(0.6f, 0.4f), 0.2, cv::viz::Color::green());
    cam_coordinates = cv::viz::WCameraPosition(0.2);

    viz.showWidget("bunny", bunny_widget);
    viz.showWidget("cam_1", cam_1, viz::makeCameraPose(Point3f(1.f,0.f,0.f), Point3f(0.f,0.f,0.f), Point3f(0.f,1.f,0.f)));
    viz.showWidget("cam_coordinate", cam_coordinates, viz::makeCameraPose(Point3f(1.f,0.f,0.f), Point3f(0.f,0.f,0.f), Point3f(0.f,1.f,0.f)));

    std::vector<Affine3f> cam_path;

    for (int i = 0, j = 0; i <= 360; ++i, j+=5)
    {
        cam_path.push_back(viz::makeCameraPose(Vec3d(0.5*cos(i*CV_PI/180.0), 0.5*sin(j*CV_PI/180.0), 0.5*sin(i*CV_PI/180.0)), Vec3f(0.f, 0.f, 0.f), Vec3f(0.f, 1.f, 0.f)));
    }

    int path_counter = 0;
    int cam_path_size = cam_path.size();

    // OTHER WIDGETS
    cv::Mat img = imread("opencv.png");

    int downSample = 4;

    int row_max = img.rows/downSample;
    int col_max = img.cols/downSample;

    cv::Mat *clouds = new cv::Mat[img.cols/downSample];
    cv::Mat *colors = new cv::Mat[img.cols/downSample];

    for (int col = 0; col < col_max; ++col)
    {
        clouds[col] = Mat::zeros(img.rows/downSample, 1, CV_32FC3);
        colors[col] = Mat::zeros(img.rows/downSample, 1, CV_8UC3);
        for (int row = 0; row < row_max; ++row)
        {
            clouds[col].at<Vec3f>(row) = Vec3f(downSample * float(col) / img.cols, 1.f-(downSample * float(row) / img.rows), 0.f);
            colors[col].at<Vec3b>(row) = img.at<Vec3b>(row*downSample,col*downSample);
        }
    }

    for (int col = 0; col < col_max; ++col)
    {
        std::stringstream strstrm;
        strstrm << "cloud_" << col;
        viz.showWidget(strstrm.str(), viz::WCloud(clouds[col], colors[col]));
        viz.getWidget(strstrm.str()).setRenderingProperty(viz::POINT_SIZE, 3.0);
        viz.getWidget(strstrm.str()).setRenderingProperty(viz::OPACITY, 0.45);
    }

    viz.showWidget("trajectory", viz::WTrajectory(cam_path, viz::WTrajectory::DISPLAY_PATH, viz::Color::yellow()));
    viz.showWidget("cam_text", viz::WText("Global View", Point2i(5,5), 28));
    viz.registerKeyboardCallback(keyboard_callback, (void *) &viz);

    int angle = 0;

    while(!viz.wasStopped())
    {
        if (path_counter == cam_path_size)
        {
            path_counter = 0;
        }

        if (!constant_cam)
        {
            viz.setViewerPose(cam_path[path_counter]);
        }

        if (angle == 360) angle = 0;

        cam_1.cast<viz::WCameraPosition>().setPose(cam_path[path_counter]);
        cam_coordinates.cast<viz::WCameraPosition>().setPose(cam_path[path_counter++]);

        for (int i = 0; i < col_max; ++i)
        {
            std::stringstream strstrm;
            strstrm << "cloud_" << i;
            viz.setWidgetPose(strstrm.str(), Affine3f().translate(Vec3f(-0.5f, 0.f, (float)(-0.7 + 0.2*sin((angle+i*10)*CV_PI / 180.0)))));
        }
        angle += 10;
        viz.spinOnce(42, true);
    }

    volatile void* a = (void*)&cvcloud_load; (void)a; //fixing warnings
}

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/rgbd/colored_kinfu.hpp>

#include "io_utils.hpp"

using namespace cv;
using namespace cv::kinfu;
using namespace cv::colored_kinfu;
using namespace cv::io_utils;

#ifdef HAVE_OPENCV_VIZ
#include <opencv2/viz.hpp>
#endif

#ifdef HAVE_OPENCV_VIZ
const std::string vizWindowName = "cloud";

struct PauseCallbackArgs
{
    PauseCallbackArgs(ColoredKinFu& _kf) : kf(_kf)
    { }

    ColoredKinFu& kf;
};

void pauseCallback(const viz::MouseEvent& me, void* args);
void pauseCallback(const viz::MouseEvent& me, void* args)
{
    if(me.type == viz::MouseEvent::Type::MouseMove       ||
       me.type == viz::MouseEvent::Type::MouseScrollDown ||
       me.type == viz::MouseEvent::Type::MouseScrollUp)
    {
        PauseCallbackArgs pca = *((PauseCallbackArgs*)(args));
        viz::Viz3d window(vizWindowName);
        UMat rendered;
        pca.kf.render(rendered, window.getViewerPose().matrix);
        imshow("render", rendered);
        waitKey(1);
    }
}
#endif

static const char* keys =
{
    "{help h usage ? | | print this message   }"
    "{depth  | | Path to folder with depth.txt and rgb.txt files listing a set of depth and rgb images }"
    "{camera |0| Index of depth camera to be used as a depth source }"
    "{coarse | | Run on coarse settings (fast but ugly) or on default (slow but looks better),"
        " in coarse mode points and normals are displayed }"
    "{idle   | | Do not run KinFu, just display depth frames }"
    "{record | | Write depth frames to specified file list"
        " (the same format as for the 'depth' key) }"
};

static const std::string message =
 "\nThis demo uses live depth input or RGB-D dataset taken from"
 "\nhttps://vision.in.tum.de/data/datasets/rgbd-dataset"
 "\nto demonstrate KinectFusion implementation \n";


int main(int argc, char **argv)
{
    bool coarse = false;
    bool idle = false;
    std::string recordPath;

    CommandLineParser parser(argc, argv, keys);
    parser.about(message);

    if(!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return -1;
    }

    if(parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if(parser.has("coarse"))
    {
        coarse = true;
    }
    if(parser.has("record"))
    {
        recordPath = parser.get<String>("record");
    }
    if(parser.has("idle"))
    {
        idle = true;
    }

    Ptr<DepthSource> ds;
    Ptr<RGBSource> rgbs;

    if (parser.has("depth"))
        ds = makePtr<DepthSource>(parser.get<String>("depth") + "/depth.txt");
    else
        ds = makePtr<DepthSource>(parser.get<int>("camera"));

    //TODO: intrinsics for camera
    rgbs = makePtr<RGBSource>(parser.get<String>("depth") + "/rgb.txt");

    if (ds->empty())
    {
        std::cerr << "Failed to open depth source" << std::endl;
        parser.printMessage();
        return -1;
    }

    Ptr<DepthWriter> depthWriter;
    Ptr<RGBWriter> rgbWriter;

    if (!recordPath.empty())
    {
        depthWriter = makePtr<DepthWriter>(recordPath);
        rgbWriter = makePtr<RGBWriter>(recordPath);
    }
    Ptr<colored_kinfu::Params> params;
    Ptr<ColoredKinFu> kf;

    params = colored_kinfu::Params::coloredTSDFParams(coarse);

    // These params can be different for each depth sensor
    ds->updateParams(*params);

    rgbs->updateParams(*params);

    // Enables OpenCL explicitly (by default can be switched-off)
    cv::setUseOptimized(false);

    // Scene-specific params should be tuned for each scene individually
    //float cubeSize = 1.f;
    //params->voxelSize = cubeSize/params->volumeDims[0]; //meters
    //params->tsdf_trunc_dist = 0.01f; //meters
    //params->icpDistThresh = 0.01f; //meters
    //params->volumePose = Affine3f().translate(Vec3f(-cubeSize/2.f, -cubeSize/2.f, 0.25f)); //meters
    //params->tsdf_max_weight = 16;

    if(!idle)
        kf = ColoredKinFu::create(params);

#ifdef HAVE_OPENCV_VIZ
    cv::viz::Viz3d window(vizWindowName);
    window.setViewerPose(Affine3f::Identity());
    bool pause = false;
#endif

    UMat rendered;
    UMat points;
    UMat normals;

    int64 prevTime = getTickCount();

    for(UMat frame = ds->getDepth(); !frame.empty(); frame = ds->getDepth())
    {
        if(depthWriter)
            depthWriter->append(frame);
        UMat rgb_frame = rgbs->getRGB();
#ifdef HAVE_OPENCV_VIZ
        if(pause)
        {
            // doesn't happen in idle mode
            kf->getCloud(points, normals);
            if(!points.empty() && !normals.empty())
            {
                viz::WCloud cloudWidget(points, viz::Color::white());
                viz::WCloudNormals cloudNormals(points, normals, /*level*/1, /*scale*/0.05, viz::Color::gray());
                window.showWidget("cloud", cloudWidget);
                window.showWidget("normals", cloudNormals);

                Vec3d volSize = kf->getParams().voxelSize*Vec3d(kf->getParams().volumeDims);
                window.showWidget("cube", viz::WCube(Vec3d::all(0),
                                                     volSize),
                                  kf->getParams().volumePose);
                PauseCallbackArgs pca(*kf);
                window.registerMouseCallback(pauseCallback, (void*)&pca);
                window.showWidget("text", viz::WText(cv::String("Move camera in this window. "
                                                                "Close the window or press Q to resume"), Point()));
                window.spin();
                window.removeWidget("text");
                window.removeWidget("cloud");
                window.removeWidget("normals");
                window.registerMouseCallback(0);
            }

            pause = false;
        }
        else
#endif
        {
            UMat cvt8;
            float depthFactor = params->depthFactor;
            convertScaleAbs(frame, cvt8, 0.25*256. / depthFactor);
            if(!idle)
            {
                imshow("depth", cvt8);
                imshow("rgb", rgb_frame);
                if(!kf->update(frame, rgb_frame))
                {
                    kf->reset();
                }
#ifdef HAVE_OPENCV_VIZ
                else
                {
                    if(coarse)
                    {
                        kf->getCloud(points, normals);
                        if(!points.empty() && !normals.empty())
                        {
                            viz::WCloud cloudWidget(points, viz::Color::white());
                            viz::WCloudNormals cloudNormals(points, normals, /*level*/1, /*scale*/0.05, viz::Color::gray());
                            window.showWidget("cloud", cloudWidget);
                            window.showWidget("normals", cloudNormals);
                        }
                    }

                    //window.showWidget("worldAxes", viz::WCoordinateSystem());
                    Vec3d volSize = kf->getParams().voxelSize*kf->getParams().volumeDims;
                    window.showWidget("cube", viz::WCube(Vec3d::all(0),
                                                         volSize),
                                      kf->getParams().volumePose);
                    window.setViewerPose(kf->getPose());
                    window.spinOnce(1, true);
                }
#endif

                kf->render(rendered);
            }
            else
            {
                rendered = cvt8;
            }
        }

        int64 newTime = getTickCount();
        putText(rendered, cv::format("FPS: %2d press R to reset, P to pause, Q to quit",
                                     (int)(getTickFrequency()/(newTime - prevTime))),
                Point(0, rendered.rows-1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255));
        prevTime = newTime;

        imshow("render", rendered);

        int c = waitKey(1);
        switch (c)
        {
        case 'r':
            if(!idle)
                kf->reset();
            break;
        case 'q':
            return 0;
#ifdef HAVE_OPENCV_VIZ
        case 'p':
            if(!idle)
                pause = true;
#endif
        default:
            break;
        }
    }

    return 0;
}

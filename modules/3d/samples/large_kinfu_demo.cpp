// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory

#include <fstream>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd/large_kinfu.hpp>

#include "io_utils.hpp"

using namespace cv;
using namespace cv::kinfu;
using namespace cv::large_kinfu;
using namespace cv::io_utils;

#ifdef HAVE_OPENCV_VIZ
#include <opencv2/viz.hpp>
#endif

#ifdef HAVE_OPENCV_VIZ
const std::string vizWindowName = "cloud";

struct PauseCallbackArgs
{
    PauseCallbackArgs(LargeKinfu& _largeKinfu) : largeKinfu(_largeKinfu) {}

    LargeKinfu& largeKinfu;
};

void pauseCallback(const viz::MouseEvent& me, void* args);
void pauseCallback(const viz::MouseEvent& me, void* args)
{
    if (me.type == viz::MouseEvent::Type::MouseMove ||
        me.type == viz::MouseEvent::Type::MouseScrollDown ||
        me.type == viz::MouseEvent::Type::MouseScrollUp)
    {
        PauseCallbackArgs pca = *((PauseCallbackArgs*)(args));
        viz::Viz3d window(vizWindowName);
        UMat rendered;
        pca.largeKinfu.render(rendered, window.getViewerPose().matrix);
        imshow("render", rendered);
        waitKey(1);
    }
}
#endif

static const char* keys = {
    "{help h usage ? | | print this message   }"
    "{depth  | | Path to depth.txt file listing a set of depth images }"
    "{camera |0| Index of depth camera to be used as a depth source }"
    "{coarse | | Run on coarse settings (fast but ugly) or on default (slow but looks better),"
    " in coarse mode points and normals are displayed }"
    "{idle   | | Do not run LargeKinfu, just display depth frames }"
    "{record | | Write depth frames to specified file list"
    " (the same format as for the 'depth' key) }"
};

static const std::string message =
    "\nThis demo uses live depth input or RGB-D dataset taken from"
    "\nhttps://vision.in.tum.de/data/datasets/rgbd-dataset"
    "\nto demonstrate Submap based large environment reconstruction"
    "\nThis module uses the newer hashtable based TSDFVolume (relatively fast) for larger "
    "reconstructions by default\n";

int main(int argc, char** argv)
{
    bool coarse = false;
    bool idle   = false;
    std::string recordPath;

    CommandLineParser parser(argc, argv, keys);
    parser.about(message);

    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return -1;
    }

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (parser.has("coarse"))
    {
        coarse = true;
    }
    if (parser.has("record"))
    {
        recordPath = parser.get<String>("record");
    }
    if (parser.has("idle"))
    {
        idle = true;
    }

    Ptr<DepthSource> ds;
    if (parser.has("depth"))
        ds = makePtr<DepthSource>(parser.get<String>("depth"));
    else
        ds = makePtr<DepthSource>(parser.get<int>("camera"));

    if (ds->empty())
    {
        std::cerr << "Failed to open depth source" << std::endl;
        parser.printMessage();
        return -1;
    }

    Ptr<DepthWriter> depthWriter;
    if (!recordPath.empty())
        depthWriter = makePtr<DepthWriter>(recordPath);

    Ptr<large_kinfu::Params> params;
    Ptr<LargeKinfu> largeKinfu;

    params = large_kinfu::Params::hashTSDFParams(coarse);

    // These params can be different for each depth sensor
    ds->updateParams(*params);

    // Disabled until there is no OpenCL accelerated HashTSDF is available
    cv::setUseOptimized(false);

    if (!idle)
        largeKinfu = LargeKinfu::create(params);

#ifdef HAVE_OPENCV_VIZ
    cv::viz::Viz3d window(vizWindowName);
    window.setViewerPose(Affine3f::Identity());
    bool pause = false;
#endif

    UMat rendered;
    UMat points;
    UMat normals;

    int64 prevTime = getTickCount();

    for (UMat frame = ds->getDepth(); !frame.empty(); frame = ds->getDepth())
    {
        if (depthWriter)
            depthWriter->append(frame);

#ifdef HAVE_OPENCV_VIZ
        if (pause)
        {
            // doesn't happen in idle mode
            largeKinfu->getCloud(points, normals);
            if (!points.empty() && !normals.empty())
            {
                viz::WCloud cloudWidget(points, viz::Color::white());
                viz::WCloudNormals cloudNormals(points, normals, /*level*/ 1, /*scale*/ 0.05,
                                                viz::Color::gray());
                window.showWidget("cloud", cloudWidget);
                window.showWidget("normals", cloudNormals);

                Vec3d volSize = largeKinfu->getParams().volumeParams.voxelSize *
                                Vec3d(largeKinfu->getParams().volumeParams.resolution);
                window.showWidget("cube", viz::WCube(Vec3d::all(0), volSize),
                                  largeKinfu->getParams().volumeParams.pose);
                PauseCallbackArgs pca(*largeKinfu);
                window.registerMouseCallback(pauseCallback, (void*)&pca);
                window.showWidget("text",
                                  viz::WText(cv::String("Move camera in this window. "
                                                        "Close the window or press Q to resume"),
                                             Point()));
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
            convertScaleAbs(frame, cvt8, 0.25 * 256. / depthFactor);
            if (!idle)
            {
                imshow("depth", cvt8);

                if (!largeKinfu->update(frame))
                {
                    largeKinfu->reset();
                    std::cout << "reset" << std::endl;
                }
#ifdef HAVE_OPENCV_VIZ
                else
                {
                    if (coarse)
                    {
                        largeKinfu->getCloud(points, normals);
                        if (!points.empty() && !normals.empty())
                        {
                            viz::WCloud cloudWidget(points, viz::Color::white());
                            viz::WCloudNormals cloudNormals(points, normals, /*level*/ 1,
                                                            /*scale*/ 0.05, viz::Color::gray());
                            window.showWidget("cloud", cloudWidget);
                            window.showWidget("normals", cloudNormals);
                        }
                    }

                    // window.showWidget("worldAxes", viz::WCoordinateSystem());
                    Vec3d volSize = largeKinfu->getParams().volumeParams.voxelSize *
                                    largeKinfu->getParams().volumeParams.resolution;
                    window.showWidget("cube", viz::WCube(Vec3d::all(0), volSize),
                                      largeKinfu->getParams().volumeParams.pose);
                    window.setViewerPose(largeKinfu->getPose());
                    window.spinOnce(1, true);
                }
#endif

                largeKinfu->render(rendered);
            }
            else
            {
                rendered = cvt8;
            }
        }

        int64 newTime = getTickCount();
        putText(rendered,
                cv::format("FPS: %2d press R to reset, P to pause, Q to quit",
                           (int)(getTickFrequency() / (newTime - prevTime))),
                Point(0, rendered.rows - 1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255));
        prevTime = newTime;
        imshow("render", rendered);

        int c = waitKey(1);
        switch (c)
        {
            case 'r':
                if (!idle)
                    largeKinfu->reset();
                break;
            case 'q': return 0;
#ifdef HAVE_OPENCV_VIZ
            case 'p':
                if (!idle)
                    pause = true;
#endif
            default: break;
        }
    }

    return 0;
}

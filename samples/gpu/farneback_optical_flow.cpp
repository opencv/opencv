#include <iostream>
#include <vector>
#include <sstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

template <typename T>
inline T mapVal(T x, T a, T b, T c, T d)
{
    x = ::max(::min(x, b), a);
    return c + (d-c) * (x-a) / (b-a);
}

static void colorizeFlow(const Mat &u, const Mat &v, Mat &dst)
{
    double uMin, uMax;
    minMaxLoc(u, &uMin, &uMax, 0, 0);
    double vMin, vMax;
    minMaxLoc(v, &vMin, &vMax, 0, 0);
    uMin = std::abs(uMin); uMax = std::abs(uMax);
    vMin = std::abs(vMin); vMax = std::abs(vMax);
    float dMax = static_cast<float>(::max(::max(uMin, uMax), ::max(vMin, vMax)));

    dst.create(u.size(), CV_8UC3);
    for (int y = 0; y < u.rows; ++y)
    {
        for (int x = 0; x < u.cols; ++x)
        {
            dst.at<uchar>(y,3*x) = 0;
            dst.at<uchar>(y,3*x+1) = (uchar)mapVal(-v.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
            dst.at<uchar>(y,3*x+2) = (uchar)mapVal(u.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
        }
    }
}

int main(int argc, char **argv)
{
    CommandLineParser cmd(argc, argv,
            "{ l | left | | specify left image }"
            "{ r | right | | specify right image }"
            "{ h | help | false | print help message }");

    if (cmd.get<bool>("help"))
    {
        cout << "Farneback's optical flow sample.\n\n"
             << "Usage: farneback_optical_flow_gpu [arguments]\n\n"
             << "Arguments:\n";
        cmd.printParams();
        return 0;
    }

    string pathL = cmd.get<string>("left");
    string pathR = cmd.get<string>("right");
    if (pathL.empty()) cout << "Specify left image path\n";
    if (pathR.empty()) cout << "Specify right image path\n";
    if (pathL.empty() || pathR.empty()) return -1;

    Mat frameL = imread(pathL, IMREAD_GRAYSCALE);
    Mat frameR = imread(pathR, IMREAD_GRAYSCALE);
    if (frameL.empty()) cout << "Can't open '" << pathL << "'\n";
    if (frameR.empty()) cout << "Can't open '" << pathR << "'\n";
    if (frameL.empty() || frameR.empty()) return -1;

    GpuMat d_frameL(frameL), d_frameR(frameR);
    GpuMat d_flowx, d_flowy;
    FarnebackOpticalFlow d_calc;
    Mat flowxy, flowx, flowy, image;

    bool running = true, gpuMode = true;
    int64 t, t0=0, t1=1, tc0, tc1;

    cout << "Use 'm' for CPU/GPU toggling\n";

    while (running)
    {
        t = getTickCount();

        if (gpuMode)
        {
            tc0 = getTickCount();
            d_calc(d_frameL, d_frameR, d_flowx, d_flowy);
            tc1 = getTickCount();
            d_flowx.download(flowx);
            d_flowy.download(flowy);
        }
        else
        {
            tc0 = getTickCount();
            calcOpticalFlowFarneback(
                        frameL, frameR, flowxy, d_calc.pyrScale, d_calc.numLevels, d_calc.winSize,
                        d_calc.numIters, d_calc.polyN, d_calc.polySigma, d_calc.flags);
            tc1 = getTickCount();

            Mat planes[] = {flowx, flowy};
            split(flowxy, planes);
            flowx = planes[0]; flowy = planes[1];
        }

        colorizeFlow(flowx, flowy, image);

        stringstream s;
        s << "mode: " << (gpuMode?"GPU":"CPU");
        putText(image, s.str(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255,0,255), 2);

        s.str("");
        s << "opt. flow FPS: " << cvRound((getTickFrequency()/(tc1-tc0)));
        putText(image, s.str(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255,0,255), 2);

        s.str("");
        s << "total FPS: " << cvRound((getTickFrequency()/(t1-t0)));
        putText(image, s.str(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255,0,255), 2);

        imshow("flow", image);

        char ch = (char)waitKey(3);
        if (ch == 27)
            running = false;
        else if (ch == 'm' || ch == 'M')
            gpuMode = !gpuMode;

        t0 = t;
        t1 = getTickCount();
    }

    return 0;
}

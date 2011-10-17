#include <iostream>
#include <iomanip>
#include <string>

#include "cvconfig.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

#ifdef HAVE_CUDA
#include "NPP_staging/NPP_staging.hpp"
#endif

using namespace std;
using namespace cv;
using namespace cv::gpu;

#if !defined(HAVE_CUDA)

int main(int argc, const char* argv[])
{
    cout << "Please compile the library with CUDA support" << endl;
    return -1;
}

#else

#define PARAM_INPUT     "--input"
#define PARAM_SCALE     "--scale"
#define PARAM_ALPHA     "--alpha"
#define PARAM_GAMMA     "--gamma"
#define PARAM_INNER     "--inner"
#define PARAM_OUTER     "--outer"
#define PARAM_SOLVER    "--solver"
#define PARAM_TIME_STEP "--time-step"
#define PARAM_HELP      "--help"

void printHelp()
{
    cout << "Usage help:\n";
    cout << setiosflags(ios::left);
    cout << "\t" << setw(15) << PARAM_ALPHA << " - set alpha\n";
    cout << "\t" << setw(15) << PARAM_GAMMA << " - set gamma\n";
    cout << "\t" << setw(15) << PARAM_INNER << " - set number of inner iterations\n";
    cout << "\t" << setw(15) << PARAM_INPUT << " - specify input file names (2 image files)\n";
    cout << "\t" << setw(15) << PARAM_OUTER << " - set number of outer iterations\n";
    cout << "\t" << setw(15) << PARAM_SCALE << " - set pyramid scale factor\n";
    cout << "\t" << setw(15) << PARAM_SOLVER << " - set number of basic solver iterations\n";
    cout << "\t" << setw(15) << PARAM_TIME_STEP << " - set frame interpolation time step\n";
    cout << "\t" << setw(15) << PARAM_HELP << " - display this help message\n";
}

int processCommandLine(int argc, const char* argv[], float& timeStep, string& frame0Name, string& frame1Name, BroxOpticalFlow& flow)
{
    timeStep = 0.25f;

    for (int iarg = 1; iarg < argc; ++iarg)
    {
        if (strcmp(argv[iarg], PARAM_INPUT) == 0)
        {
            if (iarg + 2 < argc)
            {
                frame0Name = argv[++iarg];
                frame1Name = argv[++iarg];
            }
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_SCALE) == 0)
        {
            if (iarg + 1 < argc)
                flow.scale_factor = static_cast<float>(atof(argv[++iarg]));
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_ALPHA) == 0)
        {
            if (iarg + 1 < argc)
                flow.alpha = static_cast<float>(atof(argv[++iarg]));
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_GAMMA) == 0)
        {
            if (iarg + 1 < argc)
                flow.gamma = static_cast<float>(atof(argv[++iarg]));
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_INNER) == 0)
        {
            if (iarg + 1 < argc)
                flow.inner_iterations = atoi(argv[++iarg]);
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_OUTER) == 0)
        {
            if (iarg + 1 < argc)
                flow.outer_iterations = atoi(argv[++iarg]);
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_SOLVER) == 0)
        {
            if (iarg + 1 < argc)
                flow.solver_iterations = atoi(argv[++iarg]);
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_TIME_STEP) == 0)
        {
            if (iarg + 1 < argc)
                timeStep = static_cast<float>(atof(argv[++iarg]));
            else
                return -1;
        }
        else if(strcmp(argv[iarg], PARAM_HELP) == 0)
        {
            printHelp();
            return 0;
        }
    }
    return 0;
}

template <typename T> inline T clamp (T x, T a, T b)
{
    return ((x) > (a) ? ((x) < (b) ? (x) : (b)) : (a));
}

template <typename T> inline T mapValue(T x, T a, T b, T c, T d)
{
    x = clamp(x, a, b);
    return c + (d - c) * (x - a) / (b - a);
}

void getFlowField(const Mat& u, const Mat& v, Mat& flowField)
{
    float maxDisplacement = 1.0f;

    for (int i = 0; i < u.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);

        for (int j = 0; j < u.cols; ++j)
        {
            float d = max(fabsf(ptr_u[j]), fabsf(ptr_v[j]));

            if (d > maxDisplacement) 
                maxDisplacement = d;
        }
    }

    flowField.create(u.size(), CV_8UC4);

    for (int i = 0; i < flowField.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);

        Vec4b* row = flowField.ptr<Vec4b>(i);

        for (int j = 0; j < flowField.cols; ++j)
        {
            row[j][0] = 0;
            row[j][1] = static_cast<unsigned char> (mapValue (-ptr_v[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][2] = static_cast<unsigned char> (mapValue ( ptr_u[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][3] = 255;
        }
    }
}

int main(int argc, const char* argv[])
{
    string frame0Name, frame1Name;
    float timeStep = 0.01f;

    BroxOpticalFlow d_flow(0.197f /*alpha*/, 50.0f /*gamma*/, 0.8f /*scale_factor*/, 
                           10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/);

    int result = processCommandLine(argc, argv, timeStep, frame0Name, frame1Name, d_flow);
    if (argc == 1 || result)
    {
        printHelp();
        return result;
    }

    if (frame0Name.empty() || frame1Name.empty())
    {
        cout << "Missing input file names\n";
        return -1;
    }

    Mat frame0Color = imread(frame0Name);
    Mat frame1Color = imread(frame1Name);

    if (frame0Color.empty() || frame1Color.empty())
    {
        cout << "Can't load input images\n";
        return -1;
    }

    cout << "OpenCV / NVIDIA Computer Vision\n";
    cout << "Optical Flow Demo: Frame Interpolation\n";
    cout << "=========================================\n";
    cout << "Press:\n ESC to quit\n 'a' to move to the previous frame\n 's' to move to the next frame\n";

    frame0Color.convertTo(frame0Color, CV_32F, 1.0 / 255.0);
    frame1Color.convertTo(frame1Color, CV_32F, 1.0 / 255.0);

    Mat frame0Gray, frame1Gray;

    cvtColor(frame0Color, frame0Gray, COLOR_BGR2GRAY);
    cvtColor(frame1Color, frame1Gray, COLOR_BGR2GRAY);

    GpuMat d_frame0(frame0Gray);
    GpuMat d_frame1(frame1Gray);
    
    Mat fu, fv;
    Mat bu, bv;

    GpuMat d_fu, d_fv;
    GpuMat d_bu, d_bv;

    cout << "Estimating optical flow\nForward...\n";

    d_flow(d_frame0, d_frame1, d_fu, d_fv);
    d_flow(d_frame1, d_frame0, d_bu, d_bv);
    
    d_fu.download(fu);
    d_fv.download(fv);
    
    d_bu.download(bu);
    d_bv.download(bv);

    // first frame color components (GPU memory)
    GpuMat d_b, d_g, d_r;

    // second frame color components (GPU memory)
    GpuMat d_bt, d_gt, d_rt;

    // prepare color components on host and copy them to device memory
    Mat channels[3];

    cv::split(frame0Color, channels);

    d_b.upload(channels[0]);
    d_g.upload(channels[1]);
    d_r.upload(channels[2]);

    cv::split(frame1Color, channels);

    d_bt.upload(channels[0]);
    d_gt.upload(channels[1]);
    d_rt.upload(channels[2]);

    cout << "Interpolating...\n";
    cout.precision (4);
    
    // temporary buffer
    GpuMat d_buf;

    // intermediate frame color components (GPU memory)
    GpuMat d_rNew, d_gNew, d_bNew;

    GpuMat d_newFrame;

    vector<Mat> frames;
    frames.reserve(1.0f / timeStep + 2);

    frames.push_back(frame0Color);

    // compute interpolated frames
    for (float timePos = timeStep; timePos < 1.0f; timePos += timeStep)
    {
        // interpolate blue channel
        interpolateFrames(d_b, d_bt, d_fu, d_fv, d_bu, d_bv, timePos, d_bNew, d_buf);
        // interpolate green channel
        interpolateFrames(d_g, d_gt, d_fu, d_fv, d_bu, d_bv, timePos, d_gNew, d_buf);
        // interpolate red channel
        interpolateFrames(d_r, d_rt, d_fu, d_fv, d_bu, d_bv, timePos, d_rNew, d_buf);

        GpuMat channels[] = {d_bNew, d_gNew, d_rNew};
        merge(channels, 3, d_newFrame);

        Mat newFrame;
        d_newFrame.download(newFrame);

        frames.push_back(newFrame);

        cout << timePos * 100.0f << "%\r";
    }
    cout << setw (5) << "100%\n";

    frames.push_back(frame1Color);

    int currentFrame;
    currentFrame = 0;

    Mat flowFieldForward;
    Mat flowFieldBackward;

    getFlowField(fu, fv, flowFieldForward);
    getFlowField(bu, bv, flowFieldBackward);

    imshow("Forward flow", flowFieldForward);
    imshow("Backward flow", flowFieldBackward);

    imshow("Interpolated frame", frames[currentFrame]);

    bool qPressed = false;
    while (!qPressed)
    {
        int key = toupper(waitKey(10));
        switch (key)
        {
        case 27:
            qPressed = true;
            break;
        case 'A':
            if (currentFrame > 0) 
                --currentFrame;

            imshow("Interpolated frame", frames[currentFrame]);
            break;
        case 'S':
            if (currentFrame < frames.size() - 1)
                ++currentFrame;

            imshow("Interpolated frame", frames[currentFrame]);
            break;
        }
    }

    return 0;
}

#endif

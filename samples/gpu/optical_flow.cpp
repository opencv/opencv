#include <iostream>
#include <iomanip>
#include <string>

#include "cvconfig.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

void getFlowField(const Mat& u, const Mat& v, Mat& flowField);

#ifdef HAVE_OPENGL

void needleMapDraw(void* userdata);

#endif

int main(int argc, const char* argv[])
{
    try
    {
        const char* keys =
           "{ h  | help      | false | print help message }"
           "{ l  | left      |       | specify left image }"
           "{ r  | right     |       | specify right image }"
           "{ s  | scale     | 0.8   | set pyramid scale factor }"
           "{ a  | alpha     | 0.197 | set alpha }"
           "{ g  | gamma     | 50.0  | set gamma }"
           "{ i  | inner     | 10    | set number of inner iterations }"
           "{ o  | outer     | 77    | set number of outer iterations }"
           "{ si | solver    | 10    | set number of basic solver iterations }"
           "{ t  | time_step | 0.1   | set frame interpolation time step }";

        CommandLineParser cmd(argc, argv, keys);

        if (cmd.get<bool>("help"))
        {
            cout << "Usage: optical_float [options]" << endl;
            cout << "Avaible options:" << endl;
            cmd.printParams();
            return 0;
        }

        string frame0Name = cmd.get<string>("left");
        string frame1Name = cmd.get<string>("right");
        float scale = cmd.get<float>("scale");
        float alpha = cmd.get<float>("alpha");
        float gamma = cmd.get<float>("gamma");
        int inner_iterations = cmd.get<int>("inner");
        int outer_iterations = cmd.get<int>("outer");
        int solver_iterations = cmd.get<int>("solver");
        float timeStep = cmd.get<float>("time_step");

        if (frame0Name.empty() || frame1Name.empty())
        {
            cerr << "Missing input file names" << endl;
            return -1;
        }

        Mat frame0Color = imread(frame0Name);
        Mat frame1Color = imread(frame1Name);

        if (frame0Color.empty() || frame1Color.empty())
        {
            cout << "Can't load input images" << endl;
            return -1;
        }

        cout << "OpenCV / NVIDIA Computer Vision" << endl;
        cout << "Optical Flow Demo: Frame Interpolation" << endl;
        cout << "=========================================" << endl;

        namedWindow("Forward flow");
        namedWindow("Backward flow");

        namedWindow("Needle Map", WINDOW_OPENGL);

        namedWindow("Interpolated frame");

        setGlDevice();

        cout << "Press:" << endl;
        cout << "\tESC to quit" << endl;
        cout << "\t'a' to move to the previous frame" << endl;
        cout << "\t's' to move to the next frame\n" << endl;

        frame0Color.convertTo(frame0Color, CV_32F, 1.0 / 255.0);
        frame1Color.convertTo(frame1Color, CV_32F, 1.0 / 255.0);

        Mat frame0Gray, frame1Gray;

        cvtColor(frame0Color, frame0Gray, COLOR_BGR2GRAY);
        cvtColor(frame1Color, frame1Gray, COLOR_BGR2GRAY);

        GpuMat d_frame0(frame0Gray);
        GpuMat d_frame1(frame1Gray);

        cout << "Estimating optical flow" << endl;

        BroxOpticalFlow d_flow(alpha, gamma, scale, inner_iterations, outer_iterations, solver_iterations);

        cout << "\tForward..." << endl;

        GpuMat d_fu, d_fv;

        d_flow(d_frame0, d_frame1, d_fu, d_fv);
                
        Mat flowFieldForward;
        getFlowField(Mat(d_fu), Mat(d_fv), flowFieldForward);
    
        cout << "\tBackward..." << endl;

        GpuMat d_bu, d_bv;

        d_flow(d_frame1, d_frame0, d_bu, d_bv);
        
        Mat flowFieldBackward;
        getFlowField(Mat(d_bu), Mat(d_bv), flowFieldBackward);

#ifdef HAVE_OPENGL
        cout << "Create Optical Flow Needle Map..." << endl;
        
        GpuMat d_vertex, d_colors;

        createOpticalFlowNeedleMap(d_bu, d_bv, d_vertex, d_colors);
#endif

        cout << "Interpolating..." << endl;

        // first frame color components
        GpuMat d_b, d_g, d_r;

        // second frame color components
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
    
        // temporary buffer
        GpuMat d_buf;

        // intermediate frame color components (GPU memory)
        GpuMat d_rNew, d_gNew, d_bNew;

        GpuMat d_newFrame;

        vector<Mat> frames;
        frames.reserve(static_cast<int>(1.0f / timeStep) + 2);

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

            frames.push_back(Mat(d_newFrame));

            cout << setprecision(4) << timePos * 100.0f << "%\r";
        }

        frames.push_back(frame1Color);

        cout << setw(5) << "100%" << endl;

        cout << "Done" << endl;

        imshow("Forward flow", flowFieldForward);
        imshow("Backward flow", flowFieldBackward);

#ifdef HAVE_OPENGL
        GlArrays arr;
        arr.setVertexArray(d_vertex);
        arr.setColorArray(d_colors, false);

        setOpenGlDrawCallback("Needle Map", needleMapDraw, &arr);
#endif

        int currentFrame = 0;

        imshow("Interpolated frame", frames[currentFrame]);

        while (true)
        {
            int key = toupper(waitKey(10));

            switch (key)
            {
            case 27:
                return 0;
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
    }
    catch (const exception& ex)
    {
        cerr << ex.what() << endl;
        return -1;
    }
    catch (...)
    {
        cerr << "Unknow error" << endl;
        return -1;
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

#ifdef HAVE_OPENGL

void needleMapDraw(void* userdata)
{
    const GlArrays* arr = static_cast<const GlArrays*>(userdata);

    GlCamera camera;
    camera.setOrthoProjection(0.0, 1.0, 1.0, 0.0, 0.0, 1.0);
    camera.lookAt(Point3d(0.0, 0.0, 1.0), Point3d(0.0, 0.0, 0.0), Point3d(0.0, 1.0, 0.0));

    camera.setupProjectionMatrix();
    camera.setupModelViewMatrix();

    render(*arr, RenderMode::TRIANGLES);
}

#endif

#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "utility_lib/utility_lib.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

enum Method
{
    BROX,
    FARNEBACK_GPU,
    FARNEBACK_CPU,
    PYR_LK
};

const char* method_str[] =
{
    "BROX CUDA",
    "FARNEBACK CUDA",
    "FARNEBACK CPU",
    "PYR_LK CUDA"
};

class App : public BaseApp
{
public:
    App();

protected:
    void process();
    bool processKey(int key);
    void printHelp();

private:
    void displayState(cv::Mat& frame, double proc_fps, double total_fps);

    Method method;
    float timeStep;

    BroxOpticalFlow brox;
    FarnebackOpticalFlow farneback;
    PyrLKOpticalFlow pyrlk;

    std::vector< cv::Ptr<PairFrameSource> > pairSources;
    size_t curSource;

    bool calcFlow;
};

App::App() :
    brox(0.197f /*alpha*/, 50.0f /*gamma*/, 0.8f /*scale*/, 10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/)
{
    pyrlk.winSize = Size(13, 13);
    pyrlk.iters = 1;

    method = BROX;

    timeStep = 0.1f;

    curSource = 0;

    calcFlow = true;
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
    const float maxDisplacement = 40.0f;

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

void App::process()
{
    if ((sources.size() > 0) && (sources.size() % 2 == 0))
    {
        for (size_t i = 0; i < sources.size(); i += 2)
            pairSources.push_back(PairFrameSource::get(sources[i], sources[i+1]));
    }
    else
    {
        cout << "Loading default frames source...\n";

        pairSources.push_back(PairFrameSource::get(new ImageSource("data/optical_flow/army1.png"),
                                                   new ImageSource("data/optical_flow/army2.png")));
        pairSources.push_back(PairFrameSource::get(new ImageSource("data/optical_flow/backyard1.png"),
                                                   new ImageSource("data/optical_flow/backyard2.png")));
        pairSources.push_back(PairFrameSource::get(new ImageSource("data/optical_flow/basketball1.png"),
                                                   new ImageSource("data/optical_flow/basketball2.png")));
        pairSources.push_back(PairFrameSource::get(new ImageSource("data/optical_flow/dumptruck1.png"),
                                                   new ImageSource("data/optical_flow/dumptruck2.png")));
        pairSources.push_back(PairFrameSource::get(new ImageSource("data/optical_flow/mequon1.png"),
                                                   new ImageSource("data/optical_flow/mequon2.png")));
        pairSources.push_back(PairFrameSource::get(new ImageSource("data/optical_flow/teddy1.png"),
                                                   new ImageSource("data/optical_flow/teddy2.png")));
    }

    Mat frame0, frame1;
    Mat frame0_32F, frame1_32F;
    Mat gray0, gray1;
    Mat gray0_32F, gray1_32F;
    GpuMat d_frame0, d_frame1;
    GpuMat d_frame0_32F, d_frame1_32F;

    Mat fu, fv;
    Mat bu, bv;
    Mat fuv, buv;
    GpuMat d_fu, d_fv;
    GpuMat d_bu, d_bv;

    Mat flowFieldForward, flowFieldBackward;

    Mat channels[3];

    GpuMat d_b0, d_g0, d_r0;
    GpuMat d_b1, d_g1, d_r1;

    GpuMat d_buf;
    GpuMat d_rNew, d_gNew, d_bNew;
    GpuMat d_newFrame;
    Mat newFrame;

    vector<Mat> frames;
    frames.reserve(static_cast<int>(1.0f / timeStep) + 2);
    int currentFrame = 0;
    bool forward = true;

    cv::Mat img_to_show;

    double proc_fps, total_fps;

    while (!exited)
    {
        if (calcFlow)
        {
            cout << "Calculate optical flow and interpolated frames" << endl;

            int64 start = getTickCount();

            pairSources[curSource]->next(frame0, frame1);

            frame0.convertTo(frame0_32F, CV_32F, 1.0 / 255.0);
            frame1.convertTo(frame1_32F, CV_32F, 1.0 / 255.0);

            switch (method)
            {
            case BROX:
                {
                    makeGray(frame0_32F, gray0_32F);
                    makeGray(frame1_32F, gray1_32F);

                    d_frame0_32F.upload(gray0_32F);
                    d_frame1_32F.upload(gray1_32F);

                    int64 proc_start = getTickCount();
                    brox(d_frame0_32F, d_frame1_32F, d_fu, d_fv);
                    brox(d_frame1_32F, d_frame0_32F, d_bu, d_bv);
                    proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

                    d_fu.download(fu);
                    d_fv.download(fv);
                    d_bu.download(bu);
                    d_bv.download(bv);

                    break;
                }

            case FARNEBACK_GPU:
                {
                    makeGray(frame0, gray0);
                    makeGray(frame1, gray1);

                    d_frame0.upload(gray0);
                    d_frame1.upload(gray1);

                    int64 proc_start = getTickCount();
                    farneback(d_frame0, d_frame1, d_fu, d_fv);
                    farneback(d_frame1, d_frame0, d_bu, d_bv);
                    proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

                    d_fu.download(fu);
                    d_fv.download(fv);
                    d_bu.download(bu);
                    d_bv.download(bv);

                    break;
                }

            case FARNEBACK_CPU:
                {
                    makeGray(frame0, gray0);
                    makeGray(frame1, gray1);

                    int64 proc_start = getTickCount();
                    calcOpticalFlowFarneback(gray0, gray1, fuv, farneback.pyrScale, farneback.numLevels, farneback.winSize, farneback.numIters, farneback.polyN, farneback.polySigma, farneback.flags);
                    calcOpticalFlowFarneback(gray1, gray0, buv, farneback.pyrScale, farneback.numLevels, farneback.winSize, farneback.numIters, farneback.polyN, farneback.polySigma, farneback.flags);
                    proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

                    cv::Mat uv_planes[2];
                    uv_planes[0] = fu;
                    uv_planes[1] = fv;
                    split(fuv, uv_planes);
                    uv_planes[0] = bu;
                    uv_planes[1] = bv;
                    split(buv, uv_planes);

                    d_fu.upload(fu);
                    d_fv.upload(fv);
                    d_bu.upload(bu);
                    d_bv.upload(bv);

                    break;
                }

            case PYR_LK:
                {
                    makeGray(frame0, gray0);
                    makeGray(frame1, gray1);

                    d_frame0.upload(gray0);
                    d_frame1.upload(gray1);

                    int64 proc_start = getTickCount();
                    pyrlk.dense(d_frame0, d_frame1, d_fu, d_fv);
                    pyrlk.dense(d_frame1, d_frame0, d_bu, d_bv);
                    proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

                    d_fu.download(fu);
                    d_fv.download(fv);
                    d_bu.download(bu);
                    d_bv.download(bv);

                    break;
                }
            };

            getFlowField(fu, fv, flowFieldForward);
            getFlowField(bu, bv, flowFieldBackward);

            cv::split(frame0_32F, channels);

            d_b0.upload(channels[0]);
            d_g0.upload(channels[1]);
            d_r0.upload(channels[2]);

            cv::split(frame1_32F, channels);

            d_b1.upload(channels[0]);
            d_g1.upload(channels[1]);
            d_r1.upload(channels[2]);

            frames.clear();
            frames.push_back(frame0_32F.clone());

            // compute interpolated frames
            for (float timePos = timeStep; timePos < 1.0f; timePos += timeStep)
            {
                interpolateFrames(d_b0, d_b1, d_fu, d_fv, d_bu, d_bv, timePos, d_bNew, d_buf);
                interpolateFrames(d_g0, d_g1, d_fu, d_fv, d_bu, d_bv, timePos, d_gNew, d_buf);
                interpolateFrames(d_r0, d_r1, d_fu, d_fv, d_bu, d_bv, timePos, d_rNew, d_buf);

                GpuMat channels[] = {d_bNew, d_gNew, d_rNew};
                merge(channels, 3, d_newFrame);

                d_newFrame.download(newFrame);

                frames.push_back(newFrame.clone());
            }

            frames.push_back(frame1_32F.clone());

            currentFrame = 0;
            forward = true;
            calcFlow = false;

            total_fps = getTickFrequency()  / (getTickCount() - start);
        }

        imshow("First Frame", frame0);
        imshow("Second Frame", frame1);

        imshow("Forward flow", flowFieldForward);
        imshow("Backward flow", flowFieldBackward);

        frames[currentFrame].convertTo(img_to_show, CV_8U, 255.0);

        displayState(img_to_show, proc_fps, total_fps);

        imshow("Interpolated Frames", img_to_show);

        processKey(waitKey(100) & 0xff);

        if (forward)
        {
            ++currentFrame;
            if (currentFrame == static_cast<int>(frames.size()) - 1)
                forward = false;
        }
        else
        {
            --currentFrame;
            if (currentFrame == 0)
                forward = true;
        }
    }
}

void App::displayState(Mat& frame, double proc_fps, double total_fps)
{
    const Scalar fontColorRed = CV_RGB(255, 0, 0);

    int i = 0;

    ostringstream txt;
    txt.str(""); txt << "Source size: " << frame.cols << 'x' << frame.rows;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "Method: " << method_str[method];
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "FPS (OptFlow only): " << fixed << setprecision(1) << proc_fps;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "FPS (total): " << fixed << setprecision(1) << total_fps;
    printText(frame, txt.str(), i++);

    printText(frame, "Space - switch method", i++, fontColorRed);
    printText(frame, "I - switch source", i++, fontColorRed);
}

bool App::processKey(int key)
{
    if (BaseApp::processKey(key))
        return true;

    switch (toupper(key))
    {
    case 32:
        switch (method)
        {
        case BROX:
            method = FARNEBACK_GPU;
            break;
        case FARNEBACK_GPU:
            method = FARNEBACK_CPU;
            break;
        case FARNEBACK_CPU:
            method = PYR_LK;
            break;
        case PYR_LK:
            method = BROX;
            break;
        }
        cout << "method: " << method_str[method] << endl;
        calcFlow = true;
        break;

    case 'I':
        curSource = (curSource + 1) % pairSources.size();
        calcFlow = true;
        break;

    default:
        return false;
    }

    return true;
}

void App::printHelp()
{
    cout << "Usage: demo_dense_optical_flow <frame sources>\n";
    BaseApp::printHelp();
}

RUN_APP(App)

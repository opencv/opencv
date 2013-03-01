#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/legacy/legacy.hpp>

#include "utility.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

enum Method
{
    MOG,
    FGD,
    VIBE,
    METHOD_MAX
};

const char* method_str[] =
{
    "MOG",
    "FGD",
    "VIBE"
};

namespace cv
{
    template <>
    void Ptr<CvBGStatModel>::delete_obj()
    {
        cvReleaseBGStatModel(&obj);
        obj = 0;
    }
}

class App : public BaseApp
{
public:
    App();

protected:
    void process();
    bool processKey(int key);
    void printHelp();

private:
    void displayState(Mat& outImg, double proc_fps, double total_fps);

    Method method;
    bool useGPU;
    bool reinitialize;

    int curSource;

    BackgroundSubtractorMOG mog_cpu;
    MOG_GPU mog_gpu;

    FGDStatModel fgd_gpu;
    Ptr<CvBGStatModel> fgd_cpu;

    VIBE_GPU vibe_gpu;
};

App::App()
{
    method = MOG;
    useGPU = true;
    reinitialize = true;
    curSource = 0;
}

void App::process()
{
    if (sources.empty())
    {
        cout << "Using default frames source..." << endl;
        sources.push_back(new VideoSource("data/bgfg/haut-640x480.avi"));
    }

    Mat frame;
    GpuMat d_frame;
    IplImage ipl_frame;

    Mat fgmask;
    GpuMat d_fgmask;
    Mat buf;

    Mat outImg;
    Mat foreground;

    while (!exited)
    {
        int64 total_time = getTickCount();

        if (reinitialize)
        {
            mog_cpu = cv::BackgroundSubtractorMOG();
            mog_gpu.release();

            fgd_gpu.release();
            fgd_cpu.release();

            vibe_gpu.release();

            sources[curSource]->reset();
        }

        sources[curSource]->next(frame);
        d_frame.upload(frame);
        ipl_frame = frame;
        frame.copyTo(outImg);

        double total_fps = 0.0;
        double proc_fps = 0.0;

        try
        {
            int64 proc_time = getTickCount();

            switch (method)
            {
            case MOG:
                {
                    if (useGPU)
                    {
                        if (reinitialize)
                            mog_gpu.initialize(d_frame.size(), d_frame.type());
                        mog_gpu(d_frame, d_fgmask, 0.01f);
                    }
                    else
                    {
                        if (reinitialize)
                            mog_cpu.initialize(frame.size(), frame.type());
                        mog_cpu(frame, fgmask, 0.01);
                    }
                    break;
                }
            case FGD:
                {
                    if (useGPU)
                    {
                        if (reinitialize)
                            fgd_gpu.create(d_frame);
                        fgd_gpu.update(d_frame);
                        fgd_gpu.foreground.copyTo(d_fgmask);
                    }
                    else
                    {
                        if (reinitialize)
                            fgd_cpu = cvCreateFGDStatModel(&ipl_frame);
                        cvUpdateBGStatModel(&ipl_frame, fgd_cpu);
                        Mat(fgd_cpu->foreground).copyTo(fgmask);
                    }
                    break;
                }
            case VIBE:
                {
                    if (useGPU)
                    {
                        if (reinitialize)
                            vibe_gpu.initialize(d_frame);
                        vibe_gpu(d_frame, d_fgmask);
                    }
                    break;
                }
            }

            proc_fps = getTickFrequency() / (getTickCount() - proc_time);

            if (useGPU)
                d_fgmask.download(fgmask);

            filterSpeckles(fgmask, 0, 100, 1, buf);

            add(outImg, cv::Scalar(100, 100, 0), outImg, fgmask);

            foreground.create(frame.size(), frame.type());
            foreground.setTo(0);
            frame.copyTo(foreground, fgmask);

            total_fps = getTickFrequency() / (getTickCount() - total_time);

            reinitialize = false;
        }
        catch (const cv::Exception&)
        {
            string msg = "Not enough memory";

            int fontFace = cv::FONT_HERSHEY_DUPLEX;
            int fontThickness = 2;
            double fontScale = 0.8;

            Size msgSize = getTextSize(msg, fontFace, fontScale, fontThickness, 0);

            Point org(outImg.cols / 2 - msgSize.width / 2, outImg.rows / 2 - msgSize.height / 2);

            putText(outImg, msg, org, fontFace, fontScale, CV_RGB(0, 0, 0), 5 * fontThickness / 2, 16);
            putText(outImg, msg, org, fontFace, fontScale, CV_RGB(255, 0, 0), fontThickness, 16);

            foreground.create(frame.size(), frame.type());
            foreground.setTo(0);
            cv::putText(foreground, msg, org, fontFace, fontScale, CV_RGB(0, 0, 0), 5 * fontThickness / 2, 16);
            cv::putText(foreground, msg, org, fontFace, fontScale, CV_RGB(255, 0, 0), fontThickness, 16);
        }

        displayState(outImg, proc_fps, total_fps);

        imshow("Background Subtraction Demo", outImg);
        imshow("Foreground", foreground);

        processKey(waitKey(30));
    }
}

void App::displayState(Mat& frame, double proc_fps, double total_fps)
{
    const Scalar fontColorRed = CV_RGB(255, 0, 0);

    int i = 0;

    ostringstream txt;
    txt.str(""); txt << "Source size: " << frame.cols << 'x' << frame.rows;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "Method: " << method_str[method] << (useGPU ? " CUDA" : " CPU");
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "Process FPS: " << fixed << setprecision(1) << proc_fps;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "Total FPS: " << fixed << setprecision(1) << total_fps;
    printText(frame, txt.str(), i++);

    printText(frame, "M - switch method", i++, fontColorRed);
    printText(frame, "Space - switch CUDA/CPU mode", i++, fontColorRed);
    if (sources.size() > 1)
        printText(frame, "N - next source", i++, fontColorRed);
}

bool App::processKey(int key)
{
    if (BaseApp::processKey(key))
        return true;

    switch (toupper(key & 0xff))
    {
    case 'M':
        method = static_cast<Method>((method + 1) % METHOD_MAX);
        if (method == VIBE)
            useGPU = true;
        reinitialize = true;
        cout << "Switch method to " << method_str[method] << endl;
        break;

    case 'N':
        curSource = (curSource + 1) % sources.size();
        reinitialize = true;
        cout << "Switch source to " << curSource << endl;
        break;

    case 32 /*space*/:
        if (method != VIBE)
        {
            useGPU = !useGPU;
            reinitialize = true;
            cout << "Switch mode to " << (useGPU ? " CUDA" : " CPU") << endl;
        }

    default:
        return false;
    }

    return true;
}

void App::printHelp()
{
    cout << "This sample demonstrates different Background Subtraction Algoritms" << endl;
    cout << "Usage: demo_background_subtraction [options]" << endl;
    cout << "Options:" << endl;
    BaseApp::printHelp();
}

RUN_APP(App)

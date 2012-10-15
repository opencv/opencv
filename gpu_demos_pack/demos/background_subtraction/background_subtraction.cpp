#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/legacy/legacy.hpp>

#include "utility_lib/utility_lib.h"

enum Method
{
    MOG,
    //MOG2,
    FGD,
    //GMG,
    VIBE,
    METHOD_MAX
};

const char* method_str[] =
{
    "MOG",
    //"MOG2",
    "FGD",
    //"GMG",
    "VIBE"
};

namespace cv {
// TODO: conflicts with calib3d.hpp : filterSpeckles, should be removed ?

//! Speckle filtering - filters small connected components on diparity image.
//! It sets pixel (x,y) to newVal if it coresponds to small CC with size < maxSpeckleSize.
//! Threshold for border between CC is diffThreshold;
  CV_EXPORTS void filterSpeckles(Mat& img, uchar newVal, int maxSpeckleSize, uchar diffThreshold, Mat& buf);
}

namespace cv
{
    template <>
    void cv::Ptr<CvBGStatModel>::delete_obj()
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
    void printHelp();
    bool processKey(int key);

private:
    void displayState(cv::Mat& outImg, double proc_fps, double total_fps);
    void releaseAllAlgs();

    Method method;
    bool useGpu;
    bool reinitialize;

    int curSource;

    cv::BackgroundSubtractorMOG mog_cpu;
    cv::gpu::MOG_GPU mog_gpu;

    //cv::BackgroundSubtractorMOG2 mog2_cpu;
    //cv::gpu::MOG2_GPU mog2_gpu;

    cv::gpu::FGDStatModel fgd_gpu;
    cv::Ptr<CvBGStatModel> fgd_cpu;

    //cv::gpu::GMG_GPU gmg_gpu;
    //cv::BackgroundSubtractorGMG gmg_cpu;

    cv::gpu::VIBE_GPU vibe_gpu;
};

App::App()
{
    method = MOG;
    useGpu = true;
    reinitialize = true;
    curSource = 0;

    //mog2_cpu.set("detectShadows", false);
    //mog2_gpu.bShadowDetection = false;
}

void App::process()
{
    if (sources.empty())
    {
        std::cout << "Using default frames source...\n";
        sources.push_back(new VideoSource("data/bgfg/haut-640x480.avi"));
    }

    cv::Mat frame;
    cv::gpu::GpuMat d_frame;
    IplImage ipl_frame;

    cv::Mat fgmask;
    cv::gpu::GpuMat d_fgmask;
    cv::Mat buf;

    cv::Mat outImg;
    cv::Mat foreground;

    while (!exited)
    {
        int64 total_time = cv::getTickCount();

        sources[curSource]->next(frame);
        d_frame.upload(frame);
        ipl_frame = frame;
        frame.copyTo(outImg);

        double total_fps = 0.0;
        double proc_fps = 0.0;

        try
        {
            int64 proc_time = cv::getTickCount();

            switch (method) {
            case MOG:
                {
                    if (useGpu)
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
            //case MOG2:
            //    {
            //        if (useGpu)
            //        {
            //            if (reinitialize)
            //                mog2_gpu.initialize(d_frame.size(), d_frame.type());
            //            mog2_gpu(d_frame, d_fgmask);
            //        }
            //        else
            //        {
            //            if (reinitialize)
            //            {
            //                mog2_cpu.set("detectShadows", false);
            //                mog2_cpu.initialize(frame.size(), frame.type());
            //            }
            //            mog2_cpu(frame, fgmask);
            //        }
            //        break;
            //    }
            case FGD:
                {
                    if (useGpu)
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
                        cv::Mat(fgd_cpu->foreground).copyTo(fgmask);
                    }
                    break;
                }
            //case GMG:
            //    {
            //        if (useGpu)
            //        {
            //            if (reinitialize)
            //                gmg_gpu.initialize(d_frame.size());
            //            gmg_gpu(d_frame, d_fgmask);
            //        }
            //        else
            //        {
            //            if (reinitialize)
            //                gmg_cpu.initialize(frame.size(), 0, 255);
            //            gmg_cpu(frame, fgmask);
            //        }
            //        break;
            //    }
            case VIBE:
                {
                    if (useGpu)
                    {
                        if (reinitialize)
                            vibe_gpu.initialize(d_frame);
                        vibe_gpu(d_frame, d_fgmask);
                    }
                    break;
                }
            }

            proc_fps = cv::getTickFrequency() / (cv::getTickCount() - proc_time);

            if (useGpu)
                d_fgmask.download(fgmask);

            cv::filterSpeckles(fgmask, 0, 100, 1, buf);

            cv::add(outImg, cv::Scalar(100, 100, 0), outImg, fgmask);

            foreground.create(frame.size(), frame.type());
            foreground.setTo(0);
            frame.copyTo(foreground, fgmask);

            total_fps = cv::getTickFrequency() / (cv::getTickCount() - total_time);

            reinitialize = false;
        }
        catch (const cv::Exception&)
        {
            std::string msg = "Can't allocate memory";

            int fontFace = cv::FONT_HERSHEY_DUPLEX;
            int fontThickness = 2;
            double fontScale = 0.8;

            cv::Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

            cv::Point org(outImg.cols / 2, outImg.rows / 2);
            org.x -= fontSize.width;
            org.y -= fontSize.height / 2;

            cv::putText(outImg, msg, org, fontFace, fontScale, cv::Scalar(0,0,0,255), 5 * fontThickness / 2, 16);
            cv::putText(outImg, msg, org, fontFace, fontScale, CV_RGB(255, 0, 0), fontThickness, 16);
        }

        displayState(outImg, proc_fps, total_fps);

        cv::imshow("Background Subtraction Demo", outImg);
        cv::imshow("Foreground", foreground);

        processKey(cv::waitKey(30) & 0xff);
    }
}

void App::displayState(cv::Mat& frame, double proc_fps, double total_fps)
{
    const cv::Scalar fontColorRed = CV_RGB(255, 0, 0);

    int i = 0;

    std::ostringstream txt;
    txt.str(""); txt << "Source size: " << frame.cols << 'x' << frame.rows;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "Method: " << method_str[method] << (useGpu ? " CUDA" : " CPU");
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "Process FPS: " << std::fixed << std::setprecision(1) << proc_fps;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "Total FPS: " << std::fixed << std::setprecision(1) << total_fps;
    printText(frame, txt.str(), i++);

    printText(frame, "M - switch method", i++, fontColorRed);
    printText(frame, "Space - switch CUDA/CPU mode", i++, fontColorRed);
    if (sources.size() > 1)
        printText(frame, "S - switch source", i++, fontColorRed);
}

void App::printHelp()
{
    std::cout << "Usage: demo_background_subtraction <frame_source>\n";
    BaseApp::printHelp();
}

void App::releaseAllAlgs()
{
    mog_cpu = cv::BackgroundSubtractorMOG();
    mog_gpu.release();

    //mog2_cpu = cv::BackgroundSubtractorMOG2();
    //mog2_gpu.release();

    fgd_gpu.release();
    fgd_cpu.release();

    //gmg_gpu.release();
    //gmg_cpu.release();

    vibe_gpu.release();
}

bool App::processKey(int key)
{
    if (BaseApp::processKey(key))
        return true;

    switch (toupper(key))
    {
    case 'M':
        method = static_cast<Method>((method + 1) % METHOD_MAX);
        #if defined(_LP64) || defined(_WIN64)
        if (method == VIBE)
            useGpu = true;
        #else
        if (method == VIBE || method == FGD)
            useGpu = true;
        #endif
        reinitialize = true;
        releaseAllAlgs();
        std::cout << "Switch method to " << method_str[method] << std::endl;
        break;

    case 'S':
        curSource = (curSource + 1) % sources.size();
        reinitialize = true;
        std::cout << "Switch source to " << curSource << std::endl;
        break;

    case 32:
        #if defined(_LP64) || defined(_WIN64)
        if (method != VIBE)
        {
            useGpu = !useGpu;
            reinitialize = true;
            releaseAllAlgs();
            std::cout << "Switch mode to " << (useGpu ? " CUDA" : " CPU") << std::endl;
        }
        #else
        if (method != VIBE && method != FGD)
        {
            useGpu = !useGpu;
            reinitialize = true;
            releaseAllAlgs();
            std::cout << "Switch mode to " << (useGpu ? " CUDA" : " CPU") << std::endl;
        }
        #endif

    default:
        return false;
    }

    return true;
}

RUN_APP(App)

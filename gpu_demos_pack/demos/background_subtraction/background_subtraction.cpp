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
    MOG2,
    FGD,
    GMG,
    VIBE,
    METHOD_MAX
};

const char* method_str[] =
{
    "MOG",
    "MOG2",
    "FGD",
    "GMG",
    "VIBE"
};

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

    Method method;
    bool useGpu;
    bool reinitialize;

    cv::BackgroundSubtractorMOG mog_cpu;
    cv::gpu::MOG_GPU mog_gpu;

    cv::BackgroundSubtractorMOG2 mog2_cpu;
    cv::gpu::MOG2_GPU mog2_gpu;

    cv::gpu::VIBE_GPU vibe_gpu;

    cv::gpu::FGDStatModel fgd_gpu;
    cv::Ptr<CvBGStatModel> fgd_cpu;

    cv::gpu::GMG_GPU gmg_gpu;
    cv::BackgroundSubtractorGMG gmg_cpu;
};

App::App()
{
    method = MOG;
    useGpu = true;
    reinitialize = true;

    gmg_gpu.numInitializationFrames = 40;
    gmg_cpu.numInitializationFrames = 40;
}

void App::process()
{
    if (sources.size() != 1)
    {
        std::cout << "Using default frames source...\n";
        sources.resize(1);
        sources[0] = new VideoSource("data/pedestrian_detect/mitsubishi.avi");
    }

    cv::Mat frame3;
    cv::Mat frame;
    cv::gpu::GpuMat d_frame;
    IplImage ipl_frame;

    cv::Mat fgmask;
    cv::gpu::GpuMat d_fgmask;

    cv::Mat outImg;

    while (!exited)
    {
        int64 total_time = cv::getTickCount();

        sources[0]->next(frame3);
        cv::cvtColor(frame3, frame, cv::COLOR_BGR2BGRA);
        d_frame.upload(frame);
        ipl_frame = frame3;

        int64 proc_time = cv::getTickCount();

        switch (method) {
        case MOG:
            {
                if (useGpu)
                {
                    if (reinitialize)
                    {
                        mog_gpu.initialize(d_frame.size(), d_frame.type());
                        reinitialize = false;
                    }

                    mog_gpu(d_frame, d_fgmask, 0.01);
                }
                else
                {
                    if (reinitialize)
                    {
                        mog_cpu.initialize(frame3.size(), frame3.type());
                        reinitialize = false;
                    }

                    mog_cpu(frame3, fgmask, 0.01);
                }
                break;
            }
        case MOG2:
            {
                if (useGpu)
                {
                    if (reinitialize)
                    {
                        mog2_gpu.initialize(d_frame.size(), d_frame.type());
                        reinitialize = false;
                    }

                    mog2_gpu(d_frame, d_fgmask);
                }
                else
                {
                    if (reinitialize)
                    {
                        mog2_cpu.initialize(frame3.size(), frame3.type());
                        reinitialize = false;
                    }

                    mog2_cpu(frame3, fgmask);
                }
                break;
            }
        case GMG:
            {
                if (useGpu)
                {
                    if (reinitialize)
                    {
                        gmg_gpu.initialize(d_frame.size());
                        reinitialize = false;
                    }

                    gmg_gpu(d_frame, d_fgmask);
                }
                else
                {
                    if (reinitialize)
                    {
                        gmg_cpu.initialize(frame3.size(), 0, 255);
                        reinitialize = false;
                    }

                    gmg_cpu(frame3, fgmask);
                }
                break;
            }
        case FGD:
            {
                if (useGpu)
                {
                    if (reinitialize)
                    {
                        fgd_gpu.create(d_frame);
                        reinitialize = false;
                    }

                    fgd_gpu.update(d_frame);
                    d_fgmask = fgd_gpu.foreground;
                }
                else
                {
                    if (reinitialize)
                    {
                        fgd_cpu = cvCreateFGDStatModel(&ipl_frame);
                        reinitialize = false;
                    }

                    cvUpdateBGStatModel(&ipl_frame, fgd_cpu);
                    fgmask = fgd_cpu->foreground;
                }
                break;
            }
        case VIBE:
            {
                if (useGpu)
                {
                    if (reinitialize)
                    {
                        vibe_gpu.initialize(d_frame);
                        reinitialize = false;
                    }

                    vibe_gpu(d_frame, d_fgmask);
                }
                break;
            }
        }

        double proc_fps = cv::getTickFrequency() / (cv::getTickCount() - proc_time);

        if (useGpu)
            d_fgmask.download(fgmask);

        frame.copyTo(outImg);
        cv::add(outImg, cv::Scalar(100, 100, 0), outImg, fgmask);

        double total_fps = cv::getTickFrequency() / (cv::getTickCount() - total_time);

        displayState(outImg, proc_fps, total_fps);

        cv::imshow("Background Subtraction Demo", outImg);

        processKey(cv::waitKey(10) & 0xff);
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
}

void App::printHelp()
{
    std::cout << "Usage: demo_background_subtraction <frame_source>\n";
    BaseApp::printHelp();
}

bool App::processKey(int key)
{
    if (BaseApp::processKey(key))
        return true;

    switch (toupper(key))
    {
    case 'M':
        method = static_cast<Method>((method + 1) % METHOD_MAX);
        if (method == VIBE)
            useGpu = true;
        reinitialize = true;
        std::cout << "Switch method to " << method_str[method] << std::endl;
        break;

    case 32:
        if (method != VIBE)
        {
            useGpu = !useGpu;
            reinitialize = true;
            std::cout << "Switch mode to " << (useGpu ? " CUDA" : " CPU") << std::endl;
        }

    default:
        return false;
    }

    return true;
}

RUN_APP(App)

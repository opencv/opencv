#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/video/video.hpp>

#include "utility_lib/utility_lib.h"

enum Method
{
    MOG_CPU,
    MOG_GPU,
    MOG2_CPU,
    MOG2_GPU,
    VIBE_GPU,
    FGD_STAT_GPU
};

const char* method_str[] =
{
    "MOG CPU",
    "MOG CUDA",
    "MOG2 CPU",
    "MOG2 CUDA",
    "VIBE CUDA",
    "FGD STAT CUDA"
};

class App : public BaseApp
{
public:
    App();

protected:
    void process();
    void printHelp();
    bool processKey(int key);

private:
    void displayState(cv::Mat& frame, double fps);

    Method method;

    cv::BackgroundSubtractorMOG mog_cpu;
    cv::gpu::MOG_GPU mog_gpu;

    cv::BackgroundSubtractorMOG2 mog2_cpu;
    cv::gpu::MOG2_GPU mog2_gpu;

    cv::gpu::VIBE_GPU vibe_gpu;

    cv::gpu::FGDStatModel fgd_stat_gpu;
};

App::App()
{
    method = MOG_CPU;
}

void App::process()
{
    if (sources.size() != 1)
    {
        std::cout << "Using default frames source...\n";
        sources.resize(1);
        sources[0] = new VideoSource("data/pedestrian_detect/mitsubishi.avi");
    }

    cv::Mat frame;
    cv::gpu::GpuMat d_frame;

    cv::Mat dst;
    cv::Mat fgmask;
    cv::Mat fgimg;

    cv::gpu::GpuMat d_fgmask;

    sources[0]->next(frame);
    d_frame.upload(frame);

    mog_cpu(frame, fgmask, 0.01);
    mog_gpu(d_frame, d_fgmask, 0.01);

    mog2_cpu(frame, fgmask);
    mog2_gpu(d_frame, d_fgmask);

    vibe_gpu.initialize(d_frame);

    fgd_stat_gpu.create(d_frame);

    while (!exited)
    {
        sources[0]->next(frame);

        int64 upload_time = 0;
        int64 proc_time = 0;

        // Upload
        {
            int64 upload_start = cv::getTickCount();

            d_frame.upload(frame);

            if (method == MOG_GPU || method == MOG2_GPU || method == VIBE_GPU || method == FGD_STAT_GPU)
                upload_time = cv::getTickCount() - upload_start;
        }

        // MOG_CPU
        {
            int64 proc_start = cv::getTickCount();

            mog_cpu(frame, dst, 0.01);

            if (method == MOG_CPU)
            {
                proc_time = cv::getTickCount() - proc_start;
                dst.copyTo(fgmask);
            }
        }

        // MOG_GPU
        {
            int64 proc_start = cv::getTickCount();

            mog_gpu(d_frame, d_fgmask, 0.01);

            if (method == MOG_GPU)
            {
                d_fgmask.download(fgmask);
                proc_time = cv::getTickCount() - proc_start;
            }
        }

        // MOG2_CPU
        {
            int64 proc_start = cv::getTickCount();

            mog2_cpu(frame, dst);

            if (method == MOG2_CPU)
            {
                proc_time = cv::getTickCount() - proc_start;
                dst.copyTo(fgmask);
            }
        }

        // MOG2_GPU
        {
            int64 proc_start = cv::getTickCount();

            mog2_gpu(d_frame, d_fgmask);

            if (method == MOG2_GPU)
            {
                d_fgmask.download(fgmask);
                proc_time = cv::getTickCount() - proc_start;
            }
        }

        // VIBE_GPU
        {
            int64 proc_start = cv::getTickCount();

            vibe_gpu(d_frame, d_fgmask);

            if (method == VIBE_GPU)
            {
                d_fgmask.download(fgmask);
                proc_time = cv::getTickCount() - proc_start;
            }
        }

        // FGD_STAT_GPU
        {
            int64 proc_start = cv::getTickCount();

            fgd_stat_gpu.update(d_frame);

            if (method == FGD_STAT_GPU)
            {
                fgd_stat_gpu.foreground.download(fgmask);
                proc_time = cv::getTickCount() - proc_start;
            }
        }

        double fps = cv::getTickFrequency() / proc_time;

        fgimg.setTo(0);
        frame.copyTo(fgimg, fgmask);

        displayState(fgimg, fps);

        cv::imshow("Source Frame", frame);
        cv::imshow("Foreground Mask", fgmask);
        cv::imshow("Foreground Image", fgimg);

        processKey(cv::waitKey(3) & 0xff);
    }
}

void App::displayState(cv::Mat& frame, double fps)
{
    const cv::Scalar fontColorRed = CV_RGB(255, 0, 0);

    int i = 0;

    std::ostringstream txt;
    txt.str(""); txt << "Source size: " << frame.cols << 'x' << frame.rows;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "Method: " << method_str[method];
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "FPS: " << std::fixed << std::setprecision(1) << fps;
    printText(frame, txt.str(), i++);

    printText(frame, "Space - switch method", i++, fontColorRed);
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
    case 32:
        switch (method)
        {
        case MOG_CPU:
            method = MOG_GPU;
            break;
        case MOG_GPU:
            method = MOG2_CPU;
            break;
        case MOG2_CPU:
            method = MOG2_GPU;
            break;
        case MOG2_GPU:
            method = VIBE_GPU;
            break;
        case VIBE_GPU:
            method = FGD_STAT_GPU;
            break;
        case FGD_STAT_GPU:
            method = MOG_CPU;
            break;
        }
        break;

    default:
        return false;
    }

    return true;
}

RUN_APP(App)

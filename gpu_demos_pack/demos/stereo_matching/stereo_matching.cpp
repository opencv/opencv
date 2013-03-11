#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "utility.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

class App : public BaseApp
{
public:
    App();

protected:
    void runAppLogic();
    void processAppKey(int key);
    void printAppHelp();
    bool parseAppCmdArgs(int& i, int argc, const char* argv[]);

private:
    void displayState(Mat& outImg, double proc_fps, double total_fps);

    vector< Ptr<PairFrameSource> > pairSources_;

    bool useGpu_;
    bool colorOutput_;
    bool showSource_;
    int curSource_;
    bool fullscreen_;
};

App::App()
{
    useGpu_ = true;
    colorOutput_ = true;
    showSource_ = true;
    curSource_ = 0;
    fullscreen_ = true;
}

void App::runAppLogic()
{
    if (sources_.size() > 1)
    {
        for (size_t i = 0; (i + 1) < sources_.size(); i += 2)
            pairSources_.push_back(PairFrameSource::create(sources_[i], sources_[i+1]));
    }
    else
    {
        cout << "Using default frames source... \n" << endl;

        pairSources_.push_back(PairFrameSource::create(FrameSource::video("data/stereo_matching_L.avi"),
                                                       FrameSource::video("data/stereo_matching_R.avi")));
    }

    StereoBM_GPU bm_gpu;
    bm_gpu.ndisp = 256;
    StereoBM bm_cpu;
    bm_cpu.init(StereoBM::BASIC_PRESET, bm_gpu.ndisp, bm_gpu.winSize);

    Mat left_src, right_src;
    Mat left, right;
    GpuMat d_left, d_right;
    Mat small_image;

    Mat disp, disp_16s;
    GpuMat d_disp, d_img_to_show;

    Mat outImg;

    const string wndName = "Stereo Matching Demo";

    if (fullscreen_)
    {
        namedWindow(wndName, WINDOW_NORMAL);
        setWindowProperty(wndName, WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        setWindowProperty(wndName, WND_PROP_ASPECT_RATIO, CV_WINDOW_FREERATIO);
    }

    while (isActive())
    {
        const int64 total_start = getTickCount();

        pairSources_[curSource_]->next(left_src, right_src);

        makeGray(left_src, left);
        makeGray(right_src, right);

        if (useGpu_)
        {
            d_left.upload(left);
            d_right.upload(right);
        }

        const int64 proc_start = getTickCount();

        if (useGpu_)
            bm_gpu(d_left, d_right, d_disp);
        else
            bm_cpu(left, right, disp_16s);

        const double proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

        if (colorOutput_)
        {
            if (!useGpu_)
            {
                disp_16s.convertTo(disp, CV_8U, 1.0 / 16.0);
                d_disp.upload(disp);
            }

            drawColorDisp(d_disp, d_img_to_show, bm_gpu.ndisp);
            d_img_to_show.download(outImg);
        }
        else
        {
            if (!useGpu_)
            {
                disp_16s.convertTo(disp, CV_8U, 255.0 / (16.0 * bm_gpu.ndisp));
                cvtColor(disp, outImg, COLOR_GRAY2BGR);
            }
            else
            {
                d_disp.convertTo(d_disp, d_disp.depth(), 255.0 / bm_gpu.ndisp);
                gpu::cvtColor(d_disp, d_img_to_show, COLOR_GRAY2BGR);
                d_img_to_show.download(outImg);
            }
        }

        if (showSource_)
        {
            resize(left_src, small_image, cv::Size(), 0.25, 0.25);
            Mat roi = outImg(cv::Rect(outImg.cols - small_image.cols, 0, small_image.cols, small_image.rows));

            if (colorOutput_)
                cvtColor(small_image, roi, cv::COLOR_BGR2BGRA);
            else
                small_image.copyTo(roi);
        }

        const double total_fps = getTickFrequency()  / (getTickCount() - total_start);

        displayState(outImg, proc_fps, total_fps);

        imshow(wndName, outImg);

        wait(30);
    }
}

void App::displayState(cv::Mat& outImg, double proc_fps, double total_fps)
{
    const Scalar fontColorRed = CV_RGB(255, 0, 0);

    ostringstream txt;
    int i = 0;

    txt.str(""); txt << "Source size: " << outImg.cols << 'x' << outImg.rows;
    printText(outImg, txt.str(), i++);

    printText(outImg, useGpu_ ? "Mode: CUDA" : "Mode: CPU", i++);

    txt.str(""); txt << "FPS (Stereo only): " << fixed << setprecision(1) << proc_fps;
    printText(outImg, txt.str(), i++);

    txt.str(""); txt << "FPS (total): " << fixed << setprecision(1) << total_fps;
    printText(outImg, txt.str(), i++);

    printText(outImg, "Space - switch CUDA / CPU mode", i++, fontColorRed);
    printText(outImg, "C - switch Color / Gray mode", i++, fontColorRed);
    printText(outImg, "S - show / hide source frame", i++, fontColorRed);
    if (pairSources_.size() > 1)
        printText(outImg, "N - switch source", i++, fontColorRed);
}

void App::processAppKey(int key)
{
    switch (toupper(key & 0xff))
    {
    case 32 /*space*/:
        useGpu_ = !useGpu_;
        cout << "Switch mode to " << (useGpu_ ? "CUDA" : "CPU") << endl;
        break;

    case 'C':
        colorOutput_ = !colorOutput_;
        cout << "Switch mode to " << (colorOutput_ ? "Color" : "Gray") << endl;
        break;

    case 'S':
        showSource_ = !showSource_;
        cout << (showSource_ ? "Show source" : "Hide source") << endl;
        break;

    case 'N':
        if (pairSources_.size() > 1)
        {
            curSource_ = (curSource_ + 1) % pairSources_.size();
            pairSources_[curSource_]->reset();
            cout << "Switch source to " << curSource_ << endl;
        }
        break;
    }
}

void App::printAppHelp()
{
    cout << "This sample demonstrates Stereo Matching algorithm \n" << endl;

    cout << "Usage: demo_stereo_matching [options] \n" << endl;

    cout << "Launch Options: \n"
         << "  --windowed \n"
         << "       Launch in windowed mode\n" << endl;
}

bool App::parseAppCmdArgs(int& i, int, const char* argv[])
{
    string arg = argv[i];

    if (arg == "--windowed")
    {
        fullscreen_ = false;
        return true;
    }
    else
        return false;

    return true;
}

RUN_APP(App)

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/photo/photo.hpp>

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
    void addGaussNoise(Mat& image, double sigma);

    bool useGpu_;
    bool colorInput_;
    int curSource_;
    bool fullscreen_;

    Mat noise_;
};

App::App()
{
    useGpu_ = true;
    colorInput_ = true;
    curSource_ = 0;
    fullscreen_ = false;
}

void App::runAppLogic()
{
    if (sources_.empty())
    {
        cout << "Using default frames source... \n" << endl;
        sources_.push_back(FrameSource::image("data/denoising.jpg"));
    }

    FastNonLocalMeansDenoising d_nlm;

    Mat frame, src, dst, outImg;
    GpuMat d_src, d_dst;

    const string wndName = "Denoising Demo";

    if (fullscreen_)
    {
        namedWindow(wndName, WINDOW_NORMAL);
        setWindowProperty(wndName, WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        setWindowProperty(wndName, WND_PROP_ASPECT_RATIO, CV_WINDOW_FREERATIO);
    }

    while (isActive())
    {
        const int64 total_start = getTickCount();

        sources_[curSource_]->next(frame);

        if (colorInput_)
            frame.copyTo(src);
        else
            cvtColor(frame, src, COLOR_BGR2GRAY);

        addGaussNoise(src, 20.0);

        if (useGpu_)
            d_src.upload(src);

        const int64 proc_start = getTickCount();

        if (useGpu_)
        {
            if (colorInput_)
                d_nlm.labMethod(d_src, d_dst, 20, 10);
            else
                d_nlm.simpleMethod(d_src, d_dst, 20);
        }
        else
        {
            if (colorInput_)
                fastNlMeansDenoisingColored(src, dst, 20, 10);
            else
                fastNlMeansDenoising(src, dst, 20);
        }

        const double proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

        if (useGpu_)
            d_dst.download(dst);

        outImg.create(frame.rows, frame.cols * 2, CV_8UC3);

        Mat left = outImg(Rect(0, 0, frame.cols, frame.rows));
        Mat right = outImg(Rect(frame.cols, 0, frame.cols, frame.rows));

        if (colorInput_)
        {
            src.copyTo(left);
            dst.copyTo(right);
        }
        else
        {
            cvtColor(src, left, COLOR_GRAY2BGR);
            cvtColor(dst, right, COLOR_GRAY2BGR);
        }

        const double total_fps = getTickFrequency()  / (getTickCount() - total_start);

        displayState(outImg, proc_fps, total_fps);

        imshow(wndName, outImg);

        wait(30);
    }
}

void App::displayState(Mat& outImg, double proc_fps, double total_fps)
{
    const Scalar fontColorRed = CV_RGB(255, 0, 0);

    ostringstream txt;
    int i = 0;

    txt.str(""); txt << "Source size: " << outImg.cols / 2 << 'x' << outImg.rows;
    printText(outImg, txt.str(), i++);

    printText(outImg, useGpu_ ? "Mode: CUDA" : "Mode: CPU", i++);

    txt.str(""); txt << "FPS (Denoising only): " << fixed << setprecision(1) << proc_fps;
    printText(outImg, txt.str(), i++);

    txt.str(""); txt << "FPS (total): " << fixed << setprecision(1) << total_fps;
    printText(outImg, txt.str(), i++);

    printText(outImg, "Space - switch CUDA / CPU mode", i++, fontColorRed);
    printText(outImg, "C - switch Color / Gray mode", i++, fontColorRed);
    if (sources_.size() > 1)
        printText(outImg, "N - switch source", i++, fontColorRed);
}

void App::addGaussNoise(Mat& image, double sigma)
{
    noise_.create(image.size(), CV_32FC(image.channels()));
    theRNG().fill(noise_, RNG::NORMAL, 0.0, sigma);

    addWeighted(image, 1.0, noise_, 1.0, 0.0, image, image.depth());
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
        colorInput_ = !colorInput_;
        cout << "Switch mode to " << (colorInput_ ? "Color" : "Gray") << endl;
        break;

    case 'N':
        if (sources_.size() > 1)
        {
            curSource_ = (curSource_ + 1) % sources_.size();
            sources_[curSource_]->reset();
            cout << "Switch source to " << curSource_ << endl;
        }
        break;
    }
}

void App::printAppHelp()
{
    cout << "This sample demonstrates Non-Local-Means Denoising algorithm \n" << endl;

    cout << "Usage: demo_denoising [options] \n" << endl;

    cout << "Launch Options: \n"
         << "  --fullscreen \n"
         << "       Launch in fullscreen mode \n" << endl;
}

bool App::parseAppCmdArgs(int& i, int, const char* argv[])
{
    string arg(argv[i]);

    if (arg == "--fullscreen")
    {
        fullscreen_ = true;
        return true;
    }

    return false;
}

RUN_APP(App)

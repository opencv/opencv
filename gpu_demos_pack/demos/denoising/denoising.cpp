#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/photo/photo.hpp>

#include "utility.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

class App : public BaseApp
{
public:
    App();

protected:
    void process();
    bool processKey(int key);
    void printHelp();

private:
    void displayState(Mat& frame, double proc_fps, double total_fps);

    bool useGPU;
    bool colorInput;
    int curSource;

    FastNonLocalMeansDenoising d_nlm;
};

App::App()
{
    useGPU = true;
    colorInput = false;
    curSource = 0;
}

void addGaussNoise(Mat& image, double sigma)
{
    Mat noise(image.size(), CV_32FC(image.channels()));
    theRNG().fill(noise, RNG::NORMAL, 0.0, sigma);

    addWeighted(image, 1.0, noise, 1.0, 0.0, image, image.depth());
}

void App::process()
{
    if (sources.empty())
    {
        cout << "Using default frames source..." << endl;
        sources.push_back(new ImageSource("data/denoising.jpg"));
    }

    Mat frame;
    Mat src, dst;
    GpuMat d_src, d_dst;

    Mat img_to_show;

    const float h = 20;

    while (!exited)
    {
        int64 start = getTickCount();

        sources[curSource]->next(frame);

        if (colorInput)
            src = frame;
        else
            cvtColor(frame, src, COLOR_BGR2GRAY);

        addGaussNoise(src, 20.0);

        int64 proc_start = getTickCount();

        if (useGPU)
        {
            d_src.upload(src);

            d_nlm.simpleMethod(d_src, d_dst, h);

            d_dst.download(dst);
        }
        else
        {
            if (colorInput)
                fastNlMeansDenoisingColored(src, dst, h, 10);
            else
                fastNlMeansDenoising(src, dst, h);
        }

        double proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

        img_to_show.create(frame.rows, frame.cols * 2, CV_8UC3);

        Mat left = img_to_show(Rect(0, 0, frame.cols, frame.rows));
        Mat right = img_to_show(Rect(frame.cols, 0, frame.cols, frame.rows));

        if (colorInput)
        {
            src.copyTo(left);
            dst.copyTo(right);
        }
        else
        {
            cvtColor(src, left, COLOR_GRAY2BGR);
            cvtColor(dst, right, COLOR_GRAY2BGR);
        }

        double total_fps = getTickFrequency()  / (getTickCount() - start);

        displayState(img_to_show, proc_fps, total_fps);

        imshow("Denoising Demo", img_to_show);

        processKey(waitKey(30));
    }
}

void App::displayState(Mat& frame, double proc_fps, double total_fps)
{
    const Scalar fontColorRed = CV_RGB(255, 0, 0);

    int i = 0;

    ostringstream txt;
    txt.str(""); txt << "Source size: " << frame.cols << 'x' << frame.rows / 2;
    printText(frame, txt.str(), i++);

    printText(frame, useGPU ? "Mode: CUDA" : "Mode: CPU", i++);

    txt.str(""); txt << "FPS (Denoising only): " << fixed << setprecision(1) << proc_fps;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "FPS (total): " << fixed << setprecision(1) << total_fps;
    printText(frame, txt.str(), i++);

    printText(frame, "Space - switch GPU / CPU", i++, fontColorRed);
    printText(frame, "C - switch Color / Gray", i++, fontColorRed);
    printText(frame, "N - next source", i++, fontColorRed);
}

bool App::processKey(int key)
{
    if (BaseApp::processKey(key))
        return true;

    switch (toupper(key & 0xff))
    {
    case 32:
        useGPU = !useGPU;
        cout << "Switched to " << (useGPU ? "CUDA" : "CPU") << " mode" << endl;
        break;

    case 'C':
        colorInput = !colorInput;
        cout << "Switched to " << (colorInput ? "Color" : "Gray") << " mode" << endl;
        break;

    case 'N':
        curSource = (curSource + 1) % sources.size();
        sources[curSource]->reset();
        cout << "Switch source to " << curSource << endl;
        break;

    default:
        return false;
    }

    return true;
}

void App::printHelp()
{
    cout << "This sample demonstrates Non-Local_means Denoising algorithm" << endl;
    cout << "Usage: demo_denoising [options]" << endl;
    cout << "Options:" << endl;
    BaseApp::printHelp();
}

RUN_APP(App)

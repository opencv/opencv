#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

    bool useGpu_;
    bool colorInput_;
    int curSource_;
    bool fullscreen_;
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
        cout << "Loading default frames source... \n" << endl;
        sources_.push_back(FrameSource::video("data/pedestrian_detection.avi"));
    }

    const double scale = 1.05;
    const int nlevels = 13;
    const int gr_threshold = 8;

    const double hit_threshold = 1.4;
    const bool gamma_corr = false;

    const Size win_size(48, 96);
    const Size win_stride(8, 8);

    const vector<float> detector = gpu::HOGDescriptor::getPeopleDetector48x96();

    gpu::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9,
                               gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, gamma_corr,
                               gpu::HOGDescriptor::DEFAULT_NLEVELS);
    cv::HOGDescriptor cpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1,
                              cv::HOGDescriptor::L2Hys, 0.2, gamma_corr,
                              cv::HOGDescriptor::DEFAULT_NLEVELS);

    gpu_hog.setSVMDetector(detector);
    cpu_hog.setSVMDetector(detector);

    gpu_hog.nlevels = nlevels;
    cpu_hog.nlevels = nlevels;

    Mat frame, img, outImg;
    GpuMat gpu_img;
    vector<Rect> rects;

    const string wndName = "Pedestrian Detection Demo";

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

        if (!colorInput_)
            cvtColor(frame, img, CV_BGR2GRAY);
        else if (useGpu_)
            cvtColor(frame, img, CV_BGR2BGRA);
        else
            frame.copyTo(img);

        if (colorInput_)
            frame.copyTo(outImg);
        else
            cvtColor(img, outImg, CV_GRAY2BGR);

        if (useGpu_)
            gpu_img.upload(img);

        const int64 proc_start = getTickCount();

        if (useGpu_)
            gpu_hog.detectMultiScale(gpu_img, rects, hit_threshold, win_stride, Size(0, 0), scale, gr_threshold);
        else
            cpu_hog.detectMultiScale(img, rects, hit_threshold, win_stride, Size(0, 0), scale, gr_threshold);

        const double proc_fps = getTickFrequency() / (getTickCount() - proc_start);

        for (size_t i = 0; i < rects.size(); i++)
            rectangle(outImg, rects[i], CV_RGB(0, 255, 0), 3);

        const double total_fps = getTickFrequency() / (getTickCount() - total_start);

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

    txt.str(""); txt << "Source size: " << outImg.cols << 'x' << outImg.rows;
    printText(outImg, txt.str(), i++);

    printText(outImg, useGpu_ ? "Mode: CUDA" : "Mode: CPU", i++);

    txt.str(""); txt << "FPS (PD only): " << fixed << setprecision(1) << proc_fps;
    printText(outImg, txt.str(), i++);

    txt.str(""); txt << "FPS (total): " << fixed << setprecision(1) << total_fps;
    printText(outImg, txt.str(), i++);

    printText(outImg, "Space - switch CUDA / CPU mode", i++, fontColorRed);
    printText(outImg, "C - switch Color / Gray mode", i++, fontColorRed);
    if (sources_.size() > 1)
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
    cout << "This sample demonstrates Pedestrian Detection algorithm \n" << endl;

    cout << "Usage: demo_pedestrian_detection [options] \n" << endl;

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

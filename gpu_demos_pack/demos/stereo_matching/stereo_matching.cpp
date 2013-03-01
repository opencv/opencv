#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "utility.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

enum Method
{
    BM,
    BP,
    CSBP
};

const char* method_str[] =
{
    "BM",
    "BP",
    "CSBP"
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
    void displayState(Mat& frame, double proc_fps, double total_fps);

    Method method;
    bool colorOutput;
    bool showHelp;
    bool showSource;

    StereoBM_GPU bm;
    StereoBeliefPropagation bp;
    StereoConstantSpaceBP csbp;

    vector< Ptr<PairFrameSource> > stereoSources;
    int curSource;
};

App::App()
{
    method = BM;
    colorOutput = false;
    showHelp = false;
    showSource = false;

    bm.ndisp = 80;

    bp.ndisp = bm.ndisp;
    bp.iters = 5;
    bp.levels = 5;

    csbp.ndisp = bm.ndisp;
    csbp.iters = bp.iters;
    csbp.levels = bp.levels;

    curSource = 0;
}

void App::process()
{
    if (!sources.empty() && sources.size() % 2 == 0)
    {
        for (size_t i = 0; i < sources.size(); i += 2)
            stereoSources.push_back(PairFrameSource::get(sources[i], sources[i+1]));
    }
    else
    {
        cout << "Loading default frames source..." << endl;

        stereoSources.push_back(PairFrameSource::get(new ImageSource("data/aloeL_small.png"),
                                                     new ImageSource("data/aloeR_small.png")));
        stereoSources.push_back(PairFrameSource::get(new ImageSource("data/babyL_small.png"),
                                                     new ImageSource("data/babyR_small.png")));
        stereoSources.push_back(PairFrameSource::get(new ImageSource("data/conesL_small.png"),
                                                     new ImageSource("data/conesR_small.png")));
        stereoSources.push_back(PairFrameSource::get(new ImageSource("data/teddyL_small.png"),
                                                     new ImageSource("data/teddyR_small.png")));
    }

    cout << "ndisp: " << bm.ndisp << endl;
    cout << "winSize: " << bm.winSize << endl;
    cout << "iters: " << bp.iters << endl;
    cout << "levels: " << bp.levels << endl;
    cout << endl;

    Mat left_src, right_src;
    Mat left, right;
    GpuMat d_left, d_right;
    Mat small_image;

    GpuMat d_disp, d_img_to_show;

    Mat img_to_show, img_to_show_scaled;

    namedWindow("Stereo Matching Demo", WINDOW_NORMAL);
    setWindowProperty("Stereo Matching Demo", WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

    while (!exited)
    {
        int64 start = getTickCount();

        stereoSources[curSource]->next(left_src, right_src);

        makeGray(left_src, left);
        makeGray(right_src, right);

        d_left.upload(left);
        d_right.upload(right);

        int64 proc_start = getTickCount();
        d_disp.create(d_left.size(), CV_8U);
        switch (method)
        {
        case BM: bm(d_left, d_right, d_disp); break;
        case BP: bp(d_left, d_right, d_disp); break;
        case CSBP: csbp(d_left, d_right, d_disp); break;
        }
        double proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

        if (colorOutput)
        {
            drawColorDisp(d_disp, d_img_to_show, bm.ndisp);
        }
        else
        {
            d_disp.convertTo(d_disp, d_disp.depth(), 255.0 / bm.ndisp);
            cvtColor(d_disp, d_img_to_show, COLOR_GRAY2BGR);
        }
        d_img_to_show.download(img_to_show);

        if (showSource)
        {
            resize(left_src, small_image, cv::Size(), 0.25, 0.25);
            Mat roi = img_to_show(cv::Rect(img_to_show.cols - small_image.cols, 0, small_image.cols, small_image.rows));

            if (colorOutput)
                cvtColor(small_image, roi, cv::COLOR_BGR2BGRA);
            else
                small_image.copyTo(roi);
        }

        double total_fps = getTickFrequency()  / (getTickCount() - start);

        resize(img_to_show, img_to_show_scaled, Size(), 2, 2);
        displayState(img_to_show_scaled, proc_fps, total_fps);

        imshow("Stereo Matching Demo", img_to_show_scaled);

        processKey(waitKey(30));
    }
}

void App::displayState(cv::Mat& frame, double proc_fps, double total_fps)
{
    const Scalar fontColor = CV_RGB(118, 185, 0);
    const Scalar fontColorRed = CV_RGB(255, 0, 0);
    const double fontScale = 0.6;

    int i = 0;

    ostringstream txt;
    txt.str(""); txt << "Source size: " << frame.cols << 'x' << frame.rows;
    printText(frame, txt.str(), i++, fontColor, fontScale);

    txt.str(""); txt << "Method: " << method_str[method];
    printText(frame, txt.str(), i++, fontColor, fontScale);

    txt.str(""); txt << "FPS (Stereo only): " << fixed << setprecision(1) << proc_fps;
    printText(frame, txt.str(), i++, fontColor, fontScale);

    txt.str(""); txt << "FPS (total): " << fixed << setprecision(1) << total_fps;
    printText(frame, txt.str(), i++, fontColor, fontScale);

    if (!showHelp)
    {
        printText(frame, "H - toggle hotkeys help", i++, fontColorRed, fontScale);
    }
    else
    {
        printText(frame, "Space - switch method", i++, fontColorRed, fontScale);
        printText(frame, "1/Q - increase/decrease disparities number", i++, fontColorRed, fontScale);
        printText(frame, "2/W - increase/decrease window size", i++, fontColorRed, fontScale);
        printText(frame, "3/E - increase/decrease iterations count", i++, fontColorRed, fontScale);
        printText(frame, "4/R - increase/decrease levels count", i++, fontColorRed, fontScale);
        printText(frame, "C - switch color/gray output", i++, fontColorRed, fontScale);
        printText(frame, "S - show/hide source frame", i++, fontColorRed, fontScale);
        printText(frame, "N - next source", i++, fontColorRed, fontScale);
    }
}

bool App::processKey(int key)
{
    if (BaseApp::processKey(key))
        return true;

    switch (toupper(key & 0xff))
    {
    case 32:
        switch (method)
        {
        case BM:
            method = BP;
            break;
        case BP:
            method = CSBP;
            break;
        case CSBP:
            method = BM;
            break;
        }
        cout << "method: " << method_str[method] << endl;
        break;

    case 'H':
        showHelp = !showHelp;
        break;

    case '1':
        bm.ndisp = min(bm.ndisp + 16, 256);
        cout << "ndisp: " << bm.ndisp << endl;
        bp.ndisp = bm.ndisp;
        csbp.ndisp = bm.ndisp;
        break;

    case 'Q':
        bm.ndisp = max(bm.ndisp - 16, 16);
        cout << "ndisp: " << bm.ndisp << endl;
        bp.ndisp = bm.ndisp;
        csbp.ndisp = bm.ndisp;
        break;

    case '2':
        if (method == BM)
        {
            bm.winSize = min(bm.winSize + 2, 51);
            cout << "win_size: " << bm.winSize << endl;
        }
        break;

    case 'W':
        if (method == BM)
        {
            bm.winSize = max(bm.winSize - 2, 5);
            cout << "win_size: " << bm.winSize << endl;
        }
        break;

    case '3':
        if (method == BP || method == CSBP)
        {
            bp.iters += 1;
            csbp.iters = bp.iters;
            cout << "iters: " << bp.iters << endl;
        }
        break;

    case 'E':
        if (method == BP || method == CSBP)
        {
            bp.iters = max(bp.iters - 1, 1);
            csbp.iters = bp.iters;
            cout << "iters: " << bp.iters << endl;
        }
        break;

    case '4':
        if (method == BP || method == CSBP)
        {
            bp.levels = min(bp.levels + 1, 8);
            csbp.levels = bp.levels;
            cout << "levels: " << bp.levels << endl;
        }
        break;

    case 'R':
        if (method == BP || method == CSBP)
        {
            bp.levels = max(bp.levels - 1, 1);
            csbp.levels = bp.levels;
            cout << "levels: " << bp.levels << endl;
        }
        break;

    case 'C':
        colorOutput = !colorOutput;
        cout << (colorOutput ? "Color output" : "Gray output") << endl;
        break;

    case 'S':
        showSource = !showSource;
        cout << (showSource ? "Show source" : "Hide source") << endl;
        break;

    case 'N':
        curSource = (curSource + 1) % stereoSources.size();
        stereoSources[curSource]->reset();
        break;

    default:
        return false;
    }

    return true;
}

void App::printHelp()
{
    cout << "This sample demonstrates different Stereo Matching algorithms" << endl;
    cout << "Usage: demo_stereo_matching [options]" << endl;
    cout << "Options:" << endl;
    BaseApp::printHelp();
}

RUN_APP(App)

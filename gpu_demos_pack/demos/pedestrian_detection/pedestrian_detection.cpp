#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

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
    bool parseCmdArgs(int& i, int argc, const char* argv[]);
    bool processKey(int key);
    void printHelp();

private:
    void displayState(Mat& frame, double proc_fps, double total_fps);

    bool useGPU;
    bool graySource;

    double scale;
    int nlevels;
    int gr_threshold;

    double hit_threshold;
    bool hit_threshold_auto;

    int win_width;
    int win_stride_width, win_stride_height;

    bool gamma_corr;

    int curSource;
    bool showHelp;
};

App::App()
{
    useGPU = true;
    graySource = false;

    scale = 1.05;
    nlevels = 13;
    gr_threshold = 8;
    hit_threshold = 1.4;
    hit_threshold_auto = true;

    win_width = 48;
    win_stride_width = 8;
    win_stride_height = 8;

    gamma_corr = false;

    curSource = 0;
    showHelp = false;
}

void App::process()
{
    if (sources.empty())
    {
        cout << "Loading default frames source..." << endl;
        sources.push_back(new VideoSource("data/pedestrian_detection.avi"));
    }

    if (hit_threshold_auto)
        hit_threshold = win_width == 48 ? 1.4 : 0.;

    if (win_width != 64 && win_width != 48)
    {
        cout << "Using default windows width (64)...\n";
        win_width = 64;
    }

    cout << "Scale: " << scale << endl;
    cout << "Group threshold: " << gr_threshold << endl;
    cout << "Levels number: " << nlevels << endl;
    cout << "Win width: " << win_width << endl;
    cout << "Win stride: (" << win_stride_width << ", " << win_stride_height << ")" << endl;
    cout << "Hit threshold: " << hit_threshold << endl;
    cout << "Gamma correction: " << gamma_corr << endl;
    cout << endl;

    Size win_size(win_width, win_width * 2); //(64, 128) or (48, 96)
    Size win_stride(win_stride_width, win_stride_height);

    // Create HOG descriptors and detectors here
    vector<float> detector;
    if (win_size == Size(64, 128))
        detector = gpu::HOGDescriptor::getPeopleDetector64x128();
    else
        detector = gpu::HOGDescriptor::getPeopleDetector48x96();

    gpu::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9,
                               gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, gamma_corr,
                               gpu::HOGDescriptor::DEFAULT_NLEVELS);
    cv::HOGDescriptor cpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1,
                              cv::HOGDescriptor::L2Hys, 0.2, gamma_corr,
                              cv::HOGDescriptor::DEFAULT_NLEVELS);

    gpu_hog.setSVMDetector(detector);
    cpu_hog.setSVMDetector(detector);

    Mat frame;
    Mat img, img_to_show;
    GpuMat gpu_img;

    // Iterate over all frames
    while (!exited)
    {
        int64 start = getTickCount();

        sources[curSource]->next(frame);

        // Change format of the image
        if (graySource)
            cvtColor(frame, img, CV_BGR2GRAY);
        else if (useGPU)
            cvtColor(frame, img, CV_BGR2BGRA);
        else
            frame.copyTo(img);

        // Display image
        if (graySource)
            cvtColor(img, img_to_show, CV_GRAY2BGR);
        else if (useGPU)
            cvtColor(img, img_to_show, CV_BGRA2BGR);
        else
            img_to_show = img;

        gpu_hog.nlevels = nlevels;
        cpu_hog.nlevels = nlevels;

        vector<Rect> found;

        // Perform HOG classification
        int64 proc_start = getTickCount();
        if (useGPU)
        {
            gpu_img.upload(img);
            gpu_hog.detectMultiScale(gpu_img, found, hit_threshold, win_stride, Size(0, 0), scale, gr_threshold);
        }
        else
        {
            cpu_hog.detectMultiScale(img, found, hit_threshold, win_stride, Size(0, 0), scale, gr_threshold);
        }
        double proc_fps = getTickFrequency() / (getTickCount() - proc_start);

        // Draw positive classified windows
        for (size_t i = 0; i < found.size(); i++)
            rectangle(img_to_show, found[i], CV_RGB(0, 255, 0), 3);

        double total_fps = getTickFrequency() / (getTickCount() - start);

        displayState(img_to_show, proc_fps, total_fps);

        imshow("Pedestrian Detection Demo", img_to_show);

        processKey(waitKey(3));
    }
}

void App::displayState(Mat& frame, double proc_fps, double total_fps)
{
    const Scalar fontColorRed = CV_RGB(255, 0, 0);

    int i = 0;

    ostringstream txt;
    txt.str(""); txt << "Source size: " << frame.cols << 'x' << frame.rows;
    printText(frame, txt.str(), i++);

    printText(frame, useGPU ? "Mode: CUDA" : "Mode: CPU", i++);

    txt.str(""); txt << "FPS (HOG only): " << fixed << setprecision(1) << proc_fps;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "FPS (total): " << fixed << setprecision(1) << total_fps;
    printText(frame, txt.str(), i++);

    if (!showHelp)
    {
        printText(frame, "H - toggle hotkeys help", i++, fontColorRed);
    }
    else
    {
        printText(frame, "Space - switch GPU / CPU", i++, fontColorRed);
        printText(frame, "1/Q - increase/decrease HOG scale", i++, fontColorRed);
        printText(frame, "2/W - increase/decrease levels count", i++, fontColorRed);
        printText(frame, "3/E - increase/decrease HOG group threshold", i++, fontColorRed);
        printText(frame, "4/R - increase/decrease hit threshold", i++, fontColorRed);
        printText(frame, "G - switch gray / color", i++, fontColorRed);
        printText(frame, "N - next source", i++, fontColorRed);
    }
}

bool App::processKey(int key)
{
    if (BaseApp::processKey(key))
        return true;

    switch (toupper(key & 0xff))
    {
    case 32:
        useGPU = !useGPU;
        cout << "Switched to " << (useGPU ? "CUDA" : "CPU") << " mode\n";
        break;

    case 'H':
        showHelp = !showHelp;
        break;

    case 'G':
        graySource = !graySource;
        cout << "Convert image to gray: " << (graySource ? "YES" : "NO") << endl;
        break;

    case '1':
        scale *= 1.05;
        cout << "Scale: " << scale << endl;
        break;

    case 'Q':
        scale /= 1.05;
        cout << "Scale: " << scale << endl;
        break;

    case '2':
        nlevels++;
        cout << "Levels number: " << nlevels << endl;
        break;

    case 'W':
        nlevels = max(nlevels - 1, 1);
        cout << "Levels number: " << nlevels << endl;
        break;

    case '3':
        gr_threshold++;
        cout << "Group threshold: " << gr_threshold << endl;
        break;

    case 'E':
        gr_threshold = max(0, gr_threshold - 1);
        cout << "Group threshold: " << gr_threshold << endl;
        break;

    case '4':
        hit_threshold+=0.25;
        cout << "Hit threshold: " << hit_threshold << endl;
        break;

    case 'R':
        hit_threshold = max(0.0, hit_threshold - 0.25);
        cout << "Hit threshold: " << hit_threshold << endl;
        break;

    case 'C':
        gamma_corr = !gamma_corr;
        cout << "Gamma correction: " << gamma_corr << endl;
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

bool App::parseCmdArgs(int& i, int argc, const char* argv[])
{
    string arg = argv[i];

    if (arg == "--make-gray")
    {
        graySource = true;
    }
    else if (arg == "--hit-threshold")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        hit_threshold = atof(argv[i]);
        hit_threshold_auto = false;
    }
    else if (arg == "--scale")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        scale = atof(argv[i]);
    }
    else if (arg == "--nlevels")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        nlevels = atoi(argv[i]);
    }
    else if (arg == "--win-width")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        win_width = atoi(argv[i]);
    }
    else if (arg == "--win-stride-width")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        win_stride_width = atoi(argv[i]);
    }
    else if (arg == "--win-stride-height")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        win_stride_height = atoi(argv[i]);
    }
    else if (arg == "--gr-threshold")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing value after " << arg);

        gr_threshold = atoi(argv[i]);
    }
    else if (arg == "--gamma-correct")
    {
        gamma_corr = true;
    }
    else
        return false;

    return true;
}

void App::printHelp()
{
    cout << "This sample demonstrates Pedestrian Detection algorithm" << endl;
    cout << "Usage: demo_pedestrian_detection <frames_source>" << endl
         << "  [--make-gray] # convert image to gray one" << endl
         << "  [--resize-src] # do resize of the source image" << endl
         << "  [--hit-threshold <double>] # classifying plane distance threshold (0.0 usually)" << endl
         << "  [--scale <double>] # HOG window scale factor" << endl
         << "  [--nlevels <int>] # max number of HOG window scales" << endl
         << "  [--win-width <int>] # width of the window (48 or 64)" << endl
         << "  [--win-stride-width <int>] # distance by OX axis between neighbour wins" << endl
         << "  [--win-stride-height <int>] # distance by OY axis between neighbour wins" << endl
         << "  [--gr-threshold <int>] # merging similar rects constant" << endl
         << "  [--gamma-correct <int>] # do gamma correction or not" << endl;
    BaseApp::printHelp();
}

RUN_APP(App)

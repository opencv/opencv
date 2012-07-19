#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "utility_lib/utility_lib.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

class App : public BaseApp
{
public:
    App();

protected:
    void process();
    void printHelp();
    bool processKey(int key);

private:
    void displayState(cv::Mat& frame, double proc_fps, double total_fps);

    bool use_gpu;
    bool colorOutput;
    bool showHelp;
    bool showSource;
    bool videoSource;

    StereoBM     bm_cpu;
    StereoBM_GPU bm_gpu;

    std::vector< cv::Ptr<PairFrameSource> > stereoSources;
    size_t curSource;
};

App::App()
{
    use_gpu = true;
    colorOutput = false;
    showHelp = false;
    showSource = false;
    videoSource = true;

    bm_gpu.ndisp = 256;
    bm_cpu.init(StereoBM::BASIC_PRESET, bm_gpu.ndisp, bm_gpu.winSize);

    curSource = 0;
}

void App::process()
{
    if ((sources.size() > 0) && (sources.size() % 2 == 0))
    {
        for (size_t i = 0; i < sources.size(); i += 2)
            stereoSources.push_back(PairFrameSource::get(sources[i], sources[i+1]));
    }
    else
    {
        cout << "Loading default frames source...\n";

        stereoSources.push_back(PairFrameSource::get(new VideoSource("data/stereo_matching/8sec_Toys_Kirill_1920x1080_xvid_L.avi"),
                                                     new VideoSource("data/stereo_matching/8sec_Toys_Kirill_1920x1080_xvid_R.avi")));
        stereoSources.push_back(PairFrameSource::get(new ImageSource("data/stereo_matching/aloeL.png"),
                                                     new ImageSource("data/stereo_matching/aloeR.png")));
        stereoSources.push_back(PairFrameSource::get(new ImageSource("data/stereo_matching/babyL.png"),
                                                     new ImageSource("data/stereo_matching/babyR.png")));
        stereoSources.push_back(PairFrameSource::get(new ImageSource("data/stereo_matching/conesL.png"),
                                                     new ImageSource("data/stereo_matching/conesR.png")));
        stereoSources.push_back(PairFrameSource::get(new ImageSource("data/stereo_matching/teddyL.png"),
                                                     new ImageSource("data/stereo_matching/teddyR.png")));
    }

    cout << "\nControls:\n"
         << "\tESC - exit\n"
         << "\tSpace - switch GPU / CPU\n"
         << "\t1/Q - increase/decrease disparities number\n"
         << "\t2/W - increase/decrease window size\n"
         << "\tC - switch color/gray output\n"
         << "\tI - switch source\n"
         << "\tS - show/hide source frame\n"
         << endl;

    cout << "ndisp: " << bm_gpu.ndisp << endl;
    cout << "winSize: " << bm_gpu.winSize << endl;
    cout << endl;

    Mat left_src, right_src;
    Mat left, right;
    GpuMat d_left, d_right;
    Mat small_image;

    Mat disp, disp_16s;
    GpuMat d_disp, d_img_to_show;

    Mat img_to_show;

    cv::namedWindow("bm_stereo_matching_demo", cv::WINDOW_NORMAL);
    cv::setWindowProperty("bm_stereo_matching_demo", cv::WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

    while (!exited)
    {
        int64 start = getTickCount();

        stereoSources[curSource]->next(left_src, right_src);

        makeGray(left_src, left);
        makeGray(right_src, right);

        d_left.upload(left);
        d_right.upload(right);

        int64 proc_start = getTickCount();
        if (use_gpu)
            bm_gpu(d_left, d_right, d_disp);
        else
        {
            bm_cpu(left, right, disp_16s);
        }
        double proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

        if (colorOutput)
        {
            if (!use_gpu)
            {
                disp_16s.convertTo(disp, CV_8U, 1.0 / 16.0);
                d_disp.upload(disp);
            }

            cv::gpu::drawColorDisp(d_disp, d_img_to_show, bm_gpu.ndisp);
            d_img_to_show.download(img_to_show);
        }
        else
        {
            if (!use_gpu)
            {
                disp_16s.convertTo(disp, CV_8U, 255.0 / (16.0 * bm_gpu.ndisp));
                cvtColor(disp, img_to_show, COLOR_GRAY2BGR);
            }
            else
            {
                d_disp.convertTo(d_disp, d_disp.depth(), 255.0 / bm_gpu.ndisp);
                cv::gpu::cvtColor(d_disp, d_img_to_show, COLOR_GRAY2BGR);
                d_img_to_show.download(img_to_show);
            }
        }

        if (showSource)
        {
            resize(left_src, small_image, cv::Size(), 0.25, 0.25);
            cv::Mat roi = img_to_show(cv::Rect(img_to_show.cols - small_image.cols, 0, small_image.cols, small_image.rows));

            if (colorOutput)
                cv::cvtColor(small_image, roi, cv::COLOR_BGR2BGRA);
            else
                small_image.copyTo(roi);
        }

        double total_fps = getTickFrequency()  / (getTickCount() - start);

        displayState(img_to_show, proc_fps, total_fps);

        imshow("bm_stereo_matching_demo", img_to_show);

        processKey(waitKey(3) & 0xff);
    }
}

void App::displayState(cv::Mat& frame, double proc_fps, double total_fps)
{
    const Scalar fontColorRed = CV_RGB(255, 0, 0);

    int i = 0;

    ostringstream txt;
    txt.str(""); txt << "Source size: " << frame.cols << 'x' << frame.rows;
    printText(frame, txt.str(), i++);

    printText(frame, use_gpu ? "Mode: CUDA" : "Mode: CPU", i++);

    txt.str(""); txt << "FPS (Stereo only): " << fixed << setprecision(1) << proc_fps;
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
        printText(frame, "1/Q - increase/decrease disparities number", i++, fontColorRed);
        printText(frame, "2/W - increase/decrease window size", i++, fontColorRed);
        printText(frame, "C - switch color/gray output", i++, fontColorRed);
        printText(frame, "I - switch source", i++, fontColorRed);
        printText(frame, "S - show/hide source frame", i++, fontColorRed);
    }
}

void App::printHelp()
{
    cout << "Usage: demo_bm_stereo_matching <left_frame_source> <right_frame_source>" << endl;
    BaseApp::printHelp();
}

bool App::processKey(int key)
{
    if (BaseApp::processKey(key))
        return true;

    switch (toupper(key))
    {
    case 32:
        use_gpu = !use_gpu;
        cout << "Switched to " << (use_gpu ? "CUDA" : "CPU") << " mode\n";
        break;

    case 'H':
        showHelp = !showHelp;
        break;

    case '1':
        bm_gpu.ndisp = min(bm_gpu.ndisp + 16, 256);
        cout << "ndisp: " << bm_gpu.ndisp << endl;
        bm_cpu.init(StereoBM::BASIC_PRESET, bm_gpu.ndisp, bm_gpu.winSize);
        break;

    case 'Q':
        bm_gpu.ndisp = max(bm_gpu.ndisp - 16, 16);
        cout << "ndisp: " << bm_gpu.ndisp << endl;
        bm_cpu.init(StereoBM::BASIC_PRESET, bm_gpu.ndisp, bm_gpu.winSize);
        break;

    case '2':
        bm_gpu.winSize = min(bm_gpu.winSize + 2, 51);
        bm_cpu.init(StereoBM::BASIC_PRESET, bm_gpu.ndisp, bm_gpu.winSize);
        cout << "win_size: " << bm_gpu.winSize << endl;
        break;

    case 'W':
        bm_gpu.winSize = max(bm_gpu.winSize - 2, 5);
        bm_cpu.init(StereoBM::BASIC_PRESET, bm_gpu.ndisp, bm_gpu.winSize);
        cout << "win_size: " << bm_gpu.winSize << endl;
        break;

    case 'C':
        colorOutput = !colorOutput;
        cout << (colorOutput ? "Color output" : "Gray output") << endl;
        break;

    case 'S':
        showSource = !showSource;
        cout << (showSource ? "Show source" : "Hide source") << endl;
        break;

    case 'I':
        curSource = (curSource + 1) % stereoSources.size();
        break;

    default:
        return false;
    }

    return true;
}

RUN_APP(App)

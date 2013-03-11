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

enum Method
{
    HAAR,
    LBP,
    METHOD_MAX
};

const char* method_str[] =
{
    "HAAR",
    "LBP"
};

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
    void displayState(cv::Mat& outImg, double proc_fps, double total_fps);

    string haarCascadeName_;
    string lbpCascadeName_;

    Method method_;
    bool useGpu_;
    int curSource_;
    bool fullscreen_;
    bool reloadCascade_;
};

App::App()
{
    haarCascadeName_ = "data/face_detection_haar.xml";
    lbpCascadeName_ = "data/face_detection_lbp.xml";

    method_ = HAAR;
    useGpu_ = true;
    curSource_ = 0;
    fullscreen_ = false;
    reloadCascade_ = true;
}

void App::runAppLogic()
{
    if (sources_.empty())
    {
        cout << "Using default frames source... \n" << endl;
        sources_.push_back(FrameSource::video("data/face_detection.avi"));
    }

    CascadeClassifier cascade_cpu;
    CascadeClassifier_GPU cascade_gpu;

    Mat frame_cpu, gray_cpu, outImg;
    GpuMat frame_gpu, gray_gpu, facesBuf_gpu;

    vector<Rect> faces;

    const string wndName = "Face Detection Demo";

    if (fullscreen_)
    {
        namedWindow(wndName, WINDOW_NORMAL);
        setWindowProperty(wndName, WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        setWindowProperty(wndName, WND_PROP_ASPECT_RATIO, CV_WINDOW_FREERATIO);
    }

    while (isActive())
    {
        if (reloadCascade_)
        {
            const string& cascadeName = method_ == HAAR ? haarCascadeName_ : lbpCascadeName_;
            cascade_gpu.load(cascadeName);
            cascade_cpu.load(cascadeName);
            reloadCascade_ = false;
        }

        const int64 total_start = getTickCount();

        sources_[curSource_]->next(frame_cpu);

        double proc_fps = 0.0;

        if (useGpu_)
        {
            frame_gpu.upload(frame_cpu);
            makeGray(frame_gpu, gray_gpu);

            cascade_gpu.visualizeInPlace = false;
            cascade_gpu.findLargestObject = false;

            const int64 proc_start = getTickCount();

            const int detections_num = cascade_gpu.detectMultiScale(gray_gpu, facesBuf_gpu, 1.2, 4);

            proc_fps = getTickFrequency() / (getTickCount() - proc_start);

            if (detections_num == 0)
                faces.clear();
            else
            {
                faces.resize(detections_num);
                Mat facesMat(1, detections_num, DataType<Rect>::type, &faces[0]);
                facesBuf_gpu.colRange(0, detections_num).download(facesMat);
            }
        }
        else
        {
            makeGray(frame_cpu, gray_cpu);

            const Size minSize = cascade_gpu.getClassifierSize();

            const int64 proc_start = getTickCount();

            cascade_cpu.detectMultiScale(gray_cpu, faces, 1.2, 4, CV_HAAR_SCALE_IMAGE, minSize);

            proc_fps = getTickFrequency() / (getTickCount() - proc_start);
        }

        frame_cpu.copyTo(outImg);

        for (size_t i = 0; i < faces.size(); i++)
            rectangle(outImg, faces[i], CV_RGB(0, 255, 0), 3);

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

    txt.str(""); txt << "Method: " << method_str[method_] << (useGpu_ ? " CUDA" : " CPU");
    printText(outImg, txt.str(), i++);

    txt.str(""); txt << "FPS (FD only): " << fixed << setprecision(1) << proc_fps;
    printText(outImg, txt.str(), i++);

    txt.str(""); txt << "FPS (total): " << fixed << setprecision(1) << total_fps;
    printText(outImg, txt.str(), i++);

    printText(outImg, "Space - switch CUDA / CPU mode", i++, fontColorRed);
    printText(outImg, "M - switch method", i++, fontColorRed);
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

    case 'M':
        method_ = static_cast<Method>((method_ + 1) % METHOD_MAX);
        reloadCascade_ = true;
        cout << "Switch method to " << method_str[method_] << endl;
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
    cout << "This sample demonstrates different Face Detection algorithms \n" << endl;

    cout << "Usage: demo_face_detection [options] \n" << endl;

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

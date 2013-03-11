#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <opencv2/video/video.hpp>
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

    vector< Ptr<PairFrameSource> > pairSources_;

    bool useGpu_;
    int curSource_;
    bool fullscreen_;
};

App::App()
{
    useGpu_ = true;
    curSource_ = 0;
    fullscreen_ = true;
}

static void download(const GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void download(const GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}

static void drawArrows(Mat& frame, const vector<Point2f>& prevPts, const vector<Point2f>& nextPts, const vector<uchar>& status, Scalar line_color = Scalar(0, 0, 255))
{
    for (size_t i = 0; i < prevPts.size(); ++i)
    {
        if (status[i])
        {
            int line_thickness = 1;

            Point p = prevPts[i];
            Point q = nextPts[i];

            double angle = atan2((double) p.y - q.y, (double) p.x - q.x);

            double hypotenuse = sqrt( (double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x) );

            if (hypotenuse < 3.0 || hypotenuse > 50.0)
                continue;

            // Here we lengthen the arrow by a factor of three.
            q.x = (int) (p.x - hypotenuse * cos(angle));
            q.y = (int) (p.y - hypotenuse * sin(angle));

            // Now we draw the main line of the arrow.
            line(frame, p, q, line_color, line_thickness);

            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.

            double tips_length = 9.0 * hypotenuse / 50.0 + 5.0;

            p.x = (int) (q.x + tips_length * cos(angle + CV_PI / 6));
            p.y = (int) (q.y + tips_length * sin(angle + CV_PI / 6));
            line(frame, p, q, line_color, line_thickness);

            p.x = (int) (q.x + tips_length * cos(angle - CV_PI / 6));
            p.y = (int) (q.y + tips_length * sin(angle - CV_PI / 6));
            line(frame, p, q, line_color, line_thickness);
        }
    }
}

void App::runAppLogic()
{
    if (sources_.empty())
    {
        cout << "Loading default frames source... \n" << endl;

        sources_.push_back(FrameSource::video("data/sparse_optical_flow.avi"));
    }

    for (size_t i = 0; i < sources_.size(); ++i)
        pairSources_.push_back(PairFrameSource::create(sources_[i], 2));

    GoodFeaturesToTrackDetector_GPU detector(8000, 0.01, 10.0);
    PyrLKOpticalFlow lk;

    Mat frame0, frame1;
    Mat gray;
    GpuMat d_frame0, d_frame1;
    GpuMat d_gray;

    GpuMat d_prevPts;
    GpuMat d_nextPts;
    GpuMat d_status;

    vector<Point2f> prevPts;
    vector<Point2f> nextPts;
    vector<uchar> status;

    Mat outImg;

    const string wndName = "Sparse Optical Flow Demo";

    if (fullscreen_)
    {
        namedWindow(wndName, WINDOW_NORMAL);
        setWindowProperty(wndName, WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        setWindowProperty(wndName, WND_PROP_ASPECT_RATIO, CV_WINDOW_FREERATIO);
    }

    while (isActive())
    {
        const int64 total_start = getTickCount();

        pairSources_[curSource_]->next(frame0, frame1);

        double proc_fps = 0.0;

        if (useGpu_)
        {
            d_frame0.upload(frame0);
            d_frame1.upload(frame1);

            cvtColor(d_frame0, d_gray, COLOR_BGR2GRAY);

            const int64 proc_start = getTickCount();

            detector(d_gray, d_prevPts);
            lk.sparse(d_frame0, d_frame1, d_prevPts, d_nextPts, d_status);

            proc_fps = getTickFrequency()  / (getTickCount() - proc_start);

            download(d_prevPts, prevPts);
            download(d_nextPts, nextPts);
            download(d_status, status);
        }
        else
        {
            cvtColor(frame0, gray, COLOR_BGR2GRAY);

            const int64 proc_start = getTickCount();

            goodFeaturesToTrack(gray, prevPts, detector.maxCorners, detector.qualityLevel, detector.minDistance);
            calcOpticalFlowPyrLK(frame0, frame1, prevPts, nextPts, status, noArray());

            proc_fps = getTickFrequency()  / (getTickCount() - proc_start);
        }

        frame0.copyTo(outImg);
        drawArrows(outImg, prevPts, nextPts, status, Scalar(255, 0, 0));

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

    txt.str(""); txt << "Source size: " << outImg.cols << 'x' << outImg.rows;
    printText(outImg, txt.str(), i++);

    printText(outImg, useGpu_ ? "Mode: CUDA" : "Mode: CPU", i++);

    txt.str(""); txt << "FPS (OptFlow only): " << fixed << setprecision(1) << proc_fps;
    printText(outImg, txt.str(), i++);

    txt.str(""); txt << "FPS (total): " << fixed << setprecision(1) << total_fps;
    printText(outImg, txt.str(), i++);

    printText(outImg, "Space - switch CUDA / CPU mode", i++, fontColorRed);
    if (pairSources_.size() > 1)
        printText(outImg, "N - next source", i++, fontColorRed);
}

void App::processAppKey(int key)
{
    switch (toupper(key & 0xff))
    {
    case 32 /*space*/:
        useGpu_ = !useGpu_;
        cout << "Switch mode to " << (useGpu_ ? "CUDA" : "CPU") << endl;
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
    cout << "This sample demonstrates different Sparse Optical Flow algorithms \n" << endl;

    cout << "Usage: demo_sparse_optical_flow [options] \n" << endl;

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

#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <opencv2/video/video.hpp>
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
    bool processKey(int key);
    void printHelp();

private:
    void displayState(Mat& frame, double proc_fps, double total_fps);

    bool useGPU;

    vector< Ptr<PairFrameSource> > pairSources;
    int curSource;
};

App::App()
{
    useGPU = true;

    curSource = 0;
}

void download(const GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

void download(const GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}

void drawArrows(Mat& frame, const vector<Point2f>& prevPts, const vector<Point2f>& nextPts, const vector<uchar>& status, Scalar line_color = Scalar(0, 0, 255))
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

            double tips_length = 9.0 * hypotenuse / 50.0 + 2.0;

            p.x = (int) (q.x + tips_length * cos(angle + CV_PI / 6));
            p.y = (int) (q.y + tips_length * sin(angle + CV_PI / 6));
            line(frame, p, q, line_color, line_thickness);

            p.x = (int) (q.x + tips_length * cos(angle - CV_PI / 6));
            p.y = (int) (q.y + tips_length * sin(angle - CV_PI / 6));
            line(frame, p, q, line_color, line_thickness);
        }
    }
}

void App::process()
{
    if (sources.empty())
    {
        cout << "Loading default frames source..." << endl;

        sources.push_back(new VideoSource("data/pedestrian_detect/mitsubishi.avi"));
        sources.push_back(new VideoSource("data/optical_flow/plush1_720p_10s.m2v"));
        sources.push_back(new VideoSource("data/stereo_matching/8sec_Toys_Kirill_1920x1080_xvid_L.avi"));
    }

    for (size_t i = 0; i < sources.size(); ++i)
        pairSources.push_back(PairFrameSource::get(sources[i], 2));

    Mat frame0, frame1;
    Mat gray;
    GpuMat d_frame0, d_frame1;
    GpuMat d_gray;

    Mat u, v;
    GpuMat d_u, d_v;

    GpuMat d_prevPts;
    GpuMat d_nextPts;
    GpuMat d_status;

    vector<Point2f> prevPts;
    vector<Point2f> nextPts;
    vector<uchar> status;

    GoodFeaturesToTrackDetector_GPU detector(4000, 0.01, 10.0);
    PyrLKOpticalFlow lk;

    namedWindow("Sparse Optical Flow Demo", WINDOW_NORMAL);
    setWindowProperty("Sparse Optical Flow Demo", WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

    while (!exited)
    {
        int64 start = getTickCount();

        pairSources[curSource]->next(frame0, frame1);

        d_frame0.upload(frame0);
        d_frame1.upload(frame1);

        double proc_fps;

        if (useGPU)
        {
            cvtColor(d_frame0, d_gray, COLOR_BGR2GRAY);

            int64 proc_start = getTickCount();

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

            int64 proc_start = getTickCount();

            goodFeaturesToTrack(gray, prevPts, detector.maxCorners, detector.qualityLevel, detector.minDistance);
            calcOpticalFlowPyrLK(frame0, frame1, prevPts, nextPts, status, noArray());

            proc_fps = getTickFrequency()  / (getTickCount() - proc_start);
        }

        Mat img_to_show = frame0.clone();

        drawArrows(img_to_show, prevPts, nextPts, status, Scalar(255, 0, 0));

        double total_fps = getTickFrequency()  / (getTickCount() - start);

        displayState(img_to_show, proc_fps, total_fps);

        imshow("Sparse Optical Flow Demo", img_to_show);

        processKey(waitKey(30) & 0xff);
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

    txt.str(""); txt << "FPS (OptFlow only): " << fixed << setprecision(1) << proc_fps;
    printText(frame, txt.str(), i++);

    txt.str(""); txt << "FPS (total): " << fixed << setprecision(1) << total_fps;
    printText(frame, txt.str(), i++);

    printText(frame, "Space - switch GPU / CPU", i++, fontColorRed);
    printText(frame, "N - next source", i++, fontColorRed);
}

bool App::processKey(int key)
{
    if (BaseApp::processKey(key))
        return true;

    switch (toupper(key & 0xff))
    {
    case 32 /*space*/:
        useGPU = !useGPU;
        cout << "Switched to " << (useGPU ? "CUDA" : "CPU") << " mode\n";
        break;

    case 'N':
        curSource = (curSource + 1) % pairSources.size();
        pairSources[curSource]->reset();
        break;
    }

    return false;
}

void App::printHelp()
{
    cout << "This sample demonstrates different Sparse Optical Flow algorithms" << endl;
    cout << "Usage: demo_sparse_optical_flow [options]" << endl;
    cout << "Options:" << endl;
    BaseApp::printHelp();
}

RUN_APP(App)

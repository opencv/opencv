#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "utility.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

struct Object
{
    string name;

    Mat imgColor;
    Mat imgGray;
    GpuMat d_imgGray;

    Scalar color;

    vector<KeyPoint> keypoints;

    Mat descriptors;
    GpuMat d_descriptors;

    Object(const string& name_, const Mat& imgColor_, Scalar color_) :
        name(name_), imgColor(imgColor_), color(color_)
    {
        cvtColor(imgColor, imgGray, COLOR_BGR2GRAY);
        d_imgGray.upload(imgGray);
    }
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
    void calcKeypoints(const Mat& img, const GpuMat& d_img, vector<KeyPoint>& keypoints, Mat& descriptors, GpuMat& d_descriptors);

    void match(const Mat& descriptors1, const GpuMat& d_descriptors1,
               const Mat& descriptors2, const GpuMat& d_descriptors2,
               vector<DMatch>& matches);

    void displayState(Mat& outImg, Size size, double detect_fps, double match_fps, double total_fps);

    vector<Object> objects_;

    bool useGpu_;
    bool showCorrespondences_;
    int curSource_;
    bool fullscreen_;

    SURF cpu_surf_;
    SURF_GPU gpu_surf_;

    BFMatcher cpu_matcher_;
    BFMatcher_GPU gpu_matcher_;
    GpuMat trainIdx_, distance_, allDist_;

    vector< vector<DMatch> > matchesTbl_;
};

App::App() :
    gpu_surf_(500),
    cpu_surf_(500),
    cpu_matcher_(NORM_L2),
    gpu_matcher_(NORM_L2)
{
    useGpu_ = true;
    showCorrespondences_ = true;
    curSource_ = 0;
    fullscreen_ = true;
}

void App::runAppLogic()
{
    if (objects_.empty())
    {
        cout << "Loading default objects... \n" << endl;

        objects_.push_back(Object("opengl", imread("data/features2d_opengl.jpg"), CV_RGB(0, 255, 0)));
        objects_.push_back(Object("java", imread("data/features2d_java.jpg"), CV_RGB(255, 0, 0)));
        objects_.push_back(Object("qt4", imread("data/features2d_qt4.jpg"), CV_RGB(0, 0, 255)));
    }

    if (sources_.empty())
    {
        cout << "Loading default frames source... \n" << endl;

        sources_.push_back(FrameSource::image("data/features2d_1.jpg"));
        sources_.push_back(FrameSource::image("data/features2d_2.jpg"));
        sources_.push_back(FrameSource::image("data/features2d_3.jpg"));
        sources_.push_back(FrameSource::image("data/features2d_4.jpg"));
        sources_.push_back(FrameSource::image("data/features2d_5.jpg"));
        sources_.push_back(FrameSource::image("data/features2d_6.jpg"));
        sources_.push_back(FrameSource::image("data/features2d_7.jpg"));
    }

    Mat frame, frameGray, outImg, frameDescriptors;
    GpuMat d_frameGray, d_frameDescriptors;

    vector<KeyPoint> frameKeypoints;

    vector< vector<DMatch> > matches(objects_.size());

    const string wndName = "Features2D Demo";

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

        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        d_frameGray.upload(frameGray);

        const int64 detect_start = getTickCount();
        {
            for (size_t i = 0; i < objects_.size(); ++i)
                calcKeypoints(objects_[i].imgGray, objects_[i].d_imgGray, objects_[i].keypoints, objects_[i].descriptors, objects_[i].d_descriptors);

            calcKeypoints(frameGray, d_frameGray, frameKeypoints, frameDescriptors, d_frameDescriptors);
        }
        const double detect_fps = getTickFrequency() / (getTickCount() - detect_start);

        const int64 match_start = cv::getTickCount();
        {
            for (size_t i = 0; i < objects_.size(); ++i)
                match(objects_[i].descriptors, objects_[i].d_descriptors, frameDescriptors, d_frameDescriptors, matches[i]);
        }
        const double match_fps = getTickFrequency() / (getTickCount() - match_start);

        const int offset = 350;

        Size outSize = frame.size();
        int max_height = 0;
        int sum_width = offset;
        for (size_t i = 0; i < objects_.size(); ++i)
        {
            sum_width += objects_[i].imgColor.cols;
            max_height = std::max(max_height, objects_[i].imgColor.rows);
        }
        outSize.height += max_height;
        outSize.width = std::max(outSize.width, sum_width);

        outImg.create(outSize, CV_8UC3);
        outImg.setTo(0);
        frame.copyTo(outImg(Rect(0, max_height, frame.cols, frame.rows)));

        int objX = offset;
        for (size_t i = 0; i < objects_.size(); ++i)
        {
            objects_[i].imgColor.copyTo(outImg(Rect(objX, 0, objects_[i].imgColor.cols, objects_[i].imgColor.rows)));

            putText(outImg, objects_[i].name, Point(objX, 15), FONT_HERSHEY_DUPLEX, 0.8, objects_[i].color);

            if (matches[i].size() >= 10)
            {
                static vector<Point2f> pt1;
                static vector<Point2f> pt2;

                pt1.resize(matches[i].size());
                pt2.resize(matches[i].size());

                for (size_t j = 0; j < matches[i].size(); ++j)
                {
                    DMatch m = matches[i][j];

                    KeyPoint objKp = objects_[i].keypoints[m.queryIdx];
                    KeyPoint frameKp = frameKeypoints[m.trainIdx];

                    pt1[j] = objKp.pt;
                    pt2[j] = frameKp.pt;

                    if (showCorrespondences_)
                    {
                        Point objCenter(cvRound(objKp.pt.x) + objX, cvRound(objKp.pt.y));
                        Point frameCenter(cvRound(frameKp.pt.x), cvRound(frameKp.pt.y) + max_height);

                        circle(outImg, objCenter, 3, objects_[i].color);
                        circle(outImg, frameCenter, 3, objects_[i].color);
                        line(outImg, objCenter, frameCenter, objects_[i].color);
                    }
                }

                Mat H = findHomography(pt1, pt2, RANSAC);

                if (H.empty())
                    continue;

                Point src_corners[] =
                {
                    Point(0, 0),
                    Point(objects_[i].imgColor.cols, 0),
                    Point(objects_[i].imgColor.cols, objects_[i].imgColor.rows),
                    Point(0, objects_[i].imgColor.rows)
                };
                Point dst_corners[5];

                for (int j = 0; j < 4; ++j)
                {
                    double x = src_corners[j].x;
                    double y = src_corners[j].y;

                    double Z = 1.0 / (H.at<double>(2, 0) * x + H.at<double>(2, 1) * y + H.at<double>(2, 2));
                    double X = (H.at<double>(0, 0) * x + H.at<double>(0, 1) * y + H.at<double>(0, 2)) * Z;
                    double Y = (H.at<double>(1, 0) * x + H.at<double>(1, 1) * y + H.at<double>(1, 2)) * Z;

                    dst_corners[j] = Point(cvRound(X), cvRound(Y));
                }

                for (int j = 0; j < 4; ++j)
                {
                    Point r1 = dst_corners[j % 4];
                    Point r2 = dst_corners[(j + 1) % 4];

                    line(outImg, Point(r1.x, r1.y + max_height), Point(r2.x, r2.y + max_height), objects_[i].color, 3);
                }

                putText(outImg, objects_[i].name, Point(dst_corners[0].x, dst_corners[0].y + max_height), FONT_HERSHEY_DUPLEX, 0.8, objects_[i].color);
            }

            objX += objects_[i].imgColor.cols;
        }

        double total_fps = getTickFrequency() / (getTickCount() - total_start);

        displayState(outImg, frame.size(), detect_fps, match_fps, total_fps);

        imshow(wndName, outImg);

        wait(30);
    }
}

void App::calcKeypoints(const Mat& img, const GpuMat& d_img, vector<KeyPoint>& keypoints, Mat& descriptors, GpuMat& d_descriptors)
{
    keypoints.clear();

    if (useGpu_)
        gpu_surf_(d_img, GpuMat(), keypoints, d_descriptors);
    else
        cpu_surf_(img, noArray(), keypoints, descriptors);
}

struct DMatchCmp
{
    bool operator ()(const DMatch& m1, const DMatch& m2) const
    {
        return m1.distance < m2.distance;
    }
};

void App::match(const Mat& descriptors1, const GpuMat& d_descriptors1,
                const Mat& descriptors2, const GpuMat& d_descriptors2,
                vector<DMatch>& matches)
{
    matches.clear();

    if (useGpu_)
    {
        gpu_matcher_.knnMatchSingle(d_descriptors1, d_descriptors2, trainIdx_, distance_, allDist_, 2);
        gpu_matcher_.knnMatchDownload(trainIdx_, distance_, matchesTbl_);
    }
    else
    {
        cpu_matcher_.knnMatch(descriptors1, descriptors2, matchesTbl_, 2);
    }

    for (size_t i = 0; i < matchesTbl_.size(); ++i)
    {
        if (matchesTbl_[i].size() != 2)
            continue;

        DMatch m1 = matchesTbl_[i][0];
        DMatch m2 = matchesTbl_[i][1];

        if (m1.distance < 0.55 * m2.distance)
            matches.push_back(m1);
    }

    if (useGpu_)
        sort(matches.begin(), matches.end(), DMatchCmp());
}

void App::displayState(Mat& outImg, Size size, double detect_fps, double match_fps, double total_fps)
{
    const Scalar fontColorRed = CV_RGB(255, 0, 0);

    ostringstream txt;
    int i = 0;

    txt.str(""); txt << "Source size: " << size;
    printText(outImg, txt.str(), i++);

    printText(outImg, useGpu_ ? "Mode: CUDA" : "Mode: CPU", i++);

    txt.str(""); txt << "FPS (Detect only): " << std::fixed << std::setprecision(1) << detect_fps;
    printText(outImg, txt.str(), i++);

    txt.str(""); txt << "FPS (Match only): " << std::fixed << std::setprecision(1) << match_fps;
    printText(outImg, txt.str(), i++);

    txt.str(""); txt << "FPS (Total): " << std::fixed << std::setprecision(1) << total_fps;
    printText(outImg, txt.str(), i++);

    printText(outImg, "Space - switch CUDA / CPU mode", i++, fontColorRed);
    printText(outImg, "S - show / hide correspondences", i++, fontColorRed);
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

    case 'S':
        showCorrespondences_ = !showCorrespondences_;
        if (showCorrespondences_)
            cout << "Show correspondences" << endl;
        else
            cout << "Hide correspondences" << endl;
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
    cout << "This sample demonstrates Object Detection via keypoints matching \n" << endl;

    cout << "Usage: demo_features2d [options] \n" << endl;

    cout << "Demo Options: \n"
         << "  --object <path to image> \n"
         << "       Object image \n" << endl;

    cout << "Launch Options: \n"
         << "  --windowed \n"
         << "       Launch in windowed mode\n" << endl;
}

bool App::parseAppCmdArgs(int& i, int argc, const char* argv[])
{
    string arg = argv[i];

    if (arg == "--object")
    {
        ++i;

        if (i >= argc)
            THROW_EXCEPTION("Missing file name after " << arg);

        RNG& rng = theRNG();
        objects_.push_back(Object(argv[i], imread(argv[i]), CV_RGB(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255))));
    }
    else if (arg == "--windowed")
    {
        fullscreen_ = false;
        return true;
    }
    else
        return false;

    return true;
}

RUN_APP(App)
